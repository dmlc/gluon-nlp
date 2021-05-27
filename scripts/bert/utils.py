# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Utility functions for BERT."""

import sys
import logging
import collections
import hashlib
import io
import os
from tempfile import TemporaryDirectory

import mxnet as mx
import gluonnlp as nlp

__all__ = ['tf_vocab_to_gluon_vocab', 'load_text_vocab']


def tf_vocab_to_gluon_vocab(tf_vocab):
    special_tokens = ['[UNK]', '[PAD]', '[SEP]', '[MASK]', '[CLS]']
    assert all(t in tf_vocab for t in special_tokens)
    counter = nlp.data.count_tokens(tf_vocab.keys())
    vocab = nlp.vocab.BERTVocab(counter, token_to_idx=tf_vocab)
    return vocab


def get_hash(filename):
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest(), str(sha1.hexdigest())[:8]


def read_tf_checkpoint(path):
    """read tensorflow checkpoint"""
    from tensorflow.python import pywrap_tensorflow  # pylint: disable=import-outside-toplevel
    tensors = {}
    reader = pywrap_tensorflow.NewCheckpointReader(path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        tensor = reader.get_tensor(key)
        tensors[key] = tensor
    return tensors

def profile(curr_step, start_step, end_step, profile_name='profile.json',
            early_exit=True):
    """profile the program between [start_step, end_step)."""
    if curr_step == start_step:
        mx.nd.waitall()
        mx.profiler.set_config(profile_memory=False, profile_symbolic=True,
                               profile_imperative=True, filename=profile_name,
                               aggregate_stats=True)
        mx.profiler.set_state('run')
    elif curr_step == end_step:
        mx.nd.waitall()
        mx.profiler.set_state('stop')
        logging.info(mx.profiler.dumps())
        mx.profiler.dump()
        if early_exit:
            sys.exit(0)

def load_text_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with io.open(vocab_file, 'r') as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

class QuantizableNet(mx.gluon.nn.HybridBlock):
    """
    While quantizing SymbolBlock with incorrect number of inputs in
    calibration dataloader ValueError exception is triggered instead of
    TypeError - this class is workaround for such case
    """
    def __init__(self, original_net, **kwargs):
        super(QuantizableNet, self).__init__(**kwargs)
        self.original_net = original_net

    def hybrid_forward(self, F, data0, data1, data2):
        return self.original_net(data0, data1, data2)

class QuantizableRobertaNet(mx.gluon.nn.HybridBlock):
    """
    While quantizing SymbolBlock with incorrect number of inputs in
    calibration dataloader ValueError exception is triggered instead of
    TypeError - this class is workaround for such case
    """
    def __init__(self, original_net, **kwargs):
        super(QuantizableRobertaNet, self).__init__(**kwargs)
        self.original_net = original_net

    def hybrid_forward(self, F, data0, data1):
        return self.original_net(data0, data1)

def run_graphpass(net, model_name, batch_size, seq_len, pass_name, use_roberta=False):
    data0 = mx.nd.random.uniform(shape=(batch_size, seq_len))
    data1 = mx.nd.random.uniform(shape=(batch_size, seq_len))
    data2 = mx.nd.random.uniform(shape=(batch_size,))
    net.hybridize()
    if use_roberta:
        net(data0, data2)
    else:
        net(data0, data1, data2)

    with TemporaryDirectory() as tmpdirname:
        tmp_name = 'tmp'
        prefix = os.path.join(tmpdirname, tmp_name)
        net.export(prefix, epoch=0)
        assert os.path.isfile(prefix + '-symbol.json')
        assert os.path.isfile(prefix + '-0000.params')

        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, 0)
        if use_roberta:
            arg_params['data0'] = data0
            arg_params['data1'] = data2
        else:
            arg_params['data0'] = data0
            arg_params['data1'] = data1
            arg_params['data2'] = data2

        custom_sym = sym.optimize_for(pass_name, arg_params, aux_params)
        if (pass_name == 'MHAInterleave' and mx.__version__ <= '1.7.0'):
            nheads = 12
            if model_name == 'bert_24_1024_16':
                nheads = 24
            for i in range(nheads):
                basename = 'bertencoder0_transformer' + str(i) + '_dotproductselfattentioncell0'
                arg_params.pop(basename + '_query_weight')
                arg_params.pop(basename + '_key_weight')
                arg_params.pop(basename + '_value_weight')
                arg_params.pop(basename + '_query_bias')
                arg_params.pop(basename + '_key_bias')
                arg_params.pop(basename + '_value_bias')
        if use_roberta:
            arg_params.pop('data0')
            arg_params.pop('data1')
        else:
            arg_params.pop('data0')
            arg_params.pop('data1')
            arg_params.pop('data2')

        mx.model.save_checkpoint(prefix, 0, custom_sym, arg_params, aux_params)
        mx.nd.waitall()
        import_input_names = ["data0", "data1"]
        if not use_roberta:
            import_input_names += ["data2"]
        net = mx.gluon.SymbolBlock.imports(prefix + "-symbol.json", import_input_names, prefix + "-0000.params")

    return net

class RobertaCalibIter(mx.io.DataIter):
    """
    DataIter wrapper for dataloader used with BERT - quantization
    can handle more inputs to network than required, but not in
    different order - standard BERT inputs are:
    input_ids, segment_ids, valid_length, and for Roberta:
    input_ids, valid_length
    """
    def __init__(self, calib_data):
        self._data = calib_data
        calib_iter = iter(calib_data)
        data_example = next(calib_iter)
        num_data = len(data_example)
        assert num_data > 0

        self.provide_data = [mx.io.DataDesc(name='data0', shape=(data_example[0].shape))]
        self.provide_data +=  [mx.io.DataDesc(name='data1', shape=(data_example[2].shape))]

        self.batch_size = data_example[0].shape[0]
        self.reset()

    def reset(self):
        self._iter = iter(self._data)

    def next(self):
        next_data = next(self._iter)
        next_data = [next_data[0], next_data[2]]
        return mx.io.DataBatch(data=next_data)