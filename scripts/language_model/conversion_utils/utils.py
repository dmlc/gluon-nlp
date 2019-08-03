# coding: utf-8

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

import hashlib
import itertools
import re


def _filter_dict(func, dictionary):
    return {k: v for k, v in dictionary.items() if func(k, v)}


def _split_dict(func, dictionary):
    part_one = _filter_dict(func, dictionary)
    part_two = _filter_dict(lambda *args: not func(*args), dictionary)
    return part_one, part_two


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
    from tensorflow.python import pywrap_tensorflow
    tensors = {}
    reader = pywrap_tensorflow.NewCheckpointReader(path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        tensor = reader.get_tensor(key)
        tensors[key] = tensor
    return tensors


def to_gluon_kwargs(tf_tensors):
    kwargs = dict()

    # Main model
    kwargs['num_layers'] = len(
        set(itertools.chain.from_iterable(re.findall(r'layer_\d*', k) for k in tf_tensors)))
    kwargs['hidden_size'] = tf_tensors['transformer/layer_0/ff/layer_2/kernel'].shape[0]
    kwargs['units'] = tf_tensors['transformer/layer_0/ff/layer_2/kernel'].shape[1]
    tie_r = len(tf_tensors['transformer/r_w_bias'].shape) != 3
    kwargs['num_heads'] = tf_tensors['transformer/r_w_bias'].shape[0 if tie_r else 1]

    # Embedding and softmax
    if 'transformer/adaptive_embed/lookup_table' in tf_tensors:
        # Adaptive embedding is not used
        kwargs['embed_size'] = tf_tensors['transformer/adaptive_embed/lookup_table'].shape[1]
        kwargs['tie_input_output_embeddings'] = \
            'transformer/adaptive_softmax/lookup_table' not in tf_tensors
        kwargs['tie_input_output_projections'] = \
            ['transformer/adaptive_softmax/proj' not in tf_tensors]
    else:
        # Adaptive embedding is used
        lookup_table_selector = 'transformer/adaptive_embed/cutoff_{i}/lookup_table'
        kwargs['embed_cutoffs'] = list(
            itertools.accumulate([
                tf_tensors[lookup_table_selector.format(i=i)].shape[0] for i in range(
                    len(_filter_dict(lambda k, v: k.endswith('lookup_table'), tf_tensors)))
            ][:-1]))
        kwargs['embed_size'] = tf_tensors[lookup_table_selector.format(i=0)].shape[1]
        size_of_second = tf_tensors[lookup_table_selector.format(i=1)].shape[1]
        kwargs['embed_div_val'] = kwargs['embed_size'] // size_of_second
        assert kwargs['embed_size'] % size_of_second == 0
        kwargs['tie_input_output_embeddings'] = not bool(
            _filter_dict(
                lambda k, v: k.startswith('transformer/adaptive_softmax/cutoff_') and k.endswith(
                    'lookup_table'), tf_tensors))
        proj_selector = 'transformer/adaptive_softmax/cutoff_{i}/proj'
        kwargs['tie_input_output_projections'] = [
            proj_selector.format(i=i) not in tf_tensors
            for i in range(len(kwargs['embed_cutoffs']) + 1)
        ]
        if kwargs['embed_size'] == kwargs['embed_size'] and \
           'transformer/adaptive_embed/cutoff_0/proj_W' not in tf_tensors:
            kwargs['project_same_dim'] = False

    # Dropout
    # All pre-trained TransformerXL models from
    # https://github.com/kimiyoung/transformer-xl come without dropout
    kwargs['dropout'] = 0
    kwargs['attention_dropout'] = 0

    print(kwargs)
    return kwargs, tie_r
