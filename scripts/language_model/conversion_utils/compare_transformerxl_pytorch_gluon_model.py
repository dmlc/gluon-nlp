# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# 'License'); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Script for model comparison between TF and Gluon."""

import argparse
import glob
import logging
import os
import pickle
import re
import sys

import mxnet as mx
import numpy as np
import tensorflow as tf
import torch
from absl import flags

import gluonnlp as nlp
import pytorch_transformers
from utils import read_tf_checkpoint, to_gluon_kwargs


def get_kwargs_and_corpus(args):
    # Infer model config
    with open(os.path.join(args.tf_data_dir, 'cache.pkl'), 'rb') as f:
        corpus = pickle.load(f, encoding='latin1')
    tf_checkpoint_file = os.path.expanduser(
        os.path.join(args.tf_checkpoint_dir, args.tf_model_prefix))
    tf_tensors = read_tf_checkpoint(tf_checkpoint_file)
    return to_gluon_kwargs(tf_tensors), corpus


def get_data(args):
    record_info_dir = os.path.join(args.tf_data_dir, 'tfrecords')
    assert os.path.exists(record_info_dir)
    record_info_file = glob.glob(os.path.join(record_info_dir, "record_info*json"))[0]
    eval_split, batch_size, tgt_len = re.search(r'record_info-(\w+)\.bsz-(\d+)\.tlen-(\d+).json',
                                                record_info_file).groups()
    batch_size, tgt_len = int(batch_size), int(tgt_len)

    num_core_per_host = 1
    num_hosts = 1
    eval_input_fn, eval_record_info = data_utils.get_input_fn(
        record_info_dir=record_info_dir, split=eval_split, per_host_bsz=batch_size, tgt_len=tgt_len,
        num_core_per_host=num_core_per_host, num_hosts=num_hosts, use_tpu=False)

    ##### Create computational graph
    eval_set = eval_input_fn({"batch_size": batch_size, "data_dir": record_info_dir})
    input_feed, label_feed = eval_set.make_one_shot_iterator().get_next()

    # Extract first two batches
    sess = tf.Session()
    np_features, np_labels = [], []
    for i in range(2):
        feature_i, label_i = sess.run((input_feed, label_feed))
        np_features.append(feature_i[:1])  # force batch_size of 1
        np_labels.append(label_i[:1])

    return np_features, np_labels, 1, tgt_len


def compare_transformerxl(args, kwargs, corpus):
    # Data
    np_features, np_labels, batch_size, tgt_len = get_data(args)

    # Models
    model_p = pytorch_transformers.TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')
    model_p.crit.keep_order = True
    model_p.transformer.output_attentions = False  # no change of default; breaks model if changed
    model_p.transformer.output_hidden_states = True

    with open(args.gluon_vocab_file, 'r') as f:
        vocab = nlp.Vocab.from_json(f.read())
    ctx = mx.gpu()
    model = TransformerXL(vocab_size=len(vocab), clamp_len=model_p.transformer.clamp_len, **kwargs)
    model.initialize(ctx=ctx)
    model.load_parameters(args.gluon_parameter_file, ignore_extra=False)
    model.hybridize()

    # Computation
    assert len(np_features) == 2
    mems = model.begin_mems(batch_size, model_p.config.mem_len, context=ctx)
    mems_p = None
    for batch in range(2):
        print('Batch {}'.format(batch))

        features_nd = mx.nd.array(np_features[batch], ctx=ctx)
        labels_nd = mx.nd.array(np_labels[batch], ctx=ctx)
        features_p = torch.tensor(np_features[batch], dtype=torch.long)
        labels_p = torch.tensor(np_labels[batch], dtype=torch.long)

        loss, mems, last_hidden = model(features_nd, labels_nd, mems)

        loss_p, _, mems_p, all_hidden_p = model_p(features_p, labels_p, mems_p)

        for i in range(kwargs['num_layers']):
            a_b = mems_p[i][:, 0].numpy() - mems[i][0].asnumpy()
            max_error = a_b.max()
            argmax_error = a_b.argmax()
            stdev = np.std(a_b)
            print('Layer {i}: Maximum error {err:.2e} at position {pos}. stdev={stdev:.2e}'.format(
                i=i, err=max_error, pos=np.unravel_index(argmax_error, shape=a_b.shape),
                stdev=stdev))
        a_b = loss_p.detach().numpy()[0] - loss.asnumpy()[0]
        max_error = a_b.max()
        argmax_error = a_b.argmax()
        stdev = np.std(a_b)
        print('Loss: Maximum error {err:.2e} at position {pos}. stdev={stdev:.2e}'.format(
            i=i, err=max_error, pos=np.unravel_index(argmax_error, shape=a_b.shape), stdev=stdev))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Comparison script for Tensorflow and GLuon Transformer-XL model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--transformer-xl-repo', type=str, required=True,
                        help='Path to https://github.com/kimiyoung/transformer-xl repo.')
    parser.add_argument('--tf-checkpoint-dir', type=str, required=True,
                        help='Path to Tensorflow checkpoint folder.')
    parser.add_argument(
        '--tf-model-prefix', type=str, required=True, help='Prefix of the checkpoint files. '
        'For example model.ckpt-0 or model.ckpt-1191000')
    parser.add_argument(
        '--tf-data-dir', type=str, required=True, help='Path to TransformerXL data folder. '
        'The folder should contain the tfrecords directory as well as the cache.pkl file. '
        'tfrecords can be created with the TransformerXL data_utils.py script.')
    parser.add_argument('--gluon-parameter-file', type=str, required=True,
                        help='gluon parameter file name.')
    parser.add_argument('--gluon-vocab-file', type=str, required=True,
                        help='gluon vocab file corresponding to --gluon_parameter_file.')
    parser.add_argument('--debug', action='store_true', help='debugging mode')
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)
    logging.info(args)

    # Load stuff required for unpickling
    sys.path.append(os.path.join((args.transformer_xl_repo), 'tf'))
    import vocabulary  # pylint: disable=unused-import
    import data_utils

    # Infer correct tf flags
    kwargs, corpus = get_kwargs_and_corpus(args)
    tf_argv = [
        'train.py',
        '--n_layer=' + str(kwargs['num_layers']),
        '--d_model=' + str(kwargs['units']),
        '--d_embed=' + str(kwargs['embed_size']),
        '--n_head=' + str(kwargs['num_heads']),
        '--d_head=' + str(kwargs['units'] // kwargs['num_heads']),
        '--d_inner=' + str(kwargs['hidden_size']),
        '--dropout=0.0',
        '--dropatt=0.0',
        '--same_length=True',
        '--model_dir=' + args.tf_checkpoint_dir,
        '--proj_share_all_but_first=True',
        '--untie_r=True',
        '--div_val=' + str(kwargs['embed_div_val']),
    ]
    tf_flags = flags.FLAGS(tf_argv, known_only=True)

    sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
    from transformer import TransformerXL

    compare_transformerxl(args, kwargs, corpus)
