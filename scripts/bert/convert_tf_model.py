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
# pylint:disable=redefined-outer-name,logging-format-interpolation
""" Script for converting TF Model to Gluon. """

import os
import logging
import argparse
import mxnet as mx
from gluonnlp.model import BERTEncoder, BERTModel
from gluonnlp.model.bert import bert_hparams
from utils import convert_vocab, get_hash, read_tf_checkpoint

parser = argparse.ArgumentParser(description='Conversion script for Tensorflow BERT model')
parser.add_argument('--model', type=str, default='bert_12_768_12',
                    help='BERT model name. options are bert_12_768_12 and bert_24_1024_16.'
                         'Default is bert_12_768_12')
parser.add_argument('--tf_checkpoint_dir', type=str,
                    default=os.path.join('~', 'cased_L-12_H-768_A-12/'),
                    help='Path to Tensorflow checkpoint folder. '
                         'Default is /home/ubuntu/cased_L-12_H-768_A-12/')
parser.add_argument('--out_dir', type=str,
                    default=os.path.join('~', 'output'),
                    help='Path to output folder. The folder must exist. '
                         'Default is /home/ubuntu/output/')
parser.add_argument('--debug', action='store_true', help='debugging mode')
args = parser.parse_args()
logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)
logging.info(args)

# convert vocabulary
vocab_path = os.path.join(args.tf_checkpoint_dir, 'vocab.txt')
vocab, reserved_token_idx_map = convert_vocab(vocab_path)

# vocab serialization
tmp_file_path = os.path.expanduser(os.path.join(args.out_dir, 'tmp'))
with open(tmp_file_path, 'w') as f:
    f.write(vocab.to_json())
hash_full, hash_short = get_hash(tmp_file_path)
gluon_vocab_path = os.path.expanduser(os.path.join(args.out_dir, hash_short + '.vocab'))
with open(gluon_vocab_path, 'w') as f:
    f.write(vocab.to_json())
    logging.info('vocab file saved to %s. hash = %s', gluon_vocab_path, hash_full)

# load tf model
tf_checkpoint_file = os.path.expanduser(
    os.path.join(args.tf_checkpoint_dir, 'bert_model.ckpt'))
logging.info('loading Tensorflow checkpoint %s ...', tf_checkpoint_file)
tf_tensors = read_tf_checkpoint(tf_checkpoint_file)
tf_names = sorted(tf_tensors.keys())
for name in tf_names:
    logging.debug('%s: %s', name, tf_tensors[name].shape)

# replace tensorflow parameter names with gluon parameter names
NAME_MAP = [
    ('bert/encoder/layer_', 'encoder.transformer_cells.'),
    ('/attention/self/', '.attention_cell.'),
    ('key', 'proj_key'),
    ('query', 'proj_query'),
    ('value', 'proj_value'),
    ('/attention/output/LayerNorm/', '.layer_norm.'),
    ('/attention/output/dense/', '.proj.'),
    ('cls/seq_relationship/output_weights', 'classifier.weight'),
    ('cls/seq_relationship/output_bias', 'classifier.bias'),
    ('cls/predictions/output_bias', 'decoder.3.bias'),
    ('cls/predictions/transform/dense/', 'decoder.0.'),
    ('cls/predictions/transform/LayerNorm/', 'decoder.2.'),
    ('kernel', 'weight'),
    ('/intermediate/dense/', '.ffn.ffn_1.'),
    ('/output/dense/', '.ffn.ffn_2.'),
    ('/output/LayerNorm/', '.ffn.layer_norm.'),
    ('bert/embeddings/LayerNorm/', 'encoder.layer_norm.'),
    ('bert/embeddings/position_embeddings', 'encoder.position_weight'),
    ('bert/embeddings/token_type_embeddings', 'token_type_embed.0.weight'),
    ('bert/embeddings/word_embeddings', 'word_embed.0.weight'),
    ('bert/pooler/dense/', 'pooler.'),
    ('/', '.'),
]

# convert to gluon parameters
mx_tensors = {}
logging.info('converting to Gluon checkpoint ... ')
for source_name in tf_names:
    # get the source tensor and its transpose
    source, source_t = tf_tensors[source_name], tf_tensors[source_name].T
    target, target_name = source, source_name
    for old, new in NAME_MAP:
        target_name = target_name.replace(old, new)
    # transpose kernel layer parameters
    if 'kernel' in source_name:
        target = source_t
    mx_tensors[target_name] = target
    if source_t.shape == source.shape and len(source.shape) > 1 and target is not source_t:
        logging.info('warning: %s has symmetric shape %s', target_name, target.shape)
    logging.debug('%s: %s', target_name, target.shape)

# post processings for parameters:
# - handle tied decoder weight
# - update word embedding for reserved tokens
mx_tensors['decoder.3.weight'] = mx_tensors['word_embed.0.weight']
embedding = mx_tensors['word_embed.0.weight']
for source_idx, dst_idx in reserved_token_idx_map:
    source = embedding[source_idx].copy()
    dst = embedding[dst_idx].copy()
    embedding[source_idx][:] = dst
    embedding[dst_idx][:] = source
logging.info('total number of tf parameters = %d', len(tf_tensors))
logging.info('total number of mx parameters = %d (including decoder param for weight tying)',
             len(mx_tensors))

# XXX assume no changes in BERT configs
predefined_args = bert_hparams[args.model]

# BERT encoder
encoder = BERTEncoder(attention_cell=predefined_args['attention_cell'],
                      num_layers=predefined_args['num_layers'],
                      units=predefined_args['units'],
                      hidden_size=predefined_args['hidden_size'],
                      max_length=predefined_args['max_length'],
                      num_heads=predefined_args['num_heads'],
                      scaled=predefined_args['scaled'],
                      dropout=predefined_args['dropout'],
                      use_residual=predefined_args['use_residual'])

# BERT model
bert = BERTModel(encoder, len(vocab),
                 token_type_vocab_size=predefined_args['token_type_vocab_size'],
                 units=predefined_args['units'],
                 embed_size=predefined_args['embed_size'],
                 embed_dropout=predefined_args['embed_dropout'],
                 word_embed=predefined_args['word_embed'],
                 use_pooler=True, use_decoder=True,
                 use_classifier=True)

bert.initialize(init=mx.init.Normal(0.02))

ones = mx.nd.ones((2, 8))
out = bert(ones, ones, mx.nd.array([1, 2]))
params = bert._collect_params_with_prefix()

# set parameter data
loaded_params = {}
for name in params:
    try:
        arr = mx.nd.array(mx_tensors[name])
        params[name].set_data(arr)
        loaded_params[name] = True
    # pylint: disable=broad-except
    except Exception:
        if name not in mx_tensors:
            raise RuntimeError('cannot initialize %s from tf checkpoint' % name)
        else:
            raise RuntimeError('cannot initialize %s. Expect shape = %s, but found %s' %
                               name, params[name].shape, arr.shape)

logging.info('num loaded params = %d, total num params = %d',
             len(loaded_params), len(mx_tensors))
for name in mx_tensors:
    if name not in loaded_params:
        logging.info('%s is not loaded', name)

# param serialization
bert.save_parameters(tmp_file_path)
hash_full, hash_short = get_hash(tmp_file_path)
gluon_param_path = os.path.expanduser(os.path.join(args.out_dir, hash_short + '.params'))
logging.info('param saved to %s. hash = %s', gluon_param_path, hash_full)
bert.save_parameters(gluon_param_path)
mx.nd.waitall()
