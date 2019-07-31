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

import argparse
import json
import logging
import os
import sys

import mxnet as mx
import gluonnlp as nlp
from gluonnlp.model import BERTEncoder, BERTModel
from gluonnlp.model.bert import bert_hparams

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))

from utils import (get_hash, load_text_vocab, read_tf_checkpoint,
                   tf_vocab_to_gluon_vocab)


parser = argparse.ArgumentParser(
    description='Conversion script for Tensorflow BERT model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model',
                    type=str,
                    default='bert_12_768_12',
                    choices=['bert_12_768_12', 'bert_24_1024_16'],
                    help='BERT model name')
parser.add_argument('--tf_checkpoint_dir',
                    type=str,
                    help='Path to Tensorflow checkpoint folder.')
parser.add_argument('--tf_model_prefix', type=str,
                    default='bert_model.ckpt',
                    help='name of bert checkpoint file.')
parser.add_argument('--tf_config_name', type=str,
                    default='bert_config.json',
                    help='Name of Bert config file')
parser.add_argument('--out_dir',
                    type=str,
                    default=os.path.join('~', 'output'),
                    help='Path to output folder. The folder must exist.')
parser.add_argument('--debug', action='store_true', help='debugging mode')
args = parser.parse_args()
logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)
logging.info(args)

# convert vocabulary
vocab_path = os.path.join(args.tf_checkpoint_dir, 'vocab.txt')
vocab = tf_vocab_to_gluon_vocab(load_text_vocab(vocab_path))

# vocab serialization
out_dir = os.path.expanduser(args.out_dir)
nlp.utils.mkdir(out_dir)
tmp_file_path = os.path.join(out_dir, 'tmp')
with open(tmp_file_path, 'w') as f:
    f.write(vocab.to_json())
hash_full, hash_short = get_hash(tmp_file_path)
gluon_vocab_path = os.path.join(out_dir, hash_short + '.vocab')
with open(gluon_vocab_path, 'w') as f:
    f.write(vocab.to_json())
    logging.info('vocab file saved to %s. hash = %s', gluon_vocab_path, hash_full)

# load tf model
tf_checkpoint_file = os.path.expanduser(
    os.path.join(args.tf_checkpoint_dir, args.tf_model_prefix))
logging.info('loading Tensorflow checkpoint %s ...', tf_checkpoint_file)
tf_tensors = read_tf_checkpoint(tf_checkpoint_file)
tf_names = sorted(tf_tensors.keys())

tf_names = filter(lambda name: not name.endswith('adam_m'), tf_names)
tf_names = filter(lambda name: not name.endswith('adam_v'), tf_names)
tf_names = filter(lambda name: name != 'global_step', tf_names)
tf_names = list(tf_names)
if len(tf_tensors) != len(tf_names):
    logging.info('Tensorflow model was saved with Optimizer parameters. '
                 'Ignoring them.')

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

# BERT config
tf_config_names_to_gluon_config_names = {
    'attention_probs_dropout_prob': 'embed_dropout',
    'hidden_act': None,
    'hidden_dropout_prob': 'dropout',
    'hidden_size': 'units',
    'initializer_range': None,
    'intermediate_size': 'hidden_size',
    'max_position_embeddings': 'max_length',
    'num_attention_heads': 'num_heads',
    'num_hidden_layers': 'num_layers',
    'type_vocab_size': 'token_type_vocab_size',
    'vocab_size': None
}
predefined_args = bert_hparams[args.model]
with open(os.path.join(args.tf_checkpoint_dir, args.tf_config_name), 'r') as f:
    tf_config = json.load(f)
    assert len(tf_config) == len(tf_config_names_to_gluon_config_names)
    for tf_name, gluon_name in tf_config_names_to_gluon_config_names.items():
        if tf_name is None or gluon_name is None:
            continue
        assert tf_config[tf_name] == predefined_args[gluon_name]

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

# Infer enabled BERTModel components
use_pooler = any('pooler' in n for n in mx_tensors)
use_decoder = any('decoder.0' in n for n in mx_tensors)
use_classifier = any('classifier.weight' in n for n in mx_tensors)

logging.info('Inferred that the tensorflow model provides the following parameters:')
logging.info('- use_pooler = {}'.format(use_pooler))
logging.info('- use_decoder = {}'.format(use_decoder))
logging.info('- use_classifier = {}'.format(use_classifier))

# post processings for parameters:
# - handle tied decoder weight
logging.info('total number of tf parameters = %d', len(tf_names))
if use_decoder:
    mx_tensors['decoder.3.weight'] = mx_tensors['word_embed.0.weight']
    logging.info('total number of mx parameters = %d'
                 '(including decoder param for weight tying)', len(mx_tensors))
else:
    logging.info('total number of mx parameters = %d', len(mx_tensors))

# BERT model
bert = BERTModel(encoder, len(vocab),
                 token_type_vocab_size=predefined_args['token_type_vocab_size'],
                 units=predefined_args['units'],
                 embed_size=predefined_args['embed_size'],
                 embed_dropout=predefined_args['embed_dropout'],
                 word_embed=predefined_args['word_embed'],
                 use_pooler=use_pooler, use_decoder=use_decoder,
                 use_classifier=use_classifier)

bert.initialize(init=mx.init.Normal(0.02))

ones = mx.nd.ones((2, 8))
out = bert(ones, ones, mx.nd.array([5, 6]), mx.nd.array([[1], [2]]))
params = bert._collect_params_with_prefix()
if len(params) != len(mx_tensors):
    raise RuntimeError('The Gluon BERTModel comprises {} parameter arrays, '
                       'but {} have been extracted from the tf model. '
                       'Most likely the BERTModel hyperparameters do not match '
                       'the hyperparameters of the tf model.'.format(len(params), len(mx_tensors)))

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
gluon_param_path = os.path.join(out_dir, hash_short + '.params')
logging.info('param saved to %s. hash = %s', gluon_param_path, hash_full)
bert.save_parameters(gluon_param_path)
mx.nd.waitall()
