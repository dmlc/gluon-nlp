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
""" Script for converting PyTorch Model to Gluon. """

import argparse
import json
import logging
import os
import sys

import mxnet as mx
import gluonnlp as nlp
import torch
from gluonnlp.model import BERTEncoder, BERTModel
from gluonnlp.model.bert import bert_hparams

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
from utils import get_hash, load_text_vocab, tf_vocab_to_gluon_vocab

parser = argparse.ArgumentParser(description='Conversion script for PyTorch BERT model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='bert_12_768_12',
                    choices=['bert_12_768_12', 'bert_24_1024_16'], help='BERT model name')
parser.add_argument('--pytorch_checkpoint_dir', type=str,
                    help='Path to Tensorflow checkpoint folder.')
parser.add_argument('--vocab_file', type=str, help='Full path to the vocab.txt')
parser.add_argument('--gluon_pytorch_name_mapping', type=str,
                    default='gluon_to_pytorch_naming.json',
                    help='Output of infer_pytorch_gluon_parameter_name_mapping.py')
parser.add_argument('--out_dir', type=str, default=os.path.join('~', 'output'),
                    help='Path to output folder. The folder must exist.')
parser.add_argument('--debug', action='store_true', help='debugging mode')
args = parser.parse_args()
logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)
logging.info(args)

# convert vocabulary
vocab = tf_vocab_to_gluon_vocab(load_text_vocab(args.vocab_file))

# vocab serialization
tmp_file_path = os.path.expanduser(os.path.join(args.out_dir, 'tmp'))
with open(tmp_file_path, 'w') as f:
    f.write(vocab.to_json())
hash_full, hash_short = get_hash(tmp_file_path)
gluon_vocab_path = os.path.expanduser(os.path.join(args.out_dir, hash_short + '.vocab'))
with open(gluon_vocab_path, 'w') as f:
    f.write(vocab.to_json())
    logging.info('vocab file saved to %s. hash = %s', gluon_vocab_path, hash_full)

# Load PyTorch Model
pytorch_parameters = torch.load(os.path.join(args.pytorch_checkpoint_dir, 'pytorch_model.bin'),
                                map_location=lambda storage, loc: storage)
pytorch_parameters = {k: v.numpy() for k, v in pytorch_parameters.items()}

# Make sure vocab fits to model
assert pytorch_parameters['bert.embeddings.word_embeddings.weight'].shape[0] == len(
    vocab.idx_to_token)

# Load Mapping
with open(args.gluon_pytorch_name_mapping, 'r') as f:
    mapping = json.load(f)

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
with open(os.path.join(args.pytorch_checkpoint_dir, 'bert_config.json'), 'r') as f:
    tf_config = json.load(f)
    assert len(tf_config) == len(tf_config_names_to_gluon_config_names)
    for tf_name, gluon_name in tf_config_names_to_gluon_config_names.items():
        if tf_name is None or gluon_name is None:
            continue
        assert tf_config[tf_name] == predefined_args[gluon_name]

# BERT encoder
encoder = BERTEncoder(attention_cell=predefined_args['attention_cell'],
                      num_layers=predefined_args['num_layers'], units=predefined_args['units'],
                      hidden_size=predefined_args['hidden_size'],
                      max_length=predefined_args['max_length'],
                      num_heads=predefined_args['num_heads'], scaled=predefined_args['scaled'],
                      dropout=predefined_args['dropout'],
                      use_residual=predefined_args['use_residual'])

# Infer enabled BERTModel components
use_pooler = any('pooler' in n for n in pytorch_parameters)
use_decoder = any('cls.predictions.transform.dense.weight' in n for n in pytorch_parameters)
use_classifier = any('cls.seq_relationship.weight' in n for n in pytorch_parameters)

if not use_classifier and 'classifier.weight' in pytorch_parameters and \
   pytorch_parameters['classifier.weight'].shape[0] == 2:
    logging.info('Assuming classifier weights in provided Pytorch model are '
                 'from next sentence prediction task.')
    use_classifier = True

logging.info('Inferred that the pytorch model provides the following parameters:')
logging.info('- use_pooler = {}'.format(use_pooler))
logging.info('- use_decoder = {}'.format(use_decoder))
logging.info('- use_classifier = {}'.format(use_classifier))

# BERT model
bert = BERTModel(encoder, len(vocab),
                 token_type_vocab_size=predefined_args['token_type_vocab_size'],
                 units=predefined_args['units'], embed_size=predefined_args['embed_size'],
                 embed_dropout=predefined_args['embed_dropout'],
                 word_embed=predefined_args['word_embed'], use_pooler=use_pooler,
                 use_decoder=use_decoder, use_classifier=use_classifier)

bert.initialize(init=mx.init.Normal(0.02))

ones = mx.nd.ones((2, 8))
out = bert(ones, ones, mx.nd.array([5, 6]), mx.nd.array([[1], [2]]))
params = bert._collect_params_with_prefix()
assert len(params) == len(pytorch_parameters), "Gluon model does not match PyTorch model. " \
    "Please fix the BERTModel hyperparameters"

# set parameter data
loaded_params = {}
for name in params:
    if name not in mapping:
        raise RuntimeError('Invalid json mapping file. '
                           'The parameter {} is not described in the mapping file.'.format(name))
    pytorch_name = mapping[name]
    if pytorch_name not in pytorch_parameters.keys():
        # Handle inconsistent naming in PyTorch
        # The Expected names here are based on the PyTorch version of SciBert.
        # The Inconsistencies were found in ClinicalBert
        if 'LayerNorm' in pytorch_name:
            pytorch_name = pytorch_name.replace('weight', 'gamma')
            pytorch_name = pytorch_name.replace('bias', 'beta')
            assert pytorch_name in pytorch_parameters.keys()

        if 'cls.seq_relationship' in pytorch_name:
            pytorch_name = pytorch_name.replace('cls.seq_relationship', 'classifier')

    arr = mx.nd.array(pytorch_parameters[pytorch_name])

    assert arr.shape == params[name].shape
    params[name].set_data(arr)
    loaded_params[name] = True

if len(params) != len(loaded_params):
    raise RuntimeError('The Gluon BERTModel comprises {} parameter arrays, '
                       'but {} have been extracted from the pytorch model. '.format(
                           len(params), len(loaded_params)))

# param serialization
bert.save_parameters(tmp_file_path)
hash_full, hash_short = get_hash(tmp_file_path)
gluon_param_path = os.path.expanduser(os.path.join(args.out_dir, hash_short + '.params'))
logging.info('param saved to %s. hash = %s', gluon_param_path, hash_full)
bert.save_parameters(gluon_param_path)
mx.nd.waitall()
