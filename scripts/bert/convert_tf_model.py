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
# pylint:disable=redefined-outer-name,logging-format-interpolation

import os
import sys
import pickle
import logging
import argparse
import numpy as np
import mxnet as mx
from mxnet import gluon
from tensorflow.python import pywrap_tensorflow
from gluonnlp.model import get_model, bert_12_768_12, bert_24_1024_16, BERTEncoder, BERTModel
from gluonnlp.model.bert import bert_hparams
from utils import convert_vocab, get_hash

parser = argparse.ArgumentParser(description='Conversion script for Tensorflow BERT model')
parser.add_argument('--model', type=str, default='bert_12_768_12',
                    help='BERT model name. options are bert_12_768_12 and bert_24_1024_16')
parser.add_argument('--tf_checkpoint_dir', type=str, required=True,
                    help='Path to Tensorflow checkpoint folder. e.g. /home/ubuntu/cased_L-12_H-768_A-12/')
parser.add_argument('--out_dir', type=str, required=True,
                    help='Path to output folder. e.g. /home/ubuntu/output/')
args = parser.parse_args()
logging.info(args)

###############################################################################
#                           Load Tensorflow Checkpoint                        #
###############################################################################

# read tensorflow checkpoint
def read_tf_checkpoint(path):
    print('================= Loading Tensorflow checkpoint %s ... ================='%path)
    tensors = {}
    reader = pywrap_tensorflow.NewCheckpointReader(path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        tensor = reader.get_tensor(key)
        tensors[key] = tensor
        print('%s: %s'%(key, tensor.shape))
    return tensors

ckpt_file_path = os.path.join(args.tf_checkpoint_dir, 'bert_model.ckpt')
tf_tensors = read_tf_checkpoint(ckpt_file_path)
tf_names = sorted(tf_tensors.keys())

###############################################################################
#                        Convert Tensorflow Checkpoint                        #
###############################################################################

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
  ('cls/predictions/transform/dense/','decoder.0.'),
  ('cls/predictions/transform/LayerNorm/','decoder.2.'),
  ('kernel', 'weight'),
  ('/intermediate/dense/', '.ffn.ffn_1.'),
  ('/output/dense/', '.ffn.ffn_2.'),
  ('/output/LayerNorm/', '.ffn.layer_norm.'),
  ('bert/embeddings/LayerNorm/', 'encoder.layer_norm.'),
  ('bert/embeddings/position_embeddings', 'encoder.position_weight'),
  ('bert/embeddings/token_type_embeddings', 'token_type_embed.0.weight'),
  ('bert/embeddings/word_embeddings', 'word_embed.0.weight'),
  ('bert/pooler/dense/', 'pooler.'),
  ('/','.'),
]

# convert to gluon parameters
gluon_tensors = {}
print('================= Converting to Gluon checkpoint ... ================= ')
for source_name in tf_names:
    # get the source tensor and its transpose
    source, source_t = tf_tensors[source_name], tf_tensors[source_name].T
    target, target_name = source, source_name
    for old, new in NAME_MAP:
        target_name = target_name.replace(old, new)
    # transpose kernel parameters
    if 'kernel' in source_name:
        target = source_t
    gluon_tensors[target_name] = target
    if source_t.shape == source.shape and len(source.shape) > 1 and target is not source_t:
        print('Warning: %s has shape %s'%(target_name, target.shape))
    print('%s: %s'%(target_name, target.shape))

# XXX manual tie weight
gluon_tensors['decoder.3.weight'] = gluon_tensors['word_embed.0.weight']
print('================= Total number of parameters = %d ================='%(len(tf_tensors)))
print('Total number of gluon_tensors parameters = %d (including one duplicated for decoder weight tying)'%len(gluon_tensors))

###############################################################################
#                     Convert Vocabulary and Embedding                        #
###############################################################################

vocab_path = os.path.join(args.tf_checkpoint_dir, 'vocab.txt')
vocab, swap_idx = convert_vocab(vocab_path)
embedding = gluon_tensors['word_embed.0.weight']
for pair in swap_idx:
    source_idx = pair[0]
    dst_idx = pair[1]
    source = embedding[source_idx].copy()
    dst = embedding[dst_idx].copy()
    embedding[source_idx][:] = dst
    embedding[dst_idx][:] = source


# XXX assume configs remain the same
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

tmp_file_path = os.path.join(args.out_dir, 'tmp')
with open(tmp_file_path, 'w') as f:
    f.write(vocab.to_json())
hash_long, hash_short = get_hash(tmp_file_path)
final_vocab_path = os.path.join(args.out_dir, hash_short + '.vocab')
with open(final_vocab_path, 'w') as f:
    f.write(vocab.to_json())
    print(hash_long, final_vocab_path)

bert.initialize(init=mx.init.Normal(0.02))
logging.info(bert)

ones = mx.nd.ones((2, 8))
out = bert(ones, ones, mx.nd.array([1,2]))
params = bert._collect_params_with_prefix()

loaded = {}
for name in params:
    try:
        arr = mx.nd.array(gluon_tensors[name])
        params[name].set_data(arr)
        loaded[name] = 0
    except:
        if name not in gluon_tensors:
            print('cannot initialize %s from bert checkpoint'%(name))
        else:
            print('cannot initialize ', name, 'param.shape = ', params[name].shape,
                  ' but found ', gluon_tensors[name].shape)

print('num_loaded = ', len(loaded), ' total = ', len(gluon_tensors))
for name in gluon_tensors:
    if name not in loaded:
        print('not loading', name)

bert.save_parameters(tmp_file_path)
hash_long, hash_short = get_hash(tmp_file_path)
final_param_path = os.path.join(args.out_dir, hash_short + '.params')
print(hash_long, final_param_path)
bert.save_parameters(final_param_path)
mx.nd.waitall()

'''
 /home/ubuntu/bert/cased_L-12_H-768_A-12/bert_model.ckpt  --
 /home/ubuntu/bert/uncased_L-12_H-768_A-12/bert_model.ckpt  --
 /home/ubuntu/bert/multilingual_L-12_H-768_A-12/bert_model.ckpt  --
 /home/ubuntu/bert/uncased_L-24_H-1024_A-16/bert_model.ckpt  --
'''

#bert, vocab = get_model(model, dataset_name=dataset, vocab=None,
#                        pretrained=False,
#                        root='/home/ubuntu/gluon-nlp/tests/data/model/')

#print(sorted(bert._collect_params_with_prefix().keys()))

