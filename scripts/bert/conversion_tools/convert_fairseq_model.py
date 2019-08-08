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
""" Script for converting Fairseq Roberta Model to Gluon. """

import torch
from fairseq.models.roberta import RobertaModel

ckpt = torch.load('/home/ubuntu/roberta/roberta.base/model.pt')
print(ckpt['args'])
pytorch_params = ckpt['model']


# Load the model in fairseq
roberta = RobertaModel.from_pretrained('/home/ubuntu/roberta/roberta.base')
roberta.eval()
print(roberta)
#exit()

#for k, v in pytorch_params.items():
#    print(k, v.shape)

import mxnet as mx
import gluonnlp as nlp
from gluonnlp.model import BERTEncoder, BERTModel
from gluonnlp.model.bert import bert_hparams

#mx_model = mx.nd.load('/home/ubuntu/.mxnet/models/bert_12_768_12_book_corpus_wiki_en_uncased-75cc780f.params')


predefined_args = bert_hparams['roberta_12_768_12']#args.model]

# BERT encoder
encoder = BERTEncoder(attention_cell='multi_head_self', #predefined_args['attention_cell'],
                      num_layers=predefined_args['num_layers'], units=predefined_args['units'],
                      hidden_size=predefined_args['hidden_size'],
                      max_length=predefined_args['max_length'],
                      num_heads=predefined_args['num_heads'], scaled=predefined_args['scaled'],
                      dropout=predefined_args['dropout'],
                      use_residual=predefined_args['use_residual'],
                      layer_norm_eps=predefined_args['layer_norm_eps'])

# BERT model
bert = BERTModel(encoder, 50265, #len(vocab),
                 #token_type_vocab_size=predefined_args['token_type_vocab_size'],
                 units=predefined_args['units'], embed_size=predefined_args['embed_size'],
                 embed_dropout=predefined_args['embed_dropout'],
                 word_embed=predefined_args['word_embed'], use_pooler=False,
                 use_token_type_embed=False, use_classifier=False)

bert.initialize(init=mx.init.Normal(0.02))

ones = mx.nd.ones((2, 8))
out = bert(ones, None, mx.nd.array([5, 6]), mx.nd.array([[1], [2]]))
params = bert._collect_params_with_prefix()
 
bert.save_parameters('test.params')
mx_model = mx.nd.load('test.params')
#for k, v in mx_model.items():
#    print(k, v.shape)

mapping = {
    'decoder.2' : 'decoder.lm_head.layer_norm',
    'decoder.0' : 'decoder.lm_head.dense',
    'decoder.3' : 'decoder.lm_head',
    'encoder.layer_norm' : 'decoder.sentence_encoder.emb_layer_norm',
    'encoder.position_weight' : 'decoder.sentence_encoder.embed_positions.weight',
    'encoder.transformer_cells': 'decoder.sentence_encoder.layers',
    'attention_cell.proj.' : 'self_attn.in_proj_',
    'ffn.ffn_1' : 'fc1',
    'ffn.ffn_2' : 'fc2',
    'layer_norm.gamma' : 'layer_norm.weight',
    'layer_norm.beta' : 'layer_norm.bias',
    'ffn.layer_norm' : 'final_layer_norm',
    'word_embed.0.weight' : 'decoder.sentence_encoder.embed_tokens.weight',
}

for i in range(24):
    mapping['{}.layer_norm'.format(i)] = '{}.self_attn_layer_norm'.format(i)
    mapping['{}.proj'.format(i)] = '{}.self_attn.out_proj'.format(i)

# set parameter data
loaded_params = {}
for name in params:
    pytorch_name = name
    for source, dest in mapping.items():
        pytorch_name = pytorch_name.replace(source, dest)

    assert pytorch_name in pytorch_params.keys(), 'Key ' + pytorch_name + ' for ' + name + ' not found.'
    torch_arr = pytorch_params[pytorch_name].cpu()
    # fairseq positional embedding starts with index 2
    if pytorch_name == 'decoder.sentence_encoder.embed_positions.weight':
       torch_arr = torch_arr[2:]
    arr = mx.nd.array(torch_arr)
    assert arr.shape == params[name].shape, (arr.shape, params[name].shape, name, pytorch_name)
    params[name].set_data(arr)
    loaded_params[name] = True

assert len(params) == len(loaded_params)
#print(sorted(list(mx_model.keys())))

assert len(params) == len(pytorch_params), "Gluon model does not match PyTorch model. " \
    "Please fix the BERTModel hyperparameters\n" + str(len(params)) + ' v.s. ' + str(len(pytorch_params))


tokens = roberta.encode('Hello world abc 中文!')
last_layer_features = roberta.extract_features(tokens)
pytorch_out = last_layer_features.detach().numpy()

mx_out = bert(mx.nd.array([tokens.tolist()]))
import numpy as np

print('stdev = ', mx_out.asnumpy().mean(), pytorch_out.mean())
print('stdev = ', np.std(mx_out.asnumpy() - pytorch_out))
mx.test_utils.assert_almost_equal(mx_out.asnumpy(), pytorch_out, atol=1e-3, rtol=1e-3)
mx.test_utils.assert_almost_equal(mx_out.asnumpy(), pytorch_out, atol=5e-6, rtol=5e-6)
