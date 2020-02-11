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
""" Script for converting the distilbert model from pytorch-transformer to Gluon.

Usage: 

pip3 install pytorch-transformers

python3 convert_pytorch_transformers.py

If you are not converting the distilbert model, please change the code section noted
by "TODO".

 """

import argparse
import pytorch_transformers
import torch
import mxnet as mx
import gluonnlp as nlp
import os, logging, json
from utils import get_hash, load_text_vocab, tf_vocab_to_gluon_vocab

parser = argparse.ArgumentParser(description='Conversion script for pytorch-transformer '
                                             'distilbert model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--out_dir', type=str, help='Full path to the output folder',
                    default='./converted-model')

args = parser.parse_args()


####################################################################
#                  LOAD A BERT MODEL FROM PYTORCH                  #
####################################################################
# TODO: change this to your bert model and tokenizer used in pytorch-transformer
tokenizer = pytorch_transformers.tokenization_distilbert.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = pytorch_transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')

dir_name = './temp'
gluon_dir_name = args.out_dir
nlp.utils.mkdir(dir_name)
nlp.utils.mkdir(gluon_dir_name)
model_name = 'bert_12_768_12'
model.save_pretrained(dir_name)
tokenizer.save_pretrained(dir_name)

####################################################################
#                  SHOW PYTORCH PARAMETER LIST                     #
####################################################################
pytorch_parameters = torch.load(os.path.join(dir_name, 'pytorch_model.bin'))
print('parameters in pytorch')
print(sorted(list(pytorch_parameters)))

####################################################################
#                        CONVERT VOCAB                             #
####################################################################
# convert vocabulary
vocab = tf_vocab_to_gluon_vocab(load_text_vocab(os.path.join(dir_name, 'vocab.txt')))
# vocab serialization
tmp_file_path = os.path.expanduser(os.path.join(gluon_dir_name, 'temp'))
with open(tmp_file_path, 'w') as f:
    f.write(vocab.to_json())

hash_full, hash_short = get_hash(tmp_file_path)
gluon_vocab_path = os.path.expanduser(os.path.join(gluon_dir_name, hash_short + '.vocab'))
with open(gluon_vocab_path, 'w') as f:
    f.write(vocab.to_json())
    print('vocab file saved to {}. hash = {}'.format(gluon_vocab_path, hash_full))

####################################################################
#                       CONVERT PARAMS OPTIONS                     #
####################################################################
torch_to_gluon_config_names = {
  "attention_dropout": 'dropout',
  "dim": 'embed_size',
  "dropout": 'dropout',
  "hidden_dim": 'hidden_size',
  "max_position_embeddings": 'max_length',
  "n_heads": 'num_heads',
  "n_layers": 'num_layers',
  "output_attentions": 'output_attention',
  "output_hidden_states": 'output_all_encodings',
  "vocab_size": 'vocab_size',
}

predefined_args = nlp.model.bert.bert_hparams[model_name]

with open(os.path.join(dir_name, 'config.json'), 'r') as f:
    torch_config = json.load(f)
    for name, value in torch_config.items():
        if name in torch_to_gluon_config_names:
            predefined_args[torch_to_gluon_config_names[name]] = value

# BERT encoder
encoder = nlp.model.BERTEncoder(attention_cell=predefined_args['attention_cell'],
                      num_layers=predefined_args['num_layers'], units=predefined_args['units'],
                      hidden_size=predefined_args['hidden_size'],
                      max_length=predefined_args['max_length'],
                      num_heads=predefined_args['num_heads'], scaled=predefined_args['scaled'],
                      dropout=predefined_args['dropout'],
                      use_residual=predefined_args['use_residual'])

# BERT model
bert = nlp.model.BERTModel(encoder, len(vocab),
                 units=predefined_args['units'], embed_size=predefined_args['embed_size'],
                 embed_dropout=predefined_args['embed_dropout'],
                 word_embed=predefined_args['word_embed'], use_pooler=False,
                 # TODO: for some models, we may need to change the value for use_token_type_embed,
                 # use_classifier, and use_decoder
                 use_token_type_embed=False,
                 token_type_vocab_size=predefined_args['token_type_vocab_size'],
                 use_classifier=False, use_decoder=False)

bert.initialize(init=mx.init.Normal(0.02))

ones = mx.nd.ones((2, 8))
out = bert(ones, ones, mx.nd.array([5, 6]), mx.nd.array([[1], [2]]))
params = bert._collect_params_with_prefix()
print('parameters in gluon')
print(sorted(list(params.keys())))
assert len(params) == len(pytorch_parameters), ("Gluon model does not match PyTorch model. " \
    "Please fix the BERTModel hyperparameters", len(params), len(pytorch_parameters))

####################################################################
#                       CONVERT PARAMS VALUES                      #
####################################################################
mapping = {
'encoder.layer_norm.beta': 'embeddings.LayerNorm.bias',
'encoder.layer_norm.gamma': 'embeddings.LayerNorm.weight',
'encoder.position_weight': 'embeddings.position_embeddings.weight',
'word_embed.0.weight': 'embeddings.word_embeddings.weight',
'encoder.transformer_cells': 'transformer.layer',
'attention_cell': 'attention',
'.proj.': '.attention.out_lin.',
'proj_key':'k_lin',
'proj_query':'q_lin',
'proj_value':'v_lin',
'ffn_1':'lin1',
'ffn_2':'lin2',
'ffn.layer_norm.beta':'output_layer_norm.bias',
'ffn.layer_norm.gamma':'output_layer_norm.weight',
}
secondary_map = {'layer_norm.beta':'sa_layer_norm.bias',
                 'layer_norm.gamma':'sa_layer_norm.weight'
}

# set parameter data
loaded_params = {}
for name in params:
    pytorch_name = name
    for k, v in mapping.items():
        pytorch_name = pytorch_name.replace(k, v)
    for k, v in secondary_map.items():
        pytorch_name = pytorch_name.replace(k, v)
    arr = mx.nd.array(pytorch_parameters[pytorch_name])
    assert arr.shape == params[name].shape
    params[name].set_data(arr)
    loaded_params[name] = True

if len(params) != len(loaded_params):
    raise RuntimeError('The Gluon BERTModel comprises {} parameter arrays, '
                       'but {} have been extracted from the pytorch model. '.format(
                           len(params), len(loaded_params)))

####################################################################
#                       SAVE CONVERTED PARAMS                      #
####################################################################
# param serialization
param_path = os.path.join(gluon_dir_name, 'net.params')
bert.save_parameters(param_path)
hash_full, hash_short = get_hash(param_path)
print('param saved to {}. hash = {}'.format(param_path, hash_full))


####################################################################
#                       COMPARE OUTPUTS                            #
####################################################################
text = 'Hello, my dog is cute'
# TODO: use nlp.data.GPT2Tokenizer if the GPT2 tokenizer in pytorch is used
gluon_tokenizer = nlp.data.BERTTokenizer(vocab, lower=True)
transform = nlp.data.BERTSentenceTransform(gluon_tokenizer, max_seq_length=512, pair=False, pad=False)
sample = transform([text])
words, valid_len, _ = mx.nd.array([sample[0]]), mx.nd.array([sample[1]]), mx.nd.array([sample[2]]);
# TODO: for some tokenizers, no need to truncate words
words = words[:,1:-1]
seq_encoding = bert(words, None)
print('\nconverted vocab:')
print(vocab)

print('\ntesting sample:')
print(sample)
print('\ngluon output: ', seq_encoding)

input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
outputs = model(input_ids)
last_hidden_states = outputs[0]
print('\npytorch output: ')
print(last_hidden_states)

mx.nd.waitall()
mx.test_utils.assert_almost_equal(seq_encoding.asnumpy(), last_hidden_states.detach().numpy(), atol=1e-3, rtol=1e-3)
mx.test_utils.assert_almost_equal(seq_encoding.asnumpy(), last_hidden_states.detach().numpy(), atol=1e-5, rtol=1e-5)
print('\nCongrats! The result is the same. Assertion passed.')
