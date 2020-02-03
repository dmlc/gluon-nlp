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
import argparse
import logging
import os
import sys
import io
import numpy as np

import torch
from fairseq.models.roberta import RobertaModel

import mxnet as mx
import gluonnlp as nlp
from gluonnlp.model import BERTEncoder, BERTModel
from gluonnlp.model.bert import bert_hparams
from gluonnlp.data.utils import _load_pretrained_vocab

from utils import get_hash, load_text_vocab, tf_vocab_to_gluon_vocab

parser = argparse.ArgumentParser(description='Conversion script for Fairseq RoBERTa model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--ckpt_dir', type=str, help='Full path to the roberta folder',
                    default='/home/ubuntu/roberta/roberta.base')
parser.add_argument('--model', type=str, help='Model type. ',
                    choices=['roberta_12_768_12', 'roberta_24_1024_16'],
                    default='roberta_12_768_12')
parser.add_argument('--verbose', action='store_true', help='Verbose logging')

args = parser.parse_args()

ckpt_dir = os.path.expanduser(args.ckpt_dir)

ckpt = torch.load(os.path.join(ckpt_dir, 'model.pt'))
pytorch_params = ckpt['model']

if args.verbose:
    print(ckpt['args'])
    for k, v in pytorch_params.items():
        print(k, v.shape)

# Load the model in fairseq
roberta = RobertaModel.from_pretrained(ckpt_dir)
roberta.eval()

def fairseq_vocab_to_gluon_vocab(torch_vocab):
    index_to_words = [None] * len(torch_vocab)

    bos_idx = torch_vocab.bos()
    pad_idx = torch_vocab.pad()
    eos_idx = torch_vocab.eos()
    unk_idx = torch_vocab.unk()

    index_to_words[bos_idx] = torch_vocab.symbols[bos_idx]
    index_to_words[pad_idx] = torch_vocab.symbols[pad_idx]
    index_to_words[eos_idx] = torch_vocab.symbols[eos_idx]
    index_to_words[unk_idx] = torch_vocab.symbols[unk_idx]

    specials = [bos_idx, pad_idx, eos_idx, unk_idx]

    openai_to_roberta = {}
    openai_vocab = _load_pretrained_vocab('openai_webtext', '.')

    with io.open(os.path.join(ckpt_dir, 'dict.txt'), encoding='utf-8') as f:
        for i, line in enumerate(f):
            token, count = line.split(' ')
            try:
                fake_token = int(token)
                openai_to_roberta[token] = i + len(specials)
            except ValueError:
                index_to_words[i + len(specials)] = token

    for idx, token in enumerate(openai_vocab.idx_to_token):
        if str(idx) in openai_to_roberta:
            index_to_words[openai_to_roberta[str(idx)]] = token
        else:
            assert token == u'<mask>', token

    mask_idx = torch_vocab.index(u'<mask>')
    index_to_words[mask_idx] = torch_vocab.string([mask_idx])
    assert None not in index_to_words
    word2idx = {}
    for idx, token in enumerate(index_to_words):
        word2idx[token] = idx

    vocab = nlp.vocab.Vocab(word2idx, token_to_idx=word2idx,
                            unknown_token=index_to_words[unk_idx],
                            padding_token=index_to_words[pad_idx],
                            bos_token=index_to_words[bos_idx],
                            eos_token=index_to_words[eos_idx],
                            mask_token=u'<mask>')
    return vocab

vocab = fairseq_vocab_to_gluon_vocab(roberta.task.dictionary)

predefined_args = bert_hparams[args.model]

# BERT encoder
encoder = BERTEncoder(attention_cell=predefined_args['attention_cell'],
                      num_layers=predefined_args['num_layers'], units=predefined_args['units'],
                      hidden_size=predefined_args['hidden_size'],
                      max_length=predefined_args['max_length'],
                      num_heads=predefined_args['num_heads'], scaled=predefined_args['scaled'],
                      dropout=predefined_args['dropout'],
                      use_residual=predefined_args['use_residual'],
                      layer_norm_eps=predefined_args['layer_norm_eps'])

# BERT model
bert = BERTModel(encoder, len(vocab),
                 units=predefined_args['units'], embed_size=predefined_args['embed_size'],
                 word_embed=predefined_args['word_embed'], use_pooler=False,
                 use_token_type_embed=False, use_classifier=False)

bert.initialize(init=mx.init.Normal(0.02))

ones = mx.nd.ones((2, 8))
out = bert(ones, None, mx.nd.array([5, 6]), mx.nd.array([[1], [2]]))
params = bert._collect_params_with_prefix()



mapping = {
    'decoder.2' : 'decoder.lm_head.layer_norm',
    'decoder.0' : 'decoder.lm_head.dense',
    'decoder.3' : 'decoder.lm_head',
    'encoder.layer_norm' : 'decoder.sentence_encoder.emb_layer_norm',
    'encoder.position_weight' : 'decoder.sentence_encoder.embed_positions.weight',
    'encoder.transformer_cells': 'decoder.sentence_encoder.layers',
    'attention_cell.proj_key.' : 'self_attn.in_proj_',
    'attention_cell.proj_value.' : 'self_attn.in_proj_',
    'attention_cell.proj_query.' : 'self_attn.in_proj_',
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
visited_pytorch_params = {}
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
    if 'attention_cell.proj' in name:
        unfused = ['query', 'key', 'value']
        arrs = arr.split(num_outputs=3, axis=0)
        for i, p in enumerate(unfused):
            if p in name:
                arr = arrs[i]
    else:
        assert arr.shape == params[name].shape, (arr.shape, params[name].shape, name, pytorch_name)
    params[name].set_data(arr)
    loaded_params[name] = True
    visited_pytorch_params[pytorch_name] = True

assert len(params) == len(loaded_params)
assert len(visited_pytorch_params) == len(pytorch_params), "Gluon model does not match PyTorch model. " \
    "Please fix the BERTModel hyperparameters\n" + str(len(visited_pytorch_params)) + ' v.s. ' + str(len(pytorch_params))


texts = 'Hello world. abc, def and 中文!'
torch_tokens = roberta.encode(texts)

torch_features = roberta.extract_features(torch_tokens)
pytorch_out = torch_features.detach().numpy()

mx_tokenizer = nlp.data.GPT2BPETokenizer()
mx_tokens = [vocab.bos_token] + mx_tokenizer(texts) + [vocab.eos_token]
mx_data = vocab[mx_tokens]
print(mx_tokens)
print(vocab[mx_tokens])
print(torch_tokens)
assert mx_data == torch_tokens.tolist()

mx_out = bert(mx.nd.array([mx_data]))
print('stdev = ', np.std(mx_out.asnumpy() - pytorch_out))
mx.test_utils.assert_almost_equal(mx_out.asnumpy(), pytorch_out, atol=1e-3, rtol=1e-3)
mx.test_utils.assert_almost_equal(mx_out.asnumpy(), pytorch_out, atol=5e-6, rtol=5e-6)

bert.save_parameters(os.path.join(ckpt_dir, args.model + '.params'))
with io.open(os.path.join(ckpt_dir, args.model + '.vocab'), 'w', encoding='utf-8') as f:
    f.write(vocab.to_json())
