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
"""Utility functions for BERT."""
import json
import hashlib
import gluonnlp
from tensorflow.python import pywrap_tensorflow
try:
    from tokenizer import load_vocab
except ImportError:
    from .tokenizer import load_vocab

__all__ = ['convert_vocab']

def convert_vocab(vocab_file):
    """GluonNLP specific code to convert the original vocabulary to nlp.vocab.Vocab."""
    original_vocab = load_vocab(vocab_file)
    token_to_idx = dict(original_vocab)
    num_tokens = len(token_to_idx)
    idx_to_token = [None] * len(original_vocab)
    for word in original_vocab:
        idx = int(original_vocab[word])
        idx_to_token[idx] = word

    def swap(token, target_idx, token_to_idx, idx_to_token, swap_idx):
        original_idx = token_to_idx[token]
        original_token = idx_to_token[target_idx]
        token_to_idx[token] = target_idx
        token_to_idx[original_token] = original_idx
        idx_to_token[target_idx] = token
        idx_to_token[original_idx] = original_token
        swap_idx.append((original_idx, target_idx))

    reserved_tokens = ['[PAD]', '[CLS]', '[SEP]', '[MASK]']
    unknown_token = '[UNK]'
    padding_token = '[PAD]'
    swap_idx = []
    assert unknown_token in token_to_idx
    assert padding_token in token_to_idx
    swap(unknown_token, 0, token_to_idx, idx_to_token, swap_idx)
    for i, token in enumerate(reserved_tokens):
        swap(token, i + 1, token_to_idx, idx_to_token, swap_idx)

    # sanity checks
    assert len(token_to_idx) == num_tokens
    assert len(idx_to_token) == num_tokens
    assert None not in idx_to_token
    assert len(set(idx_to_token)) == num_tokens

    vocab_dict = {}
    vocab_dict['idx_to_token'] = idx_to_token
    vocab_dict['token_to_idx'] = token_to_idx
    vocab_dict['reserved_tokens'] = reserved_tokens
    vocab_dict['unknown_token'] = unknown_token
    vocab_dict['padding_token'] = padding_token
    vocab_dict['bos_token'] = None
    vocab_dict['eos_token'] = None
    json_str = json.dumps(vocab_dict)
    converted_vocab = gluonnlp.Vocab.from_json(json_str)
    return converted_vocab, swap_idx

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
    tensors = {}
    reader = pywrap_tensorflow.NewCheckpointReader(path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        tensor = reader.get_tensor(key)
        tensors[key] = tensor
    return tensors
