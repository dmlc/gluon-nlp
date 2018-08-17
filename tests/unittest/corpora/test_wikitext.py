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

from __future__ import print_function

import json
import os

import gluonnlp as nlp
import mxnet as mx


def test_wikitext2():
    batch_size = 80
    seq_len = 35

    train = nlp.data.WikiText2(
        segment='train', root=os.path.join('tests', 'data', 'wikitext-2'))
    val = nlp.data.WikiText2(
        segment='val', root=os.path.join('tests', 'data', 'wikitext-2'))
    test = nlp.data.WikiText2(
        segment='test', root=os.path.join('tests', 'data', 'wikitext-2'))
    train_freq, val_freq, test_freq = [
        nlp.data.utils.Counter(x) for x in [train, val, test]
    ]
    assert len(train) == 2075677
    assert len(train_freq) == 33278
    assert len(val) == 216347
    assert len(val_freq) == 13777
    assert len(test) == 244102
    assert len(test_freq) == 14143
    assert test_freq['English'] == 32

    vocab = nlp.Vocab(train_freq)
    serialized_vocab = vocab.to_json()
    assert len(serialized_vocab) == 962190, len(serialized_vocab)
    assert json.loads(serialized_vocab)['idx_to_token'] == vocab._idx_to_token

    bptt_discard = nlp.data.batchify.CorpusBPTTBatchify(
        vocab, seq_len, batch_size, last_batch='discard')
    bptt_keep = nlp.data.batchify.CorpusBPTTBatchify(
        vocab, seq_len, batch_size, last_batch='keep')

    train_data = bptt_discard(train)
    assert len(train_data) == 741, len(train_data)
    for i, (data, target) in enumerate(train_data):
        mx.test_utils.assert_almost_equal(data[1:].asnumpy(), target[:-1].asnumpy())
        assert data.shape == target.shape == (seq_len, batch_size)

    train_data = bptt_keep(train)
    assert len(train_data) == 742, len(train_data)
    assert train_data[-1][0].shape[0] <= seq_len
    for i, (data, target) in enumerate(train_data):
        mx.test_utils.assert_almost_equal(data[1:].asnumpy(), target[:-1].asnumpy())
        assert data.shape == target.shape

    # skip_empty=False
    train = nlp.data.WikiText2(
        segment='train',
        skip_empty=False,
        root=os.path.join('tests', 'data', 'wikitext-2'))
    val = nlp.data.WikiText2(
        segment='val',
        skip_empty=False,
        root=os.path.join('tests', 'data', 'wikitext-2'))
    test = nlp.data.WikiText2(
        segment='test',
        skip_empty=False,
        root=os.path.join('tests', 'data', 'wikitext-2'))
    train_freq, val_freq, test_freq = [
        nlp.data.utils.Counter(x) for x in [train, val, test]
    ]
    assert len(train) == 2088628
    assert len(train_freq) == 33278
    assert len(val) == 217646
    assert len(val_freq) == 13777
    assert len(test) == 245569
    assert len(test_freq) == 14143
    assert test_freq['English'] == 32
    batched_data = nlp.data.batchify.CorpusBatchify(vocab, batch_size)(train)
    assert batched_data[:].shape == (26107, batch_size)


def test_wikitext2_raw():
    train = nlp.data.WikiText2Raw(
        segment='train', root=os.path.join('tests', 'data', 'wikitext-2'))
    val = nlp.data.WikiText2Raw(
        segment='val', root=os.path.join('tests', 'data', 'wikitext-2'))
    test = nlp.data.WikiText2Raw(
        segment='test', root=os.path.join('tests', 'data', 'wikitext-2'))
    train_freq, val_freq, test_freq = [
        nlp.data.utils.Counter(x) for x in [train, val, test]
    ]
    assert len(train) == 10843541
    assert len(train_freq) == 192
    assert len(val) == 1136862
    assert len(val_freq) == 168
    assert len(test) == 1278983
    assert len(test_freq) == 177
    assert test_freq['a'.encode('utf-8')[0]] == 81512


def test_wikitext103_raw():
    train = nlp.data.WikiText103Raw(
        segment='train', root=os.path.join('tests', 'data', 'wikitext-103'))
    val = nlp.data.WikiText103Raw(
        segment='val', root=os.path.join('tests', 'data', 'wikitext-103'))
    test = nlp.data.WikiText103Raw(
        segment='test', root=os.path.join('tests', 'data', 'wikitext-103'))
    train_freq, val_freq, test_freq = [
        nlp.data.utils.Counter(x) for x in [train, val, test]
    ]
    assert len(train) == 535800393
    assert len(train_freq) == 203
    assert len(val) == 1136862
    assert len(val_freq) == 168
    assert len(test) == 1278983
    assert len(test_freq) == 177
    assert test_freq['a'.encode('utf-8')[0]] == 81512
