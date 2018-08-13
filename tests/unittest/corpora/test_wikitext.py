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

import mxnet as mx
import pytest

import gluonnlp as nlp



@pytest.mark.parametrize('batch_size', [7, 80])
@pytest.mark.parametrize('seq_len', [7, 35])
def test_bptt_batchify(batch_size, seq_len):
    data = nlp.data.WikiText2(
        segment='test', root=os.path.join('tests', 'data', 'wikitext-2'))
    vocab = nlp.Vocab(nlp.data.utils.Counter(data[0]))

    # unsupported last_batch
    with pytest.raises(ValueError):
        data.bptt_batchify(vocab, seq_len, batch_size, last_batch='unsupported')

    # last_batch='keep'
    X, Y = zip(*(data.bptt_batchify(vocab, seq_len, batch_size, last_batch='keep')))
    X, Y = mx.nd.concat(*X, dim=0), mx.nd.concat(*Y, dim=0)
    coded = mx.nd.concat(X, Y[-1].expand_dims(0), dim=0).T.reshape(-1).asnumpy().tolist()
    assert vocab[data[0]] == coded[:len(data[0])]
    assert all(pad == vocab[vocab.padding_token] for pad in coded[len(data[0]):])

    # last_batch='discard'
    X, Y = zip(*(data.bptt_batchify(vocab, seq_len, batch_size, last_batch='discard')))
    X, Y = mx.nd.concat(*X, dim=0), mx.nd.concat(*Y, dim=0)
    coded = mx.nd.concat(X, Y[-1].expand_dims(0), dim=0).T.reshape(-1).asnumpy().tolist()
    assert len(data[0]) - len(coded) < batch_size * seq_len


def test_wikitext2():
    batch_size = 80
    seq_len = 35

    train = nlp.data.WikiText2(
        segment='train', root=os.path.join('tests', 'data', 'wikitext-2'))
    val = nlp.data.WikiText2(
        segment='val', root=os.path.join('tests', 'data', 'wikitext-2'))
    test = nlp.data.WikiText2(
        segment='test', root=os.path.join('tests', 'data', 'wikitext-2'))
    train_freq, val_freq, test_freq = [nlp.data.utils.Counter(x) for x in [train[0], val[0], test[0]]]
    assert len(train[0]) == 2075677, len(train[0])
    assert len(train_freq) == 33278, len(train_freq)
    assert len(val[0]) == 216347, len(val[0])
    assert len(val_freq) == 13777, len(val_freq)
    assert len(test[0]) == 244102, len(test[0])
    assert len(test_freq) == 14143, len(test_freq)
    assert test_freq['English'] == 32, test_freq['English']

    vocab = nlp.Vocab(train_freq)
    serialized_vocab = vocab.to_json()
    assert len(serialized_vocab) == 962190, len(serialized_vocab)
    assert json.loads(serialized_vocab)['idx_to_token'] == vocab._idx_to_token

    train_data = train.bptt_batchify(vocab, seq_len, batch_size, last_batch='discard')
    assert len(train_data) == 741, len(train_data)

    for i, (data, target) in enumerate(train_data):
        mx.test_utils.assert_almost_equal(data[1:].asnumpy(), target[:-1].asnumpy())
        assert data.shape == target.shape == (seq_len, batch_size)

    train_data = train.bptt_batchify(vocab, seq_len, batch_size, last_batch='keep')
    assert len(train_data) == 742, len(train_data)
    assert train_data[-1][0].shape[0] <= seq_len
    for i, (data, target) in enumerate(train_data):
        mx.test_utils.assert_almost_equal(data[1:].asnumpy(), target[:-1].asnumpy())
        assert data.shape == target.shape

    train_freq, val_freq, test_freq = [nlp.data.utils.Counter(x) for x in [train[0], val[0], test[0]]]
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
    assert len(train[0]) == 2088628, len(train[0])
    assert len(train_freq) == 33278, len(train_freq)
    assert len(val[0]) == 217646, len(val[0])
    assert len(val_freq) == 13777, len(val_freq)
    assert len(test[0]) == 245569, len(test[0])
    assert len(test_freq) == 14143, len(test_freq)
    assert test_freq['English'] == 32, test_freq['English']
    batched_data = train.batchify(vocab, batch_size)
    assert batched_data.shape == (26107, batch_size)


def test_wikitext2_raw():
    train = nlp.data.WikiText2Raw(segment='train', root=os.path.join(
        'tests', 'data', 'wikitext-2'))
    val = nlp.data.WikiText2Raw(segment='val', root=os.path.join(
        'tests', 'data', 'wikitext-2'))
    test = nlp.data.WikiText2Raw(segment='test', root=os.path.join(
        'tests', 'data', 'wikitext-2'))
    train_freq, val_freq, test_freq = [
        nlp.data.utils.Counter(x) for x in [train[0], val[0], test[0]]
    ]
    assert len(train[0]) == 10843541, len(train[0])
    assert len(train_freq) == 192, len(train_freq)
    assert len(val[0]) == 1136862, len(val[0])
    assert len(val_freq) == 168, len(val_freq)
    assert len(test[0]) == 1278983, len(test[0])
    assert len(test_freq) == 177, len(test_freq)
    assert test_freq['a'.encode('utf-8')[0]] == 81512, \
        test_freq['a'.encode('utf-8')[0]]


def test_wikitext103_raw():
    train = nlp.data.WikiText103Raw(segment='train', root=os.path.join(
        'tests', 'data', 'wikitext-103'))
    val = nlp.data.WikiText103Raw(segment='val', root=os.path.join(
        'tests', 'data', 'wikitext-103'))
    test = nlp.data.WikiText103Raw(segment='test', root=os.path.join(
        'tests', 'data', 'wikitext-103'))
    train_freq, val_freq, test_freq = [
        nlp.data.utils.Counter(x) for x in [train[0], val[0], test[0]]
    ]
    assert len(train[0]) == 535800393, len(train[0])
    assert len(train_freq) == 203, len(train_freq)
    assert len(val[0]) == 1136862, len(val[0])
    assert len(val_freq) == 168, len(val_freq)
    assert len(test[0]) == 1278983, len(test[0])
    assert len(test_freq) == 177, len(test_freq)
    assert test_freq['a'.encode('utf-8')[0]] == 81512, \
        test_freq['a'.encode('utf-8')[0]]

