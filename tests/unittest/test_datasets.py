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
import sys

import mxnet as mx
import numpy as np

import gluonnlp as nlp

if sys.version_info[0] == 3:
    _str_types = (str, )
else:
    _str_types = (str, unicode)


def test_wikitext2():
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

    train_data = train.bptt_batchify(vocab, 35, 80, last_batch='discard')
    assert len(train_data) == 741, len(train_data)

    for i, (data, target) in enumerate(train_data):
        mx.test_utils.assert_almost_equal(data[1:].asnumpy(), target[:-1].asnumpy())
        assert data.shape == target.shape == (35, 80)

    train_data = train.bptt_batchify(vocab, 35, 80, last_batch='keep')
    assert len(train_data) == 742, len(train_data)
    assert train_data[-1][0].shape[0] < 35
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
    batched_data = train.batchify(vocab, 80)
    assert batched_data.shape == (26107, 80)


def test_imdb():
    train = nlp.data.IMDB(
        root=os.path.join('tests', 'data', 'imdb'), segment='train')
    test = nlp.data.IMDB(
        root=os.path.join('tests', 'data', 'imdb'), segment='test')
    unsup = nlp.data.IMDB(
        root=os.path.join('tests', 'data', 'imdb'), segment='unsup')
    assert len(train) == 25000, len(train)
    assert len(test) == 25000, len(test)
    assert len(unsup) == 50000, len(unsup)

    for i, (data, score) in enumerate(train):
        assert isinstance(data, _str_types)
        assert score <= 4 or score >= 7

    for i, (data, score) in enumerate(test):
        assert isinstance(data, _str_types)
        assert score <= 4 or score >= 7

    for i, (data, score) in enumerate(unsup):
        assert isinstance(data, _str_types)
        assert score == 0


###############################################################################
# Word similarity and relatedness datasets
###############################################################################
def _assert_similarity_dataset(data):
    # Check datatypes
    assert isinstance(data[0][0], _str_types)
    assert isinstance(data[0][1], _str_types)
    assert np.isfinite(data[0][2])

    # Check score magnitude
    assert all(data.min <= row[2] <= data.max for row in data)


def test_wordsim353():
    for segment, length in (("all", 252 + 203), ("relatedness", 252),
                            ("similarity", 203)):
        data = nlp.data.WordSim353(
            segment=segment, root=os.path.join('tests', 'data', 'wordsim353'))
        assert len(data) == length, len(data)
        _assert_similarity_dataset(data)


def test_men():
    for segment, length in [("full", 3000), ("dev", 2000), ("test", 1000)]:
        data = nlp.data.MEN(
            root=os.path.join('tests', 'data', 'men'), segment=segment)
        assert len(data) == length, len(data)
        _assert_similarity_dataset(data)


def test_radinsky_mturk():
    data = nlp.data.RadinskyMTurk(
        root=os.path.join('tests', 'data', 'radinsky'))
    assert len(data) == 287
    _assert_similarity_dataset(data)


def test_verb143():
    data = nlp.data.BakerVerb143(root=os.path.join('tests', 'data', 'verb143'))
    assert len(data) == 144
    _assert_similarity_dataset(data)


def test_verb130():
    data = nlp.data.YangPowersVerb130(
        root=os.path.join('tests', 'data', 'verb130'))
    assert len(data) == 130
    _assert_similarity_dataset(data)


def test_rare_words():
    data = nlp.data.RareWords(root=os.path.join('tests', 'data', 'rarewords'))
    assert len(data) == 2034
    _assert_similarity_dataset(data)


def test_simlex999():
    data = nlp.data.SimLex999(root=os.path.join('tests', 'data', 'simlex999'))
    assert len(data) == 999
    _assert_similarity_dataset(data)


def test_simverb3500():
    data = nlp.data.SimVerb3500(
        root=os.path.join('tests', 'data', 'simverb3500'))
    assert len(data) == 3500
    _assert_similarity_dataset(data)


def test_semeval17task2():
    for segment, length in [("trial", 18), ("test", 500)]:
        data = nlp.data.SemEval17Task2(
            root=os.path.join('tests', 'data', 'semeval17task2'),
            segment=segment)
        assert len(data) == length
        _assert_similarity_dataset(data)


###############################################################################
# Word analogy datasets
###############################################################################
def test_googleanalogy():
    data = nlp.data.GoogleAnalogyTestSet(
        root=os.path.join('tests', 'data', 'google_analogy'))
    assert len(data[0]) == 4
    assert len(data) == 10675 + 8869


def test_bigger_analogy():
    data = nlp.data.BiggerAnalogyTestSet(
        root=os.path.join('tests', 'data', 'bigger_analogy'))
    assert len(data[0]) == 4
    assert len(data) == 98000


if __name__ == '__main__':
    import nose
    nose.runmodule()
