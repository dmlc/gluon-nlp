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

import datetime
import json
import os
import random

from flaky import flaky
import mxnet as mx
import numpy as np
import pytest

import gluonnlp as nlp

try:
    _str_types = (str, unicode)
except NameError:  # Python 3
    _str_types = (str, )


###############################################################################
# Registry
###############################################################################
def test_dataset_registry():
    @nlp.data.register(segment=['train'])
    class MyDataset(mx.gluon.data.Dataset):
        def __init__(self, segment='train'):
            pass

    my_dataset = nlp.data.create('MyDataset')

    with pytest.raises(RuntimeError):

        @nlp.data.register(segment='thisshouldbealistofarguments')
        class MyDataset2(mx.gluon.data.Dataset):
            def __init__(self, segment='train'):
                pass

    with pytest.raises(RuntimeError):

        @nlp.data.register(invalidargument=['train'])
        class MyDataset3(mx.gluon.data.Dataset):
            def __init__(self, segment='train'):
                pass

    @nlp.data.register()
    class MyDataset4(mx.gluon.data.Dataset):
        def __init__(self, segment='train'):
            pass

    my_dataset = nlp.data.create('MyDataset4')


    @nlp.data.register
    class MyDataset5(mx.gluon.data.Dataset):
        def __init__(self, segment='train'):
            pass

    my_dataset = nlp.data.create('MyDataset5')


###############################################################################
# Language model
###############################################################################
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


@flaky(max_runs=2, min_passes=1)
@pytest.mark.parametrize('segment,length', [('all', 352), ('relatedness', 252),
                                            ('similarity', 203)])
def test_wordsim353(segment, length):
    # 'all' has length 352 as the original dataset contains the 'money/cash'
    # pair twice with different similarity ratings, which was fixed by the
    # http://alfonseca.org/eng/research/wordsim353.html version of the dataset
    # that we are using.
    data = nlp.data.WordSim353(segment=segment, root=os.path.join(
        'tests', 'externaldata', 'wordsim353'))
    assert len(data) == length, len(data)
    _assert_similarity_dataset(data)


def test_men():
    for segment, length in [("full", 3000), ("dev", 2000), ("test", 1000)]:
        data = nlp.data.MEN(
            root=os.path.join('tests', 'data', 'men'), segment=segment)
        assert len(data) == length, len(data)
        _assert_similarity_dataset(data)


@flaky(max_runs=2, min_passes=1)
def test_radinsky_mturk():
    data = nlp.data.RadinskyMTurk(
        root=os.path.join('tests', 'externaldata', 'radinsky'))
    assert len(data) == 287
    _assert_similarity_dataset(data)


@flaky(max_runs=2, min_passes=1)
def test_verb143():
    data = nlp.data.BakerVerb143(
        root=os.path.join('tests', 'externaldata', 'verb143'))
    assert len(data) == 144
    _assert_similarity_dataset(data)


@flaky(max_runs=2, min_passes=1)
def test_verb130():
    data = nlp.data.YangPowersVerb130(
        root=os.path.join('tests', 'externaldata', 'verb130'))
    assert len(data) == 130
    _assert_similarity_dataset(data)


@flaky(max_runs=2, min_passes=1)
def test_rare_words():
    data = nlp.data.RareWords(
        root=os.path.join('tests', 'externaldata', 'rarewords'))
    assert len(data) == 2034
    _assert_similarity_dataset(data)


@flaky(max_runs=2, min_passes=1)
def test_simlex999():
    data = nlp.data.SimLex999(
        root=os.path.join('tests', 'externaldata', 'simlex999'))
    assert len(data) == 999
    _assert_similarity_dataset(data)


@flaky(max_runs=2, min_passes=1)
def test_simverb3500():
    data = nlp.data.SimVerb3500(
        root=os.path.join('tests', 'externaldata', 'simverb3500'))
    assert len(data) == 3500
    _assert_similarity_dataset(data)


@flaky(max_runs=2, min_passes=1)
def test_semeval17task2():
    for segment, length in [("trial", 18), ("test", 500)]:
        data = nlp.data.SemEval17Task2(
            root=os.path.join('tests', 'externaldata', 'semeval17task2'),
            segment=segment)
        assert len(data) == length
        _assert_similarity_dataset(data)


###############################################################################
# Word analogy datasets
###############################################################################
@flaky(max_runs=2, min_passes=1)
def test_googleanalogy():
    data = nlp.data.GoogleAnalogyTestSet(
        root=os.path.join('tests', 'externaldata', 'google_analogy'))
    assert len(data[0]) == 4
    assert len(data) == 10675 + 8869


@flaky(max_runs=2, min_passes=1)
def test_bigger_analogy():
    data = nlp.data.BiggerAnalogyTestSet(
        root=os.path.join('tests', 'externaldata', 'bigger_analogy'))
    assert len(data[0]) == 4
    assert len(data) == 98000


###############################################################################
# CONLL
###############################################################################
@flaky(max_runs=2, min_passes=1)
def test_conll2000():
    train = nlp.data.CoNLL2000(segment='train', root=os.path.join(
        'tests', 'externaldata', 'conll2000'))
    test = nlp.data.CoNLL2000(segment='test', root=os.path.join(
        'tests', 'externaldata', 'conll2000'))
    assert len(train) == 8936, len(train)
    assert len(test) == 2012, len(test)

    for i, (data, pos, chk) in enumerate(train):
        assert all(isinstance(d, _str_types) for d in data), data
        assert all(isinstance(p, _str_types) for p in pos), pos
        assert all(isinstance(c, _str_types) for c in chk), chk

    for i, (data, pos, chk) in enumerate(test):
        assert all(isinstance(d, _str_types) for d in data), data
        assert all(isinstance(p, _str_types) for p in pos), pos
        assert all(isinstance(c, _str_types) for c in chk), chk


@flaky(max_runs=2, min_passes=1)
def test_conll2001():
    for part in range(1, 4):
        train = nlp.data.CoNLL2001(part, segment='train', root=os.path.join(
            'tests', 'externaldata', 'conll2001'))
        testa = nlp.data.CoNLL2001(part, segment='testa', root=os.path.join(
            'tests', 'externaldata', 'conll2001'))
        testb = nlp.data.CoNLL2001(part, segment='testb', root=os.path.join(
            'tests', 'externaldata', 'conll2001'))
        assert len(train) == 8936, len(train)
        assert len(testa) == 2012, len(testa)
        assert len(testb) == 1671, len(testb)

        for dataset in [train, testa, testb]:
            for i, (data, pos, chk, clause) in enumerate(dataset):
                assert all(isinstance(d, _str_types) for d in data), data
                assert all(isinstance(p, _str_types) for p in pos), pos
                assert all(isinstance(c, _str_types) for c in chk), chk
                assert all(isinstance(i, _str_types) for i in clause), clause


@flaky(max_runs=2, min_passes=1)
@pytest.mark.parametrize('segment,length', [
    ('train', 15806),
    ('testa', 2895),
    ('testb', 5195),
])
def test_conll2002_ned(segment, length):
    dataset = nlp.data.CoNLL2002('ned', segment=segment, root=os.path.join(
        'tests', 'externaldata', 'conll2002'))
    assert len(dataset) == length, len(dataset)
    for i, (data, pos, ner) in enumerate(dataset):
        assert all(isinstance(d, _str_types) for d in data), data
        assert all(isinstance(p, _str_types) for p in pos), pos
        assert all(isinstance(n, _str_types) for n in ner), ner


@flaky(max_runs=2, min_passes=1)
@pytest.mark.parametrize('segment,length', [
    ('train', 8323),
    ('testa', 1915),
    ('testb', 1517),
])
def test_conll2002_esp(segment, length):
    dataset = nlp.data.CoNLL2002('esp', segment=segment, root=os.path.join(
        'tests', 'externaldata', 'conll2002'))
    assert len(dataset) == length, len(dataset)
    for i, (data, ner) in enumerate(dataset):
        assert all(isinstance(d, _str_types) for d in data), data
        assert all(isinstance(n, _str_types) for n in ner), ner


@pytest.mark.skipif(datetime.date.today() < datetime.date(2018, 7, 7),
                    reason='Disabled for 1 weeks due to server downtime.')
@flaky(max_runs=2, min_passes=1)
@pytest.mark.parametrize('segment,length', [
    ('train', 8936),
    ('dev', 2012),
    ('test', 1671),
])
def test_conll2004(segment, length):
    dataset = nlp.data.CoNLL2004(segment=segment, root=os.path.join(
        'tests', 'externaldata', 'conll2004'))
    assert len(dataset) == length, len(dataset)

    for i, x in enumerate(dataset):
        assert len(x) >= 6, x
        assert all(isinstance(d, _str_types) for f in x for d in f), x
        assert max(len(f) for f in x) == min(len(f) for f in x), x


@flaky(max_runs=2, min_passes=1)
def test_ud21():
    test_langs = list(nlp._constants.UD21_DATA_FILE_SHA1.items())
    random.shuffle(test_langs)
    test_langs = test_langs[:30]
    for lang, segments in test_langs:
        segment = list(segments.keys())
        random.shuffle(segment)
        segment = segment[0]
        dataset = nlp.data.UniversalDependencies21(
            lang=lang, segment=segment, root=os.path.join(
                'tests', 'externaldata', 'ud2.1'))
        print('processing {}: {}'.format(lang, segment))
        for i, x in enumerate(dataset):
            assert len(x) >= 9, x
            assert all(isinstance(d, _str_types) for f in x for d in f), x
            assert max(len(f) for f in x) == min(len(f) for f in x)


###############################################################################
# Translation
###############################################################################
def test_iwlst2015():
    # Test en to vi
    train_en_vi = nlp.data.IWSLT2015(segment='train', root='tests/data/iwlst2015')
    val_en_vi = nlp.data.IWSLT2015(segment='val', root='tests/data/iwlst2015')
    test_en_vi = nlp.data.IWSLT2015(segment='test', root='tests/data/iwlst2015')
    assert len(train_en_vi) == 133166
    assert len(val_en_vi) == 1553
    assert len(test_en_vi) == 1268

    en_vocab, vi_vocab = train_en_vi.src_vocab, train_en_vi.tgt_vocab
    assert len(en_vocab) == 17191
    assert len(vi_vocab) == 7709

    train_vi_en = nlp.data.IWSLT2015(segment='train', src_lang='vi', tgt_lang='en',
                                     root='tests/data/iwlst2015')
    vi_vocab, en_vocab = train_vi_en.src_vocab, train_vi_en.tgt_vocab
    assert len(en_vocab) == 17191
    assert len(vi_vocab) == 7709
    for i in range(10):
        lhs = train_en_vi[i]
        rhs = train_vi_en[i]
        assert lhs[0] == rhs[1] and rhs[0] == lhs[1]


def test_wmt2016():
    train = nlp.data.WMT2016(segment='train', src_lang='en', tgt_lang='de',
                             root='tests/data/wmt2016')
    newstests = [nlp.data.WMT2016(segment='newstest%d' %i, src_lang='en', tgt_lang='de',
                                  root='tests/data/wmt2016') for i in range(2012, 2017)]
    assert len(train) == 4549428
    assert tuple(len(ele) for ele in newstests) == (3003, 3000, 3003, 2169, 2999)

    newstest_2012_2015 = nlp.data.WMT2016(segment=['newstest%d' %i for i in range(2012, 2016)],
                                          src_lang='en', tgt_lang='de', root='tests/data/wmt2016')
    assert len(newstest_2012_2015) == 3003 + 3000 + 3003 + 2169


def test_wmt2016bpe():
    train = nlp.data.WMT2016BPE(segment='train', src_lang='en', tgt_lang='de',
                                root='tests/data/wmt2016bpe')
    newstests = [nlp.data.WMT2016BPE(segment='newstest%d' %i, src_lang='en', tgt_lang='de',
                                     root='tests/data/wmt2016bpe') for i in range(2012, 2017)]
    assert len(train) == 4500966
    assert tuple(len(ele) for ele in newstests) == (3003, 3000, 3003, 2169, 2999)

    newstest_2012_2015 = nlp.data.WMT2016BPE(segment=['newstest%d' %i for i in range(2012, 2016)],
                                             src_lang='en', tgt_lang='de', root='tests/data/wmt2016bpe')
    assert len(newstest_2012_2015) == 3003 + 3000 + 3003 + 2169
    en_vocab, de_vocab = train.src_vocab, train.tgt_vocab
    assert len(en_vocab) == 36548
    assert len(de_vocab) == 36548


def test_wmt2014():
    train = nlp.data.WMT2014(segment='train', src_lang='en', tgt_lang='de',
                             root='tests/data/wmt2014')
    newstests = [nlp.data.WMT2014(segment='newstest%d' %i, src_lang='en', tgt_lang='de',
                                  root='tests/data/wmt2014') for i in range(2009, 2015)]
    assert len(train) == 4509333
    assert tuple(len(ele) for ele in newstests) == (2525, 2489, 3003, 3003, 3000, 2737)

    newstest_2009_2013 = nlp.data.WMT2014(segment=['newstest%d' %i for i in range(2009, 2014)],
                                          src_lang='en', tgt_lang='de', root='tests/data/wmt2014')
    assert len(newstest_2009_2013) == 2525 + 2489 + 3003 + 3003 + 3000

    newstest_2014 = nlp.data.WMT2014(segment='newstest2014', src_lang='de', tgt_lang='en',
                                     root='tests/data/wmt2014')
    assert len(newstest_2014) == 3003

    newstest_2014 = nlp.data.WMT2014(segment='newstest2014', src_lang='de', tgt_lang='en', full=True,
                                     root='tests/data/wmt2014')
    assert len(newstest_2014) == 3003


def test_wmt2014bpe():
    train = nlp.data.WMT2014BPE(segment='train', src_lang='en', tgt_lang='de',
                                root='tests/data/wmt2014bpe')
    newstests = [nlp.data.WMT2014BPE(segment='newstest%d' %i, src_lang='en', tgt_lang='de',
                                     root='tests/data/wmt2014bpe') for i in range(2009, 2015)]
    assert len(train) == 4493328
    assert tuple(len(ele) for ele in newstests) == (2525, 2489, 3003, 3003, 3000, 2737)

    newstest_2009_2013 = nlp.data.WMT2014BPE(segment=['newstest%d' %i for i in range(2009, 2014)],
                                             src_lang='en', tgt_lang='de', root='tests/data/wmt2014bpe')
    assert len(newstest_2009_2013) == 2525 + 2489 + 3003 + 3003 + 3000
    en_vocab, de_vocab = train.src_vocab, train.tgt_vocab
    assert len(en_vocab) == 36794
    assert len(de_vocab) == 36794

    newstest_2014 = nlp.data.WMT2014BPE(segment='newstest2014', src_lang='de', tgt_lang='en',
                                        root='tests/data/wmt2014bpe')
    assert len(newstest_2014) == 3003

    newstest_2014 = nlp.data.WMT2014BPE(segment='newstest2014', src_lang='de', tgt_lang='en', full=True,
                                        root='tests/data/wmt2014bpe')
    assert len(newstest_2014) == 3003

###############################################################################
# Stream
###############################################################################
def test_corpus_stream():
    EOS = nlp._constants.EOS_TOKEN
    path = os.path.join('tests', 'data', 'wikitext-2')
    token_path = os.path.join('tests', 'data', 'wikitext-2/*.tokens')
    train = nlp.data.WikiText2(segment='train', root=path)
    val = nlp.data.WikiText2(segment='val', root=path)
    test = nlp.data.WikiText2(segment='test', root=path)
    corpus = nlp.data.CorpusStream(token_path, flatten=True,
                                   skip_empty=True, eos=EOS)
    counter = nlp.data.Counter(corpus)
    assert len(counter) == 33278, len(counter)
    # examine aggregated vocab
    vocab = nlp.vocab.Vocab(counter, bos_token=None, padding_token=None)
    assert len(vocab) == 33278, len(vocab)
    # examine aggregated stats
    assert sum(counter.values()) == 2075677 + 216347 + 244102
    counter = nlp.data.Counter(corpus)
    assert len(counter) == 33278, len(counter)

def test_lm_stream():
    EOS = nlp._constants.EOS_TOKEN
    path = os.path.join('tests', 'data', 'wikitext-2')
    token_path = os.path.join('tests', 'data', 'wikitext-2/*.tokens')
    train = nlp.data.WikiText2(segment='train', root=path)
    val = nlp.data.WikiText2(segment='val', root=path)
    test = nlp.data.WikiText2(segment='test', root=path)
    lm_stream = nlp.data.LanguageModelStream(token_path, skip_empty=True, eos=EOS)
    counter = nlp.data.Counter(lm_stream)
    vocab = nlp.vocab.Vocab(counter, bos_token=None)
    seq_len = 35
    batch_size = 80
    bptt_stream = lm_stream.bptt_batchify(vocab, seq_len, batch_size, last_batch='keep')
    padding_idx = vocab[vocab.padding_token]
    total_num_tokens = sum(counter.values())
    num_tokens_per_batch = seq_len * batch_size
    num_tokens = 0
    for i, (data, target, mask) in enumerate(bptt_stream):
        # count the valid tokens in the batch
        num_valid_tokens = mask.sum().asscalar()
        if num_valid_tokens == num_tokens_per_batch:
            mx.test_utils.assert_almost_equal(data[1:].asnumpy(), target[:-1].asnumpy())
            assert data.shape == target.shape == (seq_len, batch_size)
        num_tokens += num_valid_tokens
    num_batches = sum(1 for _ in bptt_stream)
    # the last token doesn't appear in data
    assert num_tokens >= total_num_tokens - batch_size, num_tokens
    assert num_tokens < total_num_tokens, num_tokens

def test_lazy_stream():
    EOS = nlp._constants.EOS_TOKEN
    path = os.path.join('tests', 'data', 'wikitext-2')
    token_path = os.path.join('tests', 'data', 'wikitext-2/*test*.tokens')
    test = nlp.data.WikiText2(segment='test', root=path)
    corpus = nlp.data.CorpusStream(token_path, flatten=True,
                                   skip_empty=True, eos=EOS, sampler='sequential')
    transformed_corpus = nlp.data.SimpleDataStream(corpus).transform(lambda s: s.lower())
    for x, y in zip(corpus, transformed_corpus):
        assert y == x.lower()

###############################################################################
# Question answering
###############################################################################
def test_load_dev_squad():
    # number of records in dataset is equal to number of different questions
    train_dataset = nlp.data.SQuAD(segment='train', root='tests/data/squad')
    assert len(train_dataset) == 87599

    val_dataset = nlp.data.SQuAD(segment='dev', root='tests/data/squad')
    assert len(val_dataset) == 10570

    # Each record is a tuple of 6 elements: record_id, question Id, question, context,
    # list of answer texts, list of answer start indices
    for record in val_dataset:
        assert len(record) == 6

def test_counter():
    x = nlp.data.Counter({'a': 10, 'b': 1, 'c': 1})
    y = x.discard(3, '<unk>')
    assert y['a'] == 10
    assert y['<unk>'] == 2

# this test is not tested on CI due to long running time
def _test_gbw_stream():
    gbw = nlp.data.GBWStream()
    counter = nlp.data.Counter(gbw)
    counter.discard(3, '<unk>')
    # reference count obtained from:
    # https://github.com/rafaljozefowicz/lm/blob/master/1b_word_vocab.txt
    assert counter['the'] == 35936573
    assert counter['.'] == 29969612
