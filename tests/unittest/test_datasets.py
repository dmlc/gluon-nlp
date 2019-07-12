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
import os
import io
import random

from flaky import flaky
import mxnet as mx
import numpy as np
import pytest

import gluonnlp as nlp
from gluonnlp.base import _str_types
from mxnet.gluon.data import SimpleDataset

###############################################################################
# Registry
###############################################################################
@pytest.mark.serial
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
# Sentiment analysis
###############################################################################
@pytest.mark.serial
@pytest.mark.remote_required
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

@pytest.mark.serial
@pytest.mark.remote_required
def test_mr():
    all = nlp.data.MR(
        root=os.path.join('tests', 'data', 'mr'))
    assert len(all) == 10662, len(all)
    for i, (data, label) in enumerate(all):
        assert isinstance(data, _str_types)
        assert label <= 1

@pytest.mark.serial
@pytest.mark.remote_required
def test_sst_1():
    train = nlp.data.SST_1(
        root=os.path.join('tests', 'data', 'sst-1'), segment='train')
    test = nlp.data.SST_1(
        root=os.path.join('tests', 'data', 'sst-1'), segment='test')
    dev = nlp.data.SST_1(
        root=os.path.join('tests', 'data', 'sst-1'), segment='dev')
    assert len(train) == 156817, len(train)
    assert len(test) == 2210, len(test)
    assert len(dev) == 1101, len(dev)
    for i, (data, label) in enumerate(train):
        assert isinstance(data, _str_types)
        assert label <= 4
    for i, (data, label) in enumerate(test):
        assert isinstance(data, _str_types)
        assert label <= 4
    for i, (data, label) in enumerate(dev):
        assert isinstance(data, _str_types)
        assert label <= 4

@pytest.mark.serial
@pytest.mark.remote_required
def test_sst_2():
    train = nlp.data.SST_2(
        root=os.path.join('tests', 'data', 'sst-2'), segment='train')
    test = nlp.data.SST_2(
        root=os.path.join('tests', 'data', 'sst-2'), segment='test')
    dev = nlp.data.SST_2(
        root=os.path.join('tests', 'data', 'sst-2'), segment='dev')
    assert len(train) == 76961, len(train)
    assert len(test) == 1821, len(test)
    assert len(dev) == 872, len(dev)
    for i, (data, label) in enumerate(train):
        assert isinstance(data, _str_types)
        assert label <= 1
    for i, (data, label) in enumerate(test):
        assert isinstance(data, _str_types)
        assert label <= 1
    for i, (data, label) in enumerate(dev):
        assert isinstance(data, _str_types)
        assert label <= 1

@pytest.mark.serial
@pytest.mark.remote_required
def test_subj():
    all = nlp.data.SUBJ(
        root=os.path.join('tests', 'data', 'mr'))
    assert len(all) == 10000, len(all)
    for i, (data, label) in enumerate(all):
        assert isinstance(data, _str_types)
        assert label <= 1

@pytest.mark.serial
@pytest.mark.remote_required
def test_trec():
    train = nlp.data.TREC(
        root=os.path.join('tests', 'data', 'trec'), segment='train')
    test = nlp.data.TREC(
        root=os.path.join('tests', 'data', 'trec'), segment='test')
    assert len(train) == 5452, len(train)
    assert len(test) == 500, len(test)
    for i, (data, label) in enumerate(train):
        assert isinstance(data, _str_types)
        assert label <= 5
    for i, (data, label) in enumerate(test):
        assert isinstance(data, _str_types)
        assert label <= 5

@pytest.mark.serial
@pytest.mark.remote_required
def test_cr():
    all = nlp.data.CR(
        root=os.path.join('tests', 'data', 'cr'))
    assert len(all) == 3775, len(all)
    for i, (data, label) in enumerate(all):
        assert isinstance(data, _str_types)
        assert label <= 1

@pytest.mark.serial
@pytest.mark.remote_required
def test_mpqa():
    all = nlp.data.MPQA(
        root=os.path.join('tests', 'data', 'mpqa'))
    assert len(all) == 10606, len(all)
    for i, (data, label) in enumerate(all):
        assert isinstance(data, _str_types)
        assert label <= 1

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
@pytest.mark.serial
@pytest.mark.remote_required
def test_wordsim353(segment, length):
    # 'all' has length 352 as the original dataset contains the 'money/cash'
    # pair twice with different similarity ratings, which was fixed by the
    # http://alfonseca.org/eng/research/wordsim353.html version of the dataset
    # that we are using.
    data = nlp.data.WordSim353(segment=segment, root=os.path.join(
        'tests', 'externaldata', 'wordsim353'))
    assert len(data) == length, len(data)
    _assert_similarity_dataset(data)


@pytest.mark.serial
@pytest.mark.remote_required
def test_men():
    for segment, length in [("full", 3000), ("dev", 2000), ("test", 1000)]:
        data = nlp.data.MEN(
            root=os.path.join('tests', 'data', 'men'), segment=segment)
        assert len(data) == length, len(data)
        _assert_similarity_dataset(data)


@flaky(max_runs=2, min_passes=1)
@pytest.mark.serial
@pytest.mark.remote_required
def test_radinsky_mturk():
    data = nlp.data.RadinskyMTurk(
        root=os.path.join('tests', 'externaldata', 'radinsky'))
    assert len(data) == 287
    _assert_similarity_dataset(data)


@flaky(max_runs=2, min_passes=1)
@pytest.mark.serial
@pytest.mark.remote_required
@pytest.mark.skipif(datetime.date.today() < datetime.date(2019, 7, 25),
                    reason="verb143 temporarily unavailable.")
def test_verb143():
    data = nlp.data.BakerVerb143(
        root=os.path.join('tests', 'externaldata', 'verb143'))
    assert len(data) == 144
    _assert_similarity_dataset(data)


@flaky(max_runs=2, min_passes=1)
@pytest.mark.serial
def test_verb130():
    data = nlp.data.YangPowersVerb130(
        root=os.path.join('tests', 'externaldata', 'verb130'))
    assert len(data) == 130
    _assert_similarity_dataset(data)


@flaky(max_runs=2, min_passes=1)
@pytest.mark.serial
@pytest.mark.remote_required
def test_rare_words():
    data = nlp.data.RareWords(
        root=os.path.join('tests', 'externaldata', 'rarewords'))
    assert len(data) == 2034
    _assert_similarity_dataset(data)


@flaky(max_runs=2, min_passes=1)
@pytest.mark.serial
@pytest.mark.remote_required
def test_simlex999():
    data = nlp.data.SimLex999(
        root=os.path.join('tests', 'externaldata', 'simlex999'))
    assert len(data) == 999
    _assert_similarity_dataset(data)


@flaky(max_runs=2, min_passes=1)
@pytest.mark.serial
@pytest.mark.remote_required
def test_simverb3500():
    data = nlp.data.SimVerb3500(
        root=os.path.join('tests', 'externaldata', 'simverb3500'))
    assert len(data) == 3500
    _assert_similarity_dataset(data)


@flaky(max_runs=2, min_passes=1)
@pytest.mark.serial
@pytest.mark.remote_required
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
@pytest.mark.serial
@pytest.mark.remote_required
def test_googleanalogy():
    data = nlp.data.GoogleAnalogyTestSet(
        root=os.path.join('tests', 'externaldata', 'google_analogy'))
    assert len(data[0]) == 4
    assert len(data) == 10675 + 8869


@flaky(max_runs=2, min_passes=1)
@pytest.mark.serial
@pytest.mark.remote_required
def test_bigger_analogy():
    data = nlp.data.BiggerAnalogyTestSet(
        root=os.path.join('tests', 'externaldata', 'bigger_analogy'))
    assert len(data[0]) == 4
    assert len(data) == 98000


###############################################################################
# CONLL
###############################################################################
@flaky(max_runs=2, min_passes=1)
@pytest.mark.serial
@pytest.mark.remote_required
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
@pytest.mark.serial
@pytest.mark.remote_required
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
@pytest.mark.serial
@pytest.mark.remote_required
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
@pytest.mark.serial
@pytest.mark.remote_required
def test_conll2002_esp(segment, length):
    dataset = nlp.data.CoNLL2002('esp', segment=segment, root=os.path.join(
        'tests', 'externaldata', 'conll2002'))
    assert len(dataset) == length, len(dataset)
    for i, (data, ner) in enumerate(dataset):
        assert all(isinstance(d, _str_types) for d in data), data
        assert all(isinstance(n, _str_types) for n in ner), ner


@flaky(max_runs=2, min_passes=1)
@pytest.mark.parametrize('segment,length', [
    ('train', 8936),
    ('dev', 2012),
    ('test', 1671),
])
@pytest.mark.serial
@pytest.mark.remote_required
def test_conll2004(segment, length):
    dataset = nlp.data.CoNLL2004(segment=segment, root=os.path.join(
        'tests', 'externaldata', 'conll2004'))
    assert len(dataset) == length, len(dataset)

    for i, x in enumerate(dataset):
        assert len(x) >= 6, x
        assert all(isinstance(d, _str_types) for f in x for d in f), x
        assert max(len(f) for f in x) == min(len(f) for f in x), x


@flaky(max_runs=2, min_passes=1)
@pytest.mark.serial
@pytest.mark.remote_required
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
@pytest.mark.serial
@pytest.mark.remote_required
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


@pytest.mark.serial
@pytest.mark.remote_required
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


@pytest.mark.serial
@pytest.mark.remote_required
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


@pytest.mark.serial
@pytest.mark.remote_required
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


@pytest.mark.serial
@pytest.mark.remote_required
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
# Question answering
###############################################################################
@pytest.mark.serial
@pytest.mark.remote_required
def test_load_dev_squad():
    # number of records in dataset is equal to number of different questions
    train_dataset = nlp.data.SQuAD(
        segment='train', version='1.1', root='tests/data/squad')
    assert len(train_dataset) == 87599

    val_dataset = nlp.data.SQuAD(
        segment='dev',version='1.1', root='tests/data/squad')
    assert len(val_dataset) == 10570

    # Each record is a tuple of 6 elements: record_id, question Id, question, context,
    # list of answer texts, list of answer start indices
    for record in val_dataset:
        assert len(record) == 6

    train_dataset_2 = nlp.data.SQuAD(
        segment='train', version='2.0', root='tests/data/squad')
    assert len(train_dataset_2) == 130319

    val_dataset = nlp.data.SQuAD(
        segment='dev', version='2.0', root='tests/data/squad')
    assert len(val_dataset) == 11873

    # Each record is a tuple of 7 elements: record_id, question Id, question, context,
    # list of answer texts, list of answer start indices, is_impossible
    for record in val_dataset:
        assert len(record) == 7

###############################################################################
# Intent Classification and Slot Labeling
###############################################################################
@pytest.mark.remote_required
@pytest.mark.parametrize('dataset,segment,expected_samples', [
    ('atis', 'train', 4478),
    ('atis', 'dev', 500),
    ('atis', 'test', 893),
    ('snips', 'train', 13084),
    ('snips', 'dev', 700),
    ('snips', 'test', 700)])
def test_intent_slot(dataset, segment, expected_samples):
    assert dataset in ['atis', 'snips']
    if dataset == 'atis':
        data_cls = nlp.data.ATISDataset
    else:
        data_cls = nlp.data.SNIPSDataset

    dataset = data_cls(segment=segment, root='tests/data/{}/{}'.format(dataset, segment))

    assert len(dataset) == expected_samples
    assert len(dataset[0]) == 3
    assert all(len(x[0]) == len(x[1]) for x in dataset)

def test_counter():
    x = nlp.data.Counter({'a': 10, 'b': 1, 'c': 1})
    y = x.discard(3, '<unk>')
    assert y['a'] == 10
    assert y['<unk>'] == 2

# this test is not tested on CI due to long running time
def _test_gbw_stream():
    gbw = nlp.data.GBWStream(root=os.path.join('tests', 'data', 'gbw'))
    counter = nlp.data.Counter(gbw)
    counter.discard(3, '<unk>')
    # reference count obtained from:
    # https://github.com/rafaljozefowicz/lm/blob/master/1b_word_vocab.txt
    assert counter['the'] == 35936573
    assert counter['.'] == 29969612
    vocab = gbw.vocab
    assert len(vocab) == 793471


def test_concatenation():
    datasets = [
            SimpleDataset([1,2,3,4]),
            SimpleDataset([5,6]),
            SimpleDataset([8,0,9]),
            ]
    dataset = nlp.data.ConcatDataset(datasets)
    assert len(dataset) == 9
    assert dataset[0] == 1
    assert dataset[5] == 6

def test_tsv():
    data =  "a,b,c\n"
    data += "d,e,f\n"
    data += "g,h,i\n"
    with open('test_tsv.tsv', 'w') as fout:
        fout.write(data)
    num_discard = 1
    field_separator = nlp.data.utils.Splitter(',')
    field_indices = [0,2]
    dataset = nlp.data.TSVDataset('test_tsv.tsv', num_discard_samples=num_discard,
                                  field_separator=field_separator,
                                  field_indices=field_indices)
    num_samples = 3 - num_discard
    idx = random.randint(0, num_samples - 1)
    assert len(dataset) == num_samples
    assert len(dataset[0]) == 2
    assert dataset[1] == [u'g', u'i']

def test_numpy_dataset():
    a = np.arange(6).reshape((2,3))
    filename = 'test_numpy_dataset'

    # test npy
    np.save(filename, a)
    dataset = nlp.data.NumpyDataset(filename + '.npy')
    assert dataset.keys is None
    assert len(dataset) == len(a)
    assert np.all(dataset[0] == a[0])
    assert np.all(dataset[1] == a[1])

    # test npz with a single array
    np.savez(filename, a)
    dataset = nlp.data.NumpyDataset(filename + '.npz')
    assert len(dataset) == len(a)
    assert np.all(dataset[0] == a[0])
    assert np.all(dataset[1] == a[1])

    # test npz with multiple arrays
    b = np.arange(16).reshape((2,8))
    np.savez(filename, a=a, b=b)
    dataset = nlp.data.NumpyDataset(filename + '.npz')
    assert dataset.keys == ['a', 'b']
    assert len(dataset) == len(a)
    assert np.all(dataset[0][0] == a[0])
    assert np.all(dataset[1][0] == a[1])
    assert np.all(dataset[0][1] == b[0])
    assert np.all(dataset[1][1] == b[1])
    dataset_b = dataset.get_field('b')
    assert np.all(dataset_b == b)

@pytest.mark.parametrize('cls,name,segment,length,fields', [
    (nlp.data.GlueCoLA, 'cola', 'train', 8551, 2),
    (nlp.data.GlueCoLA, 'cola', 'dev', 1043, 2),
    (nlp.data.GlueCoLA, 'cola', 'test', 1063, 1),
    # source: https://arxiv.org/pdf/1804.07461.pdf
    (nlp.data.GlueSST2, 'sst', 'train', 67349, 2),
    (nlp.data.GlueSST2, 'sst', 'dev', 872, 2),
    (nlp.data.GlueSST2, 'sst', 'test', 1821, 1),
    # source: http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark
    (nlp.data.GlueSTSB, 'sts', 'train', 5749, 3),
    (nlp.data.GlueSTSB, 'sts', 'dev', 1500, 3),
    (nlp.data.GlueSTSB, 'sts', 'test', 1379, 2),
    # source: https://data.quora.com/First-Quora-Dataset-Release-Question-Pairs
    (nlp.data.GlueQQP, 'qqp', 'train', 363849, 3),
    (nlp.data.GlueQQP, 'qqp', 'dev', 40430, 3),
    (nlp.data.GlueQQP, 'qqp', 'test', 390965, 2),
    # source: http://www.nyu.edu/projects/bowman/multinli/paper.pdf
    (nlp.data.GlueMNLI, 'mnli', 'train', 392702, 3),
    (nlp.data.GlueMNLI, 'mnli', 'dev_matched', 9815, 3),
    (nlp.data.GlueMNLI, 'mnli', 'dev_mismatched', 9832, 3),
    (nlp.data.GlueMNLI, 'mnli', 'test_matched', 9796, 2),
    (nlp.data.GlueMNLI, 'mnli', 'test_mismatched', 9847, 2),
    # source: https://arxiv.org/pdf/1804.07461.pdf
    (nlp.data.GlueRTE, 'rte', 'train', 2490, 3),
    (nlp.data.GlueRTE, 'rte', 'dev', 277, 3),
    (nlp.data.GlueRTE, 'rte', 'test', 3000, 2),
    # source: https://arxiv.org/pdf/1804.07461.pdf
    (nlp.data.GlueQNLI, 'qnli', 'train', 108436, 3),
    (nlp.data.GlueQNLI, 'qnli', 'dev', 5732, 3),
    (nlp.data.GlueQNLI, 'qnli', 'test', 5740, 2),
    # source: https://arxiv.org/pdf/1804.07461.pdf
    (nlp.data.GlueWNLI, 'wnli', 'train', 635, 3),
    (nlp.data.GlueWNLI, 'wnli', 'dev', 71, 3),
    (nlp.data.GlueWNLI, 'wnli', 'test', 146, 2),
    (nlp.data.GlueMRPC, 'mrpc', 'train', 3668, 3),
    (nlp.data.GlueMRPC, 'mrpc', 'dev', 408, 3),
    (nlp.data.GlueMRPC, 'mrpc', 'test', 1725, 2),
])
@pytest.mark.serial
@pytest.mark.remote_required
def test_glue_data(cls, name, segment, length, fields):
    dataset = cls(segment=segment, root=os.path.join(
        'tests', 'externaldata', 'glue', name))
    assert len(dataset) == length, len(dataset)

    for i, x in enumerate(dataset):
        assert len(x) == fields, x
