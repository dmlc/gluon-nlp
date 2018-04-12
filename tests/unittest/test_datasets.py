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
import mxnet as mx
import gluonnlp as nlp


def test_wikitext2():
    train = nlp.data.WikiText2(segment='train', root='tests/data/wikitext-2')
    val = nlp.data.WikiText2(segment='val', root='tests/data/wikitext-2')
    test = nlp.data.WikiText2(segment='test', root='tests/data/wikitext-2')
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

    train = nlp.data.WikiText2(segment='train', skip_empty=False,
                               root='tests/data/wikitext-2')
    val = nlp.data.WikiText2(segment='val', skip_empty=False,
                             root='tests/data/wikitext-2')
    test = nlp.data.WikiText2(segment='test', skip_empty=False,
                              root='tests/data/wikitext-2')
    train_freq, val_freq, test_freq = [nlp.data.utils.Counter(x) for x in [train[0], val[0], test[0]]]
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
    train = nlp.data.IMDB(root='tests/data/imdb', segment='train')
    test = nlp.data.IMDB(root='tests/data/imdb', segment='test')
    unsup = nlp.data.IMDB(root='tests/data/imdb', segment='unsup')
    assert len(train) == 25000, len(train)
    assert len(test) == 25000, len(test)
    assert len(unsup) == 50000, len(unsup)

    import sys
    if sys.version_info[0] == 3:
        str_types = (str,)
    else:
        str_types = (str, unicode)

    for i, (data, score) in enumerate(train):
        assert isinstance(data, str_types)
        assert score <= 4 or score >= 7

    for i, (data, score) in enumerate(test):
        assert isinstance(data, str_types)
        assert score <= 4 or score >= 7

    for i, (data, score) in enumerate(unsup):
        assert isinstance(data, str_types)
        assert score == 0


def test_iwlst2015():
    # Test en to vi
    train_en_vi = nlp.data.IWSLT2015(segment='train', root='tests/data/iwlst2015')
    val_en_vi = nlp.data.IWSLT2015(segment='val', root='tests/data/iwlst2015')
    test_en_vi = nlp.data.IWSLT2015(segment='test', root='tests/data/iwlst2015')
    assert len(train_en_vi) == 133317
    assert len(val_en_vi) == 1553
    assert len(test_en_vi) == 1268

    en_vocab, vi_vocab = train_en_vi.get_vocab()
    assert len(en_vocab) == 17191
    assert len(vi_vocab) == 7709

    train_vi_en = nlp.data.IWSLT2015(segment='train', src_lang='vi', tgt_lang='en',
                                     root='tests/data/iwlst2015')
    vi_vocab, en_vocab = train_vi_en.get_vocab()
    assert len(en_vocab) == 17191
    assert len(vi_vocab) == 7709
    for i in range(10):
        lhs = train_en_vi[i]
        rhs = train_vi_en[i]
        assert lhs[0] == rhs[1] and rhs[0] == lhs[1]


def test_wmt2016():
    train = nlp.data.WMT2016(segment='train', src_lang='en', tgt_lang='de',
                             root='tests/data/wmt2016')
    val = nlp.data.WMT2016(segment='val', src_lang='en', tgt_lang='de',
                           root='tests/data/wmt2016')
    test = nlp.data.WMT2016(segment='test', src_lang='en', tgt_lang='de',
                            root='tests/data/wmt2016')
    assert len(train) == 4500966
    assert len(val) == 3000
    assert len(test) == 2169
    en_vocab, de_vocab = train.get_vocab()
    assert len(en_vocab) == 36548
    assert len(de_vocab) == 36548


if __name__ == '__main__':
    import nose
    nose.runmodule()
