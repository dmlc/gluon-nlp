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
import itertools
import json
import os
import random

import numpy as np
import pytest
from flaky import flaky

import gluonnlp as nlp
import mxnet as mx
from gluonnlp.base import _str_types


def test_corpus_stream():
    EOS = nlp._constants.EOS_TOKEN
    path = os.path.join('tests', 'data', 'wikitext-2')
    token_path = os.path.join('tests', 'data', 'wikitext-2/*.tokens')
    train = nlp.data.WikiText2(segment='train', root=path)
    val = nlp.data.WikiText2(segment='val', root=path)
    test = nlp.data.WikiText2(segment='test', root=path)
    corpus = nlp.data.CorpusStream(token_path, flatten=True,
                                   skip_empty=True, eos=EOS)
    counter = nlp.data.Counter(itertools.chain.from_iterable(corpus))
    assert len(counter) == 33278, len(counter)
    # examine aggregated vocab
    vocab = nlp.vocab.Vocab(counter, bos_token=None, padding_token=None)
    assert len(vocab) == 33278, len(vocab)
    # examine aggregated stats
    assert sum(counter.values()) == 2075677 + 216347 + 244102
    counter = nlp.data.Counter(itertools.chain.from_iterable(corpus))
    assert len(counter) == 33278, len(counter)


@pytest.mark.parametrize('prefetch', [None, "thread", "process"])
def test_lazy_stream(prefetch):
    EOS = nlp._constants.EOS_TOKEN
    path = os.path.join('tests', 'data', 'wikitext-2')
    token_path = os.path.join('tests', 'data', 'wikitext-2/*test*.tokens')
    test = nlp.data.WikiText2(segment='test', root=path)
    corpus = nlp.data.CorpusStream(token_path, flatten=True, skip_empty=True, eos=EOS)
    if prefetch:
        prefetch_corpus = nlp.data.PrefetchingStream(corpus, worker_type=prefetch)
    else:
        prefetch_corpus = corpus
    transformed_corpus = prefetch_corpus.transform(lambda d: [s.lower() for s in d])
    prefetch_corpus_iter = iter(prefetch_corpus)
    transformed_corpus_iter = iter(transformed_corpus)
    for x in corpus:
        y = next(prefetch_corpus_iter)
        z = next(transformed_corpus_iter)
        assert all([sx.lower() == sy.lower() == sz for sx, sy, sz in zip(x, y, z)])


@pytest.mark.parametrize('num_prefetch', [0, 1, 10])
@pytest.mark.parametrize('worker_type', ['thread', 'process'])
def test_prefetch_stream(num_prefetch, worker_type):
    EOS = nlp._constants.EOS_TOKEN
    path = os.path.join('tests', 'data', 'wikitext-2')
    token_path = os.path.join('tests', 'data', 'wikitext-2/*test*.tokens')
    test = nlp.data.WikiText2(segment='test', root=path)
    corpus = nlp.data.CorpusStream(token_path, flatten=True, skip_empty=True, eos=EOS)
    if num_prefetch < 1:
        with pytest.raises(ValueError):
            prefetch_corpus = nlp.data.PrefetchingStream(
                corpus, num_prefetch=num_prefetch, worker_type=worker_type)
    else:
        prefetch_corpus = nlp.data.PrefetchingStream(
            corpus, num_prefetch=num_prefetch, worker_type=worker_type)
        prefetch_corpus_iter = iter(prefetch_corpus)
        for x in corpus:
            y = next(prefetch_corpus_iter)
            assert all([sx == sy for sx, sy in zip(x, y)])


###############################################################################
# Language model
###############################################################################
def test_lm_stream():
    EOS = nlp._constants.EOS_TOKEN
    path = os.path.join('tests', 'data', 'wikitext-2')
    token_path = os.path.join('tests', 'data', 'wikitext-2/*.tokens')
    train = nlp.data.WikiText2(segment='train', root=path)
    val = nlp.data.WikiText2(segment='val', root=path)
    test = nlp.data.WikiText2(segment='test', root=path)
    lm_stream = nlp.data.LanguageModelStream(token_path, skip_empty=True, eos=EOS, bos=EOS)
    counter = nlp.data.Counter(itertools.chain.from_iterable(lm_stream))
    vocab = nlp.vocab.Vocab(counter, bos_token=None)
    seq_len = 35
    batch_size = 80
    bptt_stream = lm_stream.bptt_batchify(vocab, seq_len, batch_size, last_batch='keep')
    padding_idx = vocab[vocab.padding_token]
    total_num_tokens = sum(counter.values())
    num_tokens_per_batch = seq_len * batch_size
    num_tokens = 0
    for i, (data, target) in enumerate(bptt_stream):
        # count the valid tokens in the batch
        mask = data == padding_idx
        num_valid_tokens = mask.sum().asscalar()
        if num_valid_tokens == num_tokens_per_batch:
            mx.test_utils.assert_almost_equal(data[1:].asnumpy(), target[:-1].asnumpy())
            assert data.shape == target.shape == (seq_len, batch_size)
        num_tokens += num_valid_tokens
    num_batches = sum(1 for _ in bptt_stream)
    # the last token doesn't appear in data
    assert num_tokens < total_num_tokens


###############################################################################
# Embedding training
###############################################################################
@pytest.mark.parametrize('reduce_window_size_randomly', [True, False])
@pytest.mark.parametrize('shuffle', [True, False])
def test_context_stream(reduce_window_size_randomly, shuffle):
    data = nlp.data.Text8(segment='train')[:3]
    counter = nlp.data.count_tokens(itertools.chain.from_iterable(data))
    vocab = nlp.Vocab(counter)
    data = [vocab[sentence] for sentence in data]
    data = nlp.data.SimpleDataStream([data, data])
    idx_to_pdiscard = [0] * len(vocab)

    context_stream = nlp.data.ContextStream(
        stream=data, batch_size=8, p_discard=idx_to_pdiscard, window_size=5,
        reduce_window_size_randomly=reduce_window_size_randomly,
        shuffle=shuffle)

    assert len(list(context_stream)) == 7500
