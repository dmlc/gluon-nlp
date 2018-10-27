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

import itertools
import os
import pytest
import gluonnlp as nlp

@pytest.mark.serial
def test_corpus_stream(
        stream_identity_wrappers,
        wikitext2_simpledatasetstream_skipempty_flatten_and_counter):
    stream, _ = wikitext2_simpledatasetstream_skipempty_flatten_and_counter

    stream = stream_identity_wrappers(stream)
    counter = nlp.data.Counter(itertools.chain.from_iterable(stream))
    assert len(counter) == 33278, len(counter)
    # examine aggregated vocab
    vocab = nlp.vocab.Vocab(counter, bos_token=None, padding_token=None)
    assert len(vocab) == 33278, len(vocab)
    # examine aggregated stats
    assert sum(counter.values()) == 2075677 + 216347 + 244102
    counter = nlp.data.Counter(itertools.chain.from_iterable(stream))
    assert len(counter) == 33278, len(counter)


@pytest.mark.serial
def test_lazy_stream(stream_identity_wrappers):
    EOS = nlp._constants.EOS_TOKEN
    path = os.path.join('tests', 'data', 'wikitext-2')
    token_path = os.path.join('tests', 'data', 'wikitext-2/*test*.tokens')
    corpus = nlp.data.WikiText2(segment='test', root=path)

    # We don't use wikitext2_simpledatasetstream_skipempty_flatten_and_counter
    # here as there is no need to work on more than the 'test' segment. The
    # fixture goes over all segments.
    stream = nlp.data.SimpleDatasetStream(
        nlp.data.CorpusDataset,
        token_path,
        flatten=True,
        skip_empty=True,
        eos=EOS)
    wrapped_stream = stream_identity_wrappers(stream)
    transformed_stream = wrapped_stream.transform(lambda d: [s.lower() for s in d])

    wrapped_stream_iter = iter(wrapped_stream)
    transformed_stream_iter = iter(transformed_stream)
    for dataset in stream:
        prefetched_dataset = next(wrapped_stream_iter)
        transformed_dataset = next(transformed_stream_iter)
        assert all([
            w1.lower() == w2.lower() == w3 == w4.lower() for w1, w2, w3, w4 in
            zip(dataset, prefetched_dataset, transformed_dataset, corpus)
        ])


@pytest.mark.parametrize('num_prefetch', [0, 1, 10])
@pytest.mark.parametrize('worker_type', ['thread', 'process'])
@pytest.mark.serial
def test_prefetch_stream(num_prefetch, worker_type):
    EOS = nlp._constants.EOS_TOKEN
    path = os.path.join('tests', 'data', 'wikitext-2')
    token_path = os.path.join('tests', 'data', 'wikitext-2/*test*.tokens')
    test = nlp.data.WikiText2(segment='test', root=path)
    corpus = nlp.data.SimpleDatasetStream(
        nlp.data.CorpusDataset, token_path, flatten=True, skip_empty=True)
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
