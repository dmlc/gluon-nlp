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

import pytest

import numpy as np
import gluonnlp as nlp


@pytest.mark.parametrize('reduce_window_size_randomly', [True, False])
@pytest.mark.parametrize('shuffle', [True, False])
def test_center_context_batchify_dataset(reduce_window_size_randomly, shuffle):
    data = nlp.data.Text8(segment='train')[:3]
    counter = nlp.data.count_tokens(itertools.chain.from_iterable(data))
    vocab = nlp.Vocab(counter)
    data = [vocab[sentence] for sentence in data]

    idx_to_pdiscard = [0] * len(vocab)
    batchify = nlp.data.batchify.EmbeddingCenterContextBatchify(
        batch_size=8,
        window_size=5,
        reduce_window_size_randomly=reduce_window_size_randomly,
        shuffle=shuffle)

    batches = list(batchify(data))
    assert len(batches) == 3750  # given fixed batch_size == 8


@pytest.mark.parametrize('reduce_window_size_randomly', [True, False])
@pytest.mark.parametrize('shuffle', [True, False])
def test_center_context_batchify_stream(reduce_window_size_randomly, shuffle):
    data = nlp.data.Text8(segment='train')[:3]
    counter = nlp.data.count_tokens(itertools.chain.from_iterable(data))
    vocab = nlp.Vocab(counter)
    data = [vocab[sentence] for sentence in data]

    idx_to_pdiscard = [0] * len(vocab)
    batchify = nlp.data.batchify.EmbeddingCenterContextBatchify(
        batch_size=8,
        window_size=5,
        reduce_window_size_randomly=reduce_window_size_randomly,
        shuffle=shuffle)

    stream = nlp.data.SimpleDataStream([data, data])
    batches = list(itertools.chain.from_iterable(stream.transform(batchify)))
    assert len(batches) == 7500


def test_center_context_batchify_artificial():
    dataset = [np.arange(1000).tolist()]
    batchify = nlp.data.batchify.EmbeddingCenterContextBatchify(
        batch_size=2, window_size=1)
    samples = batchify(dataset)

    center, context, mask = next(iter(samples))

    assert center.asnumpy().tolist() == [[0], [1]]
    assert context.asnumpy().tolist() == [[1, 0], [0, 2]]
    assert mask.asnumpy().tolist() == [[1, 0], [1, 1]]
