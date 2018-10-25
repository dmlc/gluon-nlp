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
@pytest.mark.parametrize('cbow', [True, False])
@pytest.mark.parametrize('stream', [True, False])
def test_center_context_batchify_stream(reduce_window_size_randomly, shuffle,
                                        cbow, stream):
    dataset = [np.arange(100).tolist()] * 3
    batchify = nlp.data.batchify.EmbeddingCenterContextBatchify(
        batch_size=8,
        window_size=5,
        reduce_window_size_randomly=reduce_window_size_randomly,
        shuffle=shuffle,
        cbow=cbow)
    if stream:
        stream = nlp.data.SimpleDataStream([dataset, dataset])
        batches = list(
            itertools.chain.from_iterable(stream.transform(batchify)))
    else:
        samples = batchify(dataset)
        batches = list(samples)
    if cbow:
        assert len(batches) == 37 if not stream else 74
    elif not reduce_window_size_randomly:
        assert len(batches) == 363 if not stream else 726
    else:
        pass


@pytest.mark.parametrize('cbow', [True, False])
def test_center_context_batchify(cbow):
    dataset = [np.arange(100).tolist()]
    batchify = nlp.data.batchify.EmbeddingCenterContextBatchify(
        batch_size=3, window_size=1, cbow=cbow)
    samples = batchify(dataset)

    center, context = next(iter(samples))
    (contexts_data, contexts_row, contexts_col) = context

    assert center.dtype == np.int64
    assert contexts_data.dtype == np.float32
    assert contexts_row.dtype == np.int64
    assert contexts_col.dtype == np.int64

    if cbow:
        assert center.asnumpy().tolist() == [0, 1, 2]
        assert contexts_data.asnumpy().tolist() == [1, 0.5, 0.5, 0.5, 0.5]
        assert contexts_row.asnumpy().tolist() == [0, 1, 1, 2, 2]
        assert contexts_col.asnumpy().tolist() == [1, 0, 2, 1, 3]
    else:
        assert center.asnumpy().tolist() == [0, 1, 1]
        assert contexts_data.asnumpy().tolist() == [1, 1, 1]
        assert contexts_row.asnumpy().tolist() == [0, 1, 2]
        assert contexts_col.asnumpy().tolist() == [1, 0, 2]
