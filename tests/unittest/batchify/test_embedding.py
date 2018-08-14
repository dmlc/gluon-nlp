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

import gluonnlp as nlp


@pytest.mark.parametrize('reduce_window_size_randomly', [True, False])
@pytest.mark.parametrize('shuffle', [True, False])
def test_center_context_batchify(reduce_window_size_randomly, shuffle):
    data = nlp.data.Text8(segment='train')[:3]
    counter = nlp.data.count_tokens(itertools.chain.from_iterable(data))
    vocab = nlp.Vocab(counter)
    data = [vocab[sentence] for sentence in data]
    data = nlp.data.SimpleDataStream([data, data])
    idx_to_pdiscard = [0] * len(vocab)

    context_stream = nlp.data.batchify.EmbeddingCenterContextBatchify(
        stream=data, batch_size=8, p_discard=idx_to_pdiscard, window_size=5,
        reduce_window_size_randomly=reduce_window_size_randomly,
        shuffle=shuffle)

    assert len(list(context_stream)) == 7500
