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

import pytest
import numpy as np
import mxnet as mx

import os
import gluonnlp as nlp


@pytest.fixture(params=[True, False])
def sparse_grad(request):
    return request.param


@pytest.fixture(params=[True, False])
def hybridize(request):
    return request.param


@pytest.mark.parametrize('wordsmask', [True, False])
def test_simple_embedding(sparse_grad, hybridize, wordsmask):
    token_to_idx = dict(hello=0, world=1)
    embedding = nlp.model.train.SimpleEmbeddingModel(token_to_idx, 30,
                                                     sparse_grad=sparse_grad)
    embedding.initialize()
    if hybridize:
        embedding.hybridize()

    # Without mask
    words = mx.nd.arange(2)
    with mx.autograd.record():
        loss = embedding(words, wordsmask=mx.nd.ones_like(words)
                         if wordsmask else None)
    loss.backward()
    loss.asnumpy()


@pytest.mark.parametrize('wordsmask', [True, False])
@pytest.mark.parametrize('subwordsmask', [True, False])
def test_fasttext_embedding(sparse_grad, hybridize, wordsmask, subwordsmask):
    token_to_idx = dict(hello=0, world=1)
    subwords = nlp.vocab.create_subword_function(
        'NGramHashes', ngrams=[3, 4, 5, 6], num_subwords=1000)
    embedding = nlp.model.train.FasttextEmbeddingModel(
        token_to_idx, subwords, 30, sparse_grad=sparse_grad)
    embedding.initialize()
    if hybridize:
        embedding.hybridize()

    words = mx.nd.arange(2).reshape((1, -1))
    subwords = words.reshape((1, -1, 1))
    with mx.autograd.record():
        loss = embedding(words, subwords, wordsmask=mx.nd.ones_like(words)
                         if wordsmask else None,
                         subwordsmask=mx.nd.ones_like(subwords)
                         if subwordsmask else None).sum()
    loss.backward()
    loss.asnumpy()

    # With word deduplication
    subwords = mx.nd.arange(1).reshape((1, 1))
    with mx.autograd.record():
        loss = embedding(
            words, subwords, wordsmask=mx.nd.ones_like(words)
            if wordsmask else None, subwordsmask=mx.nd.ones_like(subwords)
            if subwordsmask else None,
            words_to_unique_subwords_indices=mx.nd.arange(2)).sum()
    loss.backward()
    loss.asnumpy()


def test_fasttext_embedding_load_binary_compare_vec():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    token_embedding_vec = nlp.embedding.TokenEmbedding.from_file(
        os.path.join(str(test_dir), 'test_embedding', 'lorem_ipsum.vec'),
        unknown_token=None)

    model = nlp.model.train.FasttextEmbeddingModel.load_fasttext_format(
        os.path.join(str(test_dir), 'test_embedding', 'lorem_ipsum.bin'))
    idx_to_vec = model[token_embedding_vec.idx_to_token]
    assert np.all(
        np.isclose(a=token_embedding_vec.idx_to_vec.asnumpy(),
                   b=idx_to_vec.asnumpy(), atol=0.001))
    assert all(token in model for token in token_embedding_vec.idx_to_token)
