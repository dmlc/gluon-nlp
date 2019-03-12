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

import os

import mxnet as mx
import numpy as np
import pytest
from numpy.testing import assert_allclose

import gluonnlp as nlp


@pytest.fixture(params=[True, False])
def sparse_grad(request):
    return request.param


def test_csr_embedding(sparse_grad, hybridize):
    token_to_idx = dict(hello=0, world=1)
    embedding = nlp.model.train.CSREmbeddingModel(token_to_idx, 30,
                                                  sparse_grad=sparse_grad)
    embedding.initialize()
    if hybridize:
        embedding.hybridize()

    one_word_per_row = mx.nd.sparse.csr_matrix(
        ([1.0, 1.0],
         ([0, 1], [0, 1])), shape=(2, len(token_to_idx)), dtype=np.float32)
    two_words_per_row = mx.nd.sparse.csr_matrix(
        ([1.0, 1.0],
         ([0, 0], [0, 1])), shape=(1, len(token_to_idx)), dtype=np.float32)
    emb = embedding(one_word_per_row)
    emb2 = embedding(two_words_per_row)
    assert_allclose(emb.sum(axis=0, keepdims=True).asnumpy(), emb2.asnumpy())
    assert_allclose(emb.asnumpy(), embedding[["hello", "world"]].asnumpy())
    assert_allclose(emb[0].asnumpy(), embedding["hello"].asnumpy())


def test_fasttext_embedding(sparse_grad, hybridize):
    token_to_idx = dict(hello=0, world=1)
    num_subwords = 100
    subwords = nlp.vocab.create_subword_function('NGramHashes', ngrams=[
        3, 4, 5, 6], num_subwords=num_subwords)
    embedding = nlp.model.train.FasttextEmbeddingModel(
        token_to_idx, subwords, 30, sparse_grad=sparse_grad)
    embedding.initialize()
    if hybridize:
        embedding.hybridize()

    words = mx.nd.arange(2).reshape((1, -1))
    subwords = words.reshape((1, -1, 1))

    word_and_subwords = mx.nd.sparse.csr_matrix(
        ([0.5, 0.5], ([0, 0], [0, 100])),
        shape=(1, len(token_to_idx) + num_subwords), dtype=np.float32)
    emb = embedding(word_and_subwords)
    emb2 = embedding.weight.data()[word_and_subwords.indices].mean(
        axis=0, keepdims=True)
    assert_allclose(emb.asnumpy(), emb2.asnumpy())


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


def test_word2vec_embedding_load_binary_format():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    word2vec_vec = nlp.embedding.Word2Vec.from_file(
        os.path.join(str(test_dir), 'test_embedding', 'lorem_ipsum_w2v.vec'),
        elem_delim=' '
    )
    word2vec_bin = nlp.embedding.Word2Vec.from_w2v_binary(
        os.path.join(str(test_dir), 'test_embedding', 'lorem_ipsum_w2v.bin')
    )
    idx_to_vec = word2vec_bin[word2vec_vec.idx_to_token]
    assert np.all(
        np.isclose(a=word2vec_vec.idx_to_vec.asnumpy(),
                   b=idx_to_vec.asnumpy(), atol=0.001))
    assert all(token in word2vec_bin for token in word2vec_vec.idx_to_token)
