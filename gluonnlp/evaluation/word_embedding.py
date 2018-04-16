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

# pylint: disable=
"""Evaluation helpers for word embeddings."""

import attr
import mxnet as mx
import numpy as np
from scipy import stats

__all__ = ['WordEmbeddingSimilarityEvaluator']


@attr.s()
class WordEmbeddingEvaluator(object):
    """Helper class to evaluate word embeddings."""
    dataset = attr.ib()
    vocabulary = attr.ib()


@attr.s()
class WordEmbeddingSimilarityEvaluator(WordEmbeddingEvaluator):
    correlation_coefficient = attr.ib(
        default='spearmanr',
        validator=attr.validators.in_(['spearmanr', 'pearsonr']))

    # Words and ground truth scores
    _w1s = None
    _w2s = None
    _scores = None
    _context = None

    def __attrs_post_init__(self):
        # Construct nd arrays from dataset
        w1s = []
        w2s = []
        scores = []
        for word1, word2, score in self.dataset:
            if (word1 in self.vocabulary and word2 in self.vocabulary):
                w1s.append(word1)
                w2s.append(word2)
                scores.append(score)

        print(('Using {num_use} of {num_total} word pairs '
               'from {ds} for evaluation.').format(
                   num_use=len(w1s),
                   num_total=len(self.dataset),
                   ds=self.dataset.__class__.__name__))

        self._w1s = w1s
        self._w2s = w2s
        self._scores = np.array(scores)

    def __len__(self):
        return len(self._w1s)

    def __call__(self, token_embedding):
        if not len(self):
            return 0

        w1s_embedding = mx.nd.L2Normalization(token_embedding[self._w1s])
        w2s_embedding = mx.nd.L2Normalization(token_embedding[self._w2s])

        batch_size, embedding_size = w1s_embedding.shape

        cosine_similarity = mx.nd.batch_dot(
            w1s_embedding.reshape((batch_size, 1, embedding_size)),
            w2s_embedding.reshape((batch_size, embedding_size, 1)))
        cosine_similarity_np = cosine_similarity.asnumpy().flatten()

        if self.correlation_coefficient == 'spearmanr':
            r = stats.spearmanr(cosine_similarity_np, self._scores).correlation
        elif self.correlation_coefficient == 'pearsonr':
            r = stats.pearsonr(cosine_similarity_np, self._scores).correlation
        else:
            raise ValueError('Invalid correlation_coefficient: {}'.format(
                self.correlation_coefficient))

        return r
