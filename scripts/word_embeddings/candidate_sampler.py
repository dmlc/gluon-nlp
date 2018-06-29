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
"""Candidate samplers"""

__all__ = ['UnigramCandidateSampler']

import mxnet as mx
import numpy as np

import gluonnlp as nlp

try:
    from numba import njit
    numba_njit = njit(nogil=True)
except ImportError:

    def numba_njit(func):
        return func


@numba_njit
def _candidates_mask(negatives, true_samples, true_samples_mask):
    assert len(negatives.shape) == 2
    assert len(true_samples.shape) == 2
    assert negatives.shape[0] == true_samples.shape[0]

    negatives_mask = np.ones(negatives.shape)

    for i in range(negatives.shape[0]):
        for j in range(negatives.shape[1]):
            for z in range(true_samples.shape[1]):
                if (true_samples_mask[i, z]
                        and negatives[i, j] == true_samples[i, z]):
                    negatives_mask[i, j] = 0
    return negatives_mask


class UnigramCandidateSampler(nlp.data.CandidateSampler):
    """Unigram Candidate Sampler

    Parameters
    ----------
    weights : mx.nd.NDArray
        Unnormalized class probabilities.

    """

    def __init__(self, weights):
        self.N = weights.size
        total_weights = weights.sum()
        self.prob = (weights * self.N / total_weights).asnumpy().tolist()
        self.alias = [0] * self.N

        # sort the data into the outcomes with probabilities
        # that are high and low than 1/N.
        low = []
        high = []
        for i in range(self.N):
            if self.prob[i] < 1.0:
                low.append(i)
            else:
                high.append(i)

        # pair low with high
        while len(low) > 0 and len(high) > 0:
            l = low.pop()
            h = high.pop()

            self.alias[l] = h
            self.prob[h] = self.prob[h] - (1.0 - self.prob[l])

            if self.prob[h] < 1.0:
                low.append(h)
            else:
                high.append(h)

        for i in low + high:
            self.prob[i] = 1
            self.alias[i] = i

        # convert to ndarrays
        self.prob = mx.nd.array(self.prob)
        self.alias = mx.nd.array(self.alias)

    def __call__(self, shape, true_samples=None, true_samples_mask=None):
        """Draw shape samples from the distribution.

        If true_samples is specified, also returns a mask that masks any random
        elements that are also part of the same row as true_samples.

        """

        # Draw samples
        idx = mx.nd.array(np.random.randint(0, self.N, size=shape))
        prob = self.prob[idx]
        alias = self.alias[idx]
        where = mx.nd.random.uniform(shape=shape) < prob
        hit = idx * where
        alt = alias * (1 - where)
        candidates = hit + alt

        # Remove accidental hits
        if true_samples is not None:
            candidates_np = candidates.asnumpy()
            true_samples_np = true_samples.asnumpy()
            if true_samples_mask is not None:
                true_samples_mask_np = true_samples_mask.asnumpy()
            else:
                true_samples_mask_np = np.ones_like(true_samples_np)

            candidates_mask = mx.nd.array(
                _candidates_mask(candidates_np, true_samples_np,
                                 true_samples_mask_np))

            return candidates, candidates_mask
        else:
            return candidates
