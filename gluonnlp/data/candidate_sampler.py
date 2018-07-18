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

__all__ = ['CandidateSampler', 'UnigramCandidateSampler']

import mxnet as mx


class CandidateSampler(object):
    """Abstract Candidate Sampler

    After initializing one of the concrete candidate sample implementations,
    generate samples by calling the resulting object.

    """

    def __call__(self):
        raise NotImplementedError


class UnigramCandidateSampler(CandidateSampler):
    """Unigram Candidate Sampler

    Draw random samples from a unigram distribution with specified weights
    using the alias method.

    Parameters
    ----------
    weights : mx.nd.NDArray
        Unnormalized class probabilities. Samples are drawn and returned on the
        same context as weights.context.

    """

    def __init__(self, weights):
        self._context = weights.context
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
        self.prob = mx.nd.array(self.prob, ctx=self._context)
        self.alias = mx.nd.array(self.alias, ctx=self._context)

    def __call__(self, shape):
        """Draw samples from uniform distribution and return sampled candidates.

        Parameters
        ----------
        shape: int or list/tuple of int
            Shape of samples to return.

        Returns
        -------
        samples: NDArray
            The sampled candidate classes.
        """
        idx = mx.nd.random.uniform(low=0, high=self.N, shape=shape,
                                   ctx=self._context,
                                   dtype='float64').floor().astype('float32')
        prob = self.prob[idx]
        alias = self.alias[idx]
        where = mx.nd.random.uniform(shape=shape, ctx=self._context) < prob
        hit = idx * where
        alt = alias * (1 - where)
        candidates = hit + alt

        return candidates
