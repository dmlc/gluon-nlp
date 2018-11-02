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

import functools
import operator

import mxnet as mx
import numpy as np


class UnigramCandidateSampler(mx.gluon.HybridBlock):
    """Unigram Candidate Sampler

    Draw random samples from a unigram distribution with specified weights
    using the alias method.

    Parameters
    ----------
    weights : mx.nd.NDArray
        Unnormalized class probabilities. Samples are drawn and returned on the
        same context as weights.context.
    shape : int or tuple of int
        Shape of data to be sampled.
        TODO: Specifying the shape is only a workaround until random_like
        operators are available in mxnet
    dtype : str or np.dtype, default 'float32'
        Data type of the candidates. Make sure that the dtype precision is
        large enough to represent the size of your weights array precisely. For
        example, float32 can not distinguish 2**24 from 2**24 + 1.

    """

    def __init__(self, weights, shape, dtype='float32'):
        super(UnigramCandidateSampler, self).__init__()
        self._shape = shape
        self._dtype = dtype
        self.N = weights.size

        if (np.dtype(dtype) == np.float32 and weights.size > 2**24) or \
           (np.dtype(dtype) == np.float16 and weights.size > 2**11):
            s = 'dtype={dtype} can not represent all weights'
            raise ValueError(s.format(dtype=dtype))

        total_weights = weights.sum()
        prob = (weights * self.N / total_weights).asnumpy().tolist()
        alias = [0] * self.N

        # sort the data into the outcomes with probabilities
        # that are high and low than 1/N.
        low = []
        high = []
        for i in range(self.N):
            if prob[i] < 1.0:
                low.append(i)
            else:
                high.append(i)

        # pair low with high
        while len(low) > 0 and len(high) > 0:
            l = low.pop()
            h = high.pop()

            alias[l] = h
            prob[h] = prob[h] - (1.0 - prob[l])

            if prob[h] < 1.0:
                low.append(h)
            else:
                high.append(h)

        for i in low + high:
            prob[i] = 1
            alias[i] = i

        # store
        prob = mx.nd.array(prob, dtype='float64')
        alias = mx.nd.array(alias, dtype='float64')
        self.prob = self.params.get_constant('prob', prob)
        self.alias = self.params.get_constant('alias', alias)

    def __repr__(self):
        s = '{block_name}({len_weights}, {dtype})'
        return s.format(block_name=self.__class__.__name__, len_weights=self.N,
                        dtype=self._dtype)

    def hybrid_forward(self, F, candidates_like, prob, alias):
        # pylint: disable=unused-argument
        """Draw samples from uniform distribution and return sampled candidates.

        Parameters
        ----------
        candidates_like: mxnet.nd.NDArray or mxnet.sym.Symbol
            This input specifies the shape of the to be sampled candidates. #
            TODO shape selection is not yet supported. Shape must be specified
            in the constructor.

        Returns
        -------
        samples: mxnet.nd.NDArray or mxnet.sym.Symbol
            The sampled candidates of shape candidates_like.shape. Candidates
            are sampled based on the weights specified on creation of the
            UnigramCandidateSampler.
        """
        flat_shape = functools.reduce(operator.mul, self._shape)
        idx = F.random.uniform(low=0, high=self.N, shape=flat_shape,
                               dtype='float64').floor()
        prob = F.gather_nd(prob, idx.reshape((1, -1)))
        alias = F.gather_nd(alias, idx.reshape((1, -1)))
        where = F.random.uniform(shape=flat_shape,
                                 dtype='float64') < prob
        hit = idx * where
        alt = alias * (1 - where)
        candidates = (hit + alt).reshape(self._shape)

        return candidates.astype(self._dtype)
