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
"""Attention cells."""

__all__ = ['AdaptiveEmbedding', 'ProjectedEmbedding']

from typing import List

import mxnet as mx


class ProjectedEmbedding(mx.gluon.HybridBlock):
    """Projected Embedding"""

    def __init__(self, vocab_size: int, embed_size: int, units: int, project_same_dim: bool = True,
                 embedding_initializer=None, projection_initializer=None, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self._vocab_size = vocab_size
        self._embed_size = embed_size
        self._units = units
        self._project_same_dim = project_same_dim
        self._emb_scale = units**0.5

        with self.name_scope():
            self.embedding_weight = self.params.get('embedding_weight',
                                                    shape=(vocab_size, embed_size),
                                                    init=embedding_initializer)
            if units != embed_size or project_same_dim:
                self.projection_weight = self.params.get('projection_weight',
                                                         shape=(units, embed_size),
                                                         init=projection_initializer)

    def hybrid_forward(self, F, inp, **params):  # pylint: disable=arguments-differ
        emb = F.Embedding(data=inp, weight=params['embedding_weight'], input_dim=self._vocab_size,
                          output_dim=self._embed_size)
        if self._units != self._embed_size or self._project_same_dim:
            emb = F.FullyConnected(data=emb, weight=params['projection_weight'], no_bias=True,
                                   flatten=False, num_hidden=self._units)
        return emb * self._emb_scale


class AdaptiveEmbedding(mx.gluon.HybridBlock):
    """Adaptive Embedding

    Baevski, A., & Auli, M. (2019). Adaptive input representations for neural
    language modeling. In International Conference on Learning Representations.

    """

    # TODO: Transformer-XL has a sample_softmax argument here

    def __init__(self, vocab_size: int, embed_size: int, units: int, cutoffs: List[int],
                 div_val: int = 1, project_same_dim: bool = True, embedding_initializer=None,
                 projection_initializer=None, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        # Sanity checks
        if cutoffs != sorted(cutoffs):
            raise ValueError('cutoffs must be a sorted list of cutoff values. '
                             'Got {}, but expected {}'.format(cutoffs, sorted(cutoffs)))
        if not cutoffs:
            raise ValueError('cutoffs must not be empty. Got {}'.format(cutoffs))
        if cutoffs[0] <= 0:
            raise ValueError('The first cutoff value ({}) must be greater 0.'.format(cutoffs[0]))
        if cutoffs[-1] >= vocab_size:
            raise ValueError(
                'The last cutoff value ({}) must be smaller than vocab_size ({}).'.format(
                    cutoffs[-1], vocab_size))

        self._vocab_size = vocab_size
        self._embed_size = embed_size
        self._cutoffs = [0] + cutoffs + [vocab_size]
        self._div_val = div_val
        self._units = units
        self._project_same_dim = project_same_dim
        self._emb_scale = units**0.5

        with self.name_scope():
            if self._div_val == 1:
                name = 'embedding0_weight'
                setattr(
                    self, name,
                    self.params.get(name, shape=(vocab_size, embed_size),
                                    init=embedding_initializer))

                if units != embed_size or project_same_dim:
                    name = 'projection0_weight'
                    setattr(
                        self, name,
                        self.params.get(name, shape=(units, embed_size),
                                        init=projection_initializer))
            else:
                for i, (l_idx, r_idx) in enumerate(zip(self._cutoffs, self._cutoffs[1:])):
                    name = 'embedding{}_weight'.format(i)
                    setattr(
                        self, name,
                        self.params.get(name, shape=(r_idx - l_idx, embed_size // div_val**i),
                                        init=embedding_initializer))

                    if units != embed_size // div_val**i or project_same_dim:
                        name = 'projection{}_weight'.format(i)
                        setattr(
                            self, name,
                            self.params.get(name, shape=(units, embed_size // div_val**i),
                                            init=projection_initializer))

    def hybrid_forward(self, F, inp, **params):  # pylint: disable=arguments-differ
        if self._div_val == 1:
            emb = F.Embedding(data=inp, weight=params['embedding0_weight'],
                              input_dim=self._vocab_size, output_dim=self._embed_size)
            if self._units != self._embed_size or self._project_same_dim:
                emb = F.FullyConnected(data=emb, weight=params['projection0_weight'], no_bias=True,
                                       flatten=False, num_hidden=self._units)
        else:
            inp_flat = inp.reshape((-1, ))
            zeros_like_inp_flat = F.zeros_like(inp_flat)
            ones_like_inp_flat = F.ones_like(inp_flat)
            emb_flat = None
            for i, (l_idx, r_idx) in enumerate(zip(self._cutoffs, self._cutoffs[1:])):
                cond_i = F.broadcast_logical_and(inp_flat >= l_idx, inp_flat < r_idx)
                inp_i = F.where(cond_i, inp_flat - l_idx, zeros_like_inp_flat)
                mask_i = F.expand_dims(F.where(cond_i, ones_like_inp_flat, zeros_like_inp_flat),
                                       axis=1)

                emb_i = F.Embedding(data=inp_i, weight=params['embedding{}_weight'.format(i)],
                                    input_dim=r_idx - l_idx,
                                    output_dim=self._embed_size // self._div_val**i)
                emb_i = F.broadcast_mul(emb_i, mask_i)
                if self._units != self._embed_size // self._div_val**i or self._project_same_dim:
                    emb_i = F.FullyConnected(data=emb_i,
                                             weight=params['projection{}_weight'.format(i)],
                                             no_bias=True, flatten=False, num_hidden=self._units)

                if emb_flat is None:  # i == 0
                    emb_flat = emb_i
                else:
                    emb_flat = emb_flat + emb_i

            emb = F.reshape_like(emb_flat, inp, lhs_begin=0, lhs_end=1)

        emb = emb * self._emb_scale

        return emb
