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
"""Attention cells."""

__all__ = ['PositionalEmbeddingMultiHeadAttentionCell']

import math
import numpy as np

import mxnet as mx

def _masked_softmax(F, att_score, mask, dtype):
    """Ignore the masked elements when calculating the softmax

    Parameters
    ----------
    F : symbol or ndarray
    att_score : Symborl or NDArray
        Shape (batch_size, query_length, memory_length)
    mask : Symbol or NDArray or None
        Shape (batch_size, query_length, memory_length)
    Returns
    -------
    att_weights : Symborl or NDArray
        Shape (batch_size, query_length, memory_length)
    """
    if mask is not None:
        # Fill in the masked scores with a very small value
        neg = -1e18
        if np.dtype(dtype) == np.float16:
            neg = -1e4
        else:
            try:
                # if AMP (automatic mixed precision) is enabled, -1e18 will cause NaN.
                from mxnet.contrib import amp
                if amp.amp._amp_initialized:
                    neg = -1e4
            except ImportError:
                pass
        att_score = F.where(mask, att_score, neg * F.ones_like(att_score))
        att_weights = F.softmax(att_score, axis=-1) * mask
    else:
        att_weights = F.softmax(att_score, axis=-1)
    return att_weights

class PositionalEmbeddingMultiHeadAttentionCell(mx.gluon.HybridBlock):
    """Multi-head Attention Cell with positional embeddings.

    Parameters
    ----------
    d_head
        Number of projected units for respectively query, key, value and
        positional embeddings per attention head.
    num_heads
        Number of parallel attention heads
    dropout
    scaled
    weight_initializer : str or `Initializer` or None, default None
        Initializer of the weights.
    bias_initializer : str or `Initializer`, default 'zeros'
        Initializer of the bias.
    """

    def __init__(self, d_head: int, num_heads: int, dropout: float, scaled: bool,
                 weight_initializer=None, bias_initializer='zeros', dtype='float32', prefix=None,
                 params=None):
        super().__init__(prefix=prefix, params=params)
        self._d_head = d_head
        self._num_heads = num_heads
        self._dropout = dropout
        self._scaled = scaled
        self._dtype = dtype
        units = ['query', 'key', 'value', 'emb']
        with self.name_scope():
            for name in units:
                setattr(
                    self, 'proj_{}'.format(name),
                    mx.gluon.nn.Dense(units=d_head * num_heads, use_bias=False, flatten=False,
                                      weight_initializer=weight_initializer,
                                      bias_initializer=bias_initializer, prefix='{}_'.format(name)))
            self.query_key_bias = self.params.get('query_key_bias', shape=(num_heads, d_head),
                                                  init=bias_initializer)
            self.query_emb_bias = self.params.get('query_emb_bias', shape=(num_heads, d_head),
                                                  init=bias_initializer)
            if dropout:
                self._dropout_layer = mx.gluon.nn.Dropout(dropout)

    def hybrid_forward(self, F, query, key, value, emb, mask, query_key_bias, query_emb_bias):  # pylint: disable=arguments-differ
        """Compute the attention.

        Parameters
        ----------
        query : Symbol or NDArray
            Query vector. Shape (batch_size, query_length, query_dim)
        key : Symbol or NDArray
            Key of the memory. Shape (batch_size, memory_length, key_dim)
        value : Symbol or NDArray
            Value of the memory. Shape (batch_size, memory_length, value_dim)
        emb : Symbol or NDArray
            Positional embeddings. Shape (memory_length, value_dim)
        mask : Symbol or NDArray
            Mask of the memory slots. Shape (batch_size, query_length, memory_length)
            Only contains 0 or 1 where 0 means that the memory slot will not be used.
            If set to None. No mask will be used.

        Returns
        -------
        context_vec : Symbol or NDArray
            Shape (batch_size, query_length, context_vec_dim)
        att_weights : Symbol or NDArray
            Attention weights of multiple heads.
            Shape (batch_size, num_heads, query_length, memory_length)
        """
        att_weights = self._compute_weight(F, query, key, emb, mask, query_key_bias=query_key_bias,
                                           query_emb_bias=query_emb_bias)
        context_vec = self._read_by_weight(F, att_weights, value)
        return context_vec, att_weights

    def _project(self, F, name, x):
        # Shape (batch_size, query_length, num_heads * d_head)
        x = getattr(self, 'proj_{}'.format(name))(x)
        # Shape (batch_size * num_heads, query_length, d_head)
        x = F.transpose(x.reshape(shape=(0, 0, self._num_heads, -1)),
                        axes=(0, 2, 1, 3))\
             .reshape(shape=(-1, 0, 0), reverse=True)
        return x

    @staticmethod
    def _rel_shift(F, x):
        """Perform relative shift operation following Dai et al. (2019) Appendix B

        Unlike Dai et al.'s PyTorch implementation, the relative shift is
        performed on the last two dimensions of the ndarray x.

        Requires len(x.shape) == 3 due to F.swapaxes not supporting negative
        indices

        """
        # Zero pad along last axis
        zero_pad = F.zeros_like(F.slice_axis(x, axis=-1, begin=0, end=1))
        x_padded = F.concat(zero_pad, x, dim=-1)
        # Reshape to x.shape[:-2] + [x.shape[-1] + 1, x.shape[-2]]
        x_padded = F.reshape_like(x_padded, F.swapaxes(x_padded, 1, 2))
        # Remove padded elements
        x_padded = F.slice_axis(x_padded, axis=-2, begin=1, end=None)
        # Reshape back to original shape
        x = F.reshape_like(x_padded, x)
        return x

    def _compute_weight(self, F, query, key, emb, mask, query_key_bias, query_emb_bias):
        # Project query, key and emb
        proj_query = self.proj_query(query).reshape(shape=(0, 0, self._num_heads, -1))
        proj_key = self.proj_key(key).reshape(shape=(0, 0, self._num_heads, -1))
        proj_emb = self.proj_emb(emb).reshape(shape=(-1, self._num_heads, self._d_head))

        # Add biases and transpose to (batch_size, num_heads, query_length,
        # d_head) or (num_heads, query_length, d_head)
        query_with_key_bias = F.transpose(
            F.broadcast_add(proj_query, F.reshape(query_key_bias, shape=(1, 1, 0, 0),
                                                  reverse=True)), axes=(0, 2, 1, 3))
        query_with_emb_bias = F.transpose(
            F.broadcast_add(proj_query, F.reshape(query_emb_bias, shape=(1, 1, 0, 0),
                                                  reverse=True)), axes=(0, 2, 1, 3))
        proj_key = F.transpose(proj_key, axes=(0, 2, 1, 3))
        proj_emb = F.transpose(proj_emb, axes=(1, 0, 2))

        # Broadcast emb along batch axis
        proj_emb = F.broadcast_like(F.expand_dims(proj_emb, axis=0), proj_key)

        # Merge batch and num_heads axes
        query_with_key_bias = query_with_key_bias.reshape(shape=(-1, 0, 0), reverse=True)
        proj_key = proj_key.reshape(shape=(-1, 0, 0), reverse=True)
        query_with_emb_bias = query_with_emb_bias.reshape(shape=(-1, 0, 0), reverse=True)
        proj_emb = proj_emb.reshape(shape=(-1, 0, 0), reverse=True)

        if mask is not None:
            # Insert and broadcast along num_heads axis. Merge num_heads and
            # batch_size axes: (batch_size * num_heads, query_length,
            # memory_length)
            mask = F.broadcast_axis(F.expand_dims(mask, axis=1), axis=1, size=self._num_heads)\
                    .reshape(shape=(-1, 0, 0), reverse=True)

        att_score_AC = F.batch_dot(query_with_key_bias, proj_key, transpose_b=True)
        att_score_BD = F.batch_dot(query_with_emb_bias, proj_emb, transpose_b=True)

        # Relative shift
        shifted_att_score_BD = self._rel_shift(F, att_score_BD)

        att_score = att_score_AC + shifted_att_score_BD
        if self._scaled:
            att_score = att_score / math.sqrt(self._d_head)

        att_weights = _masked_softmax(F, att_score, mask, self._dtype)
        if self._dropout:
            att_weights = self._dropout_layer(att_weights)

        return att_weights.reshape(shape=(-1, self._num_heads, 0, 0), reverse=True)

    def _read_by_weight(self, F, att_weights, value):
        att_weights = att_weights.reshape(shape=(-1, 0, 0), reverse=True)
        proj_value = self._project(F, 'value', value)
        context_vec = F.batch_dot(att_weights, proj_value)
        context_vec = F.transpose(
            context_vec.reshape(shape=(-1, self._num_heads, 0, 0), reverse=True),
            axes=(0, 2, 1, 3)).reshape(shape=(0, 0, -1))
        return context_vec
