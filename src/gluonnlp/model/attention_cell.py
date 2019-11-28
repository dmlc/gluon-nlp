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

__all__ = ['AttentionCell', 'MultiHeadAttentionCell', 'MLPAttentionCell', 'DotProductAttentionCell']

import math
import numpy as np
import mxnet as mx
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.contrib.amp import amp
from .block import L2Normalization


def _apply_mask(F, att_score, mask, dtype):
    """Fill in the masked scores with a very small value

    Parameters
    ----------
    F : symbol or ndarray
    att_score : Symbol or NDArray
        Shape (batch_size, query_length, memory_length)
    mask : Symbol or NDArray or None
        Shape (batch_size, query_length, memory_length)
    Returns
    -------
    att_score : Symbol or NDArray
        Shape (batch_size, query_length, memory_length)
    """
    # Fill in the masked scores with a very small value
    neg = -1e18
    if np.dtype(dtype) == np.float16:
        neg = -1e4
    else:
        # if AMP (automatic mixed precision) is enabled, -1e18 will cause NaN.
        if amp._amp_initialized:
            neg = -1e4
    att_score = F.where(mask, att_score, neg * F.ones_like(att_score))
    return att_score


# TODO(sxjscience) Add mask flag to softmax operator. Think about how to accelerate the kernel
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
    att_weights : Symbol or NDArray
        Shape (batch_size, query_length, memory_length)
    """
    if mask is not None:
        # Fill in the masked scores with a very small value
        att_score = _apply_mask(F, att_score, mask, dtype)
        att_weights = F.softmax(att_score, axis=-1) * mask
    else:
        att_weights = F.softmax(att_score, axis=-1)
    return att_weights


# TODO(sxjscience) In the future, we should support setting mask/att_weights as sparse tensors
class AttentionCell(HybridBlock):
    """Abstract class for attention cells. Extend the class
     to implement your own attention method.
     One typical usage is to define your own `_compute_weight()` function to calculate the weights::

        cell = AttentionCell()
        out = cell(query, key, value, mask)

    """
    def __init__(self, prefix=None, params=None):
        self._dtype = np.float32
        super(AttentionCell, self).__init__(prefix=prefix, params=params)

    def cast(self, dtype):
        self._dtype = dtype
        super(AttentionCell, self).cast(dtype)

    def _compute_weight(self, F, query, key, mask=None):
        """Compute attention weights based on the query and the keys

        Parameters
        ----------
        F : symbol or ndarray
        query : Symbol or NDArray
            The query vectors. Shape (batch_size, query_length, query_dim)
        key : Symbol or NDArray
            Key of the memory. Shape (batch_size, memory_length, key_dim)
        mask : Symbol or NDArray or None
            Mask the memory slots. Shape (batch_size, query_length, memory_length)
            Only contains 0 or 1 where 0 means that the memory slot will not be used.
            If set to None. No mask will be used.

        Returns
        -------
        att_weights : Symbol or NDArray
            For single-head attention, Shape (batch_size, query_length, memory_length)
            For multi-head attention, Shape (batch_size, num_heads, query_length, memory_length)
        """
        raise NotImplementedError

    def _read_by_weight(self, F, att_weights, value):
        """Read from the value matrix given the attention weights.

        Parameters
        ----------
        F : symbol or ndarray
        att_weights : Symbol or NDArray
            Attention weights.
            For single-head attention,
                Shape (batch_size, query_length, memory_length).
            For multi-head attention,
                Shape (batch_size, num_heads, query_length, memory_length).
        value : Symbol or NDArray
            Value of the memory. Shape (batch_size, memory_length, total_value_dim)

        Returns
        -------
        context_vec: Symbol or NDArray
            Shape (batch_size, query_length, context_vec_dim)
        """
        output = F.batch_dot(att_weights, value)
        return output

    def __call__(self, query, key, value=None, mask=None):  # pylint: disable=arguments-differ
        """Compute the attention.

        Parameters
        ----------
        query : Symbol or NDArray
            Query vector. Shape (batch_size, query_length, query_dim)
        key : Symbol or NDArray
            Key of the memory. Shape (batch_size, memory_length, key_dim)
        value : Symbol or NDArray or None, default None
            Value of the memory. If set to None, the value will be set as the key.
            Shape (batch_size, memory_length, value_dim)
        mask : Symbol or NDArray or None, default None
            Mask of the memory slots. Shape (batch_size, query_length, memory_length)
            Only contains 0 or 1 where 0 means that the memory slot will not be used.
            If set to None. No mask will be used.

        Returns
        -------
        context_vec : Symbol or NDArray
            Shape (batch_size, query_length, context_vec_dim)
        att_weights : Symbol or NDArray
            Attention weights. Shape (batch_size, query_length, memory_length)
        """
        return super(AttentionCell, self).__call__(query, key, value, mask)

    def hybrid_forward(self, F, query, key, value=None, mask=None):  # pylint: disable=arguments-differ
        if value is None:
            value = key
        att_weights = self._compute_weight(F, query, key, mask)
        context_vec = self._read_by_weight(F, att_weights, value)
        return context_vec, att_weights


class MultiHeadAttentionCell(AttentionCell):
    r"""Multi-head Attention Cell.

    In the MultiHeadAttentionCell, the input query/key/value will be linearly projected
    for `num_heads` times with different projection matrices. Each projected key, value, query
    will be used to calculate the attention weights and values. The output of each head will be
    concatenated to form the final output.

    The idea is first proposed in "[Arxiv2014] Neural Turing Machines" and
    is later adopted in "[NIPS2017] Attention is All You Need" to solve the
    Neural Machine Translation problem.

    Parameters
    ----------
    base_cell : AttentionCell
    query_units : int
        Total number of projected units for query. Must be divided exactly by num_heads.
    key_units : int
        Total number of projected units for key. Must be divided exactly by num_heads.
    value_units : int
        Total number of projected units for value. Must be divided exactly by num_heads.
    num_heads : int
        Number of parallel attention heads
    use_bias : bool, default True
        Whether to use bias when projecting the query/key/values
    weight_initializer : str or `Initializer` or None, default None
        Initializer of the weights.
    bias_initializer : str or `Initializer`, default 'zeros'
        Initializer of the bias.
    prefix : str or None, default None
        See document of `Block`.
    params : str or None, default None
        See document of `Block`.
    """
    def __init__(self, base_cell, query_units, key_units, value_units, num_heads, use_bias=True,
                 weight_initializer=None, bias_initializer='zeros', prefix=None, params=None):
        super(MultiHeadAttentionCell, self).__init__(prefix=prefix, params=params)
        self._base_cell = base_cell
        self._num_heads = num_heads
        self._use_bias = use_bias
        units = {'query': query_units, 'key': key_units, 'value': value_units}
        for name, unit in units.items():
            if unit % self._num_heads != 0:
                raise ValueError(
                    'In MultiHeadAttetion, the {name}_units should be divided exactly'
                    ' by the number of heads. Received {name}_units={unit}, num_heads={n}'.format(
                        name=name, unit=unit, n=num_heads))
            setattr(self, '_{}_units'.format(name), unit)
            with self.name_scope():
                setattr(
                    self, 'proj_{}'.format(name),
                    nn.Dense(units=unit, use_bias=self._use_bias, flatten=False,
                             weight_initializer=weight_initializer,
                             bias_initializer=bias_initializer, prefix='{}_'.format(name)))

    def __call__(self, query, key, value=None, mask=None):
        """Compute the attention.

        Parameters
        ----------
        query : Symbol or NDArray
            Query vector. Shape (batch_size, query_length, query_dim)
        key : Symbol or NDArray
            Key of the memory. Shape (batch_size, memory_length, key_dim)
        value : Symbol or NDArray or None, default None
            Value of the memory. If set to None, the value will be set as the key.
            Shape (batch_size, memory_length, value_dim)
        mask : Symbol or NDArray or None, default None
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
        return super(MultiHeadAttentionCell, self).__call__(query, key, value, mask)

    def _project(self, F, name, x):
        # Shape (batch_size, query_length, query_units)
        x = getattr(self, 'proj_{}'.format(name))(x)
        # Shape (batch_size * num_heads, query_length, ele_units)
        x = F.transpose(x.reshape(shape=(0, 0, self._num_heads, -1)),
                        axes=(0, 2, 1, 3))\
             .reshape(shape=(-1, 0, 0), reverse=True)
        return x

    def _compute_weight(self, F, query, key, mask=None):
        query = self._project(F, 'query', query)
        key = self._project(F, 'key', key)
        if mask is not None:
            mask = F.broadcast_axis(F.expand_dims(mask, axis=1),
                                    axis=1, size=self._num_heads)\
                    .reshape(shape=(-1, 0, 0), reverse=True)
        att_weights = self._base_cell._compute_weight(F, query, key, mask)
        return att_weights.reshape(shape=(-1, self._num_heads, 0, 0), reverse=True)

    def _read_by_weight(self, F, att_weights, value):
        att_weights = att_weights.reshape(shape=(-1, 0, 0), reverse=True)
        value = self._project(F, 'value', value)
        context_vec = self._base_cell._read_by_weight(F, att_weights, value)
        context_vec = F.transpose(context_vec.reshape(shape=(-1, self._num_heads, 0, 0),
                                                      reverse=True),
                                  axes=(0, 2, 1, 3)).reshape(shape=(0, 0, -1))
        return context_vec


class MLPAttentionCell(AttentionCell):
    r"""Concat the query and the key and use a single-hidden-layer MLP to get the attention score.
    We provide two mode, the standard mode and the normalized mode.

    In the standard mode::

        score = v tanh(W [h_q, h_k] + b)

    In the normalized mode (Same as TensorFlow)::

        score = g v / ||v||_2 tanh(W [h_q, h_k] + b)

    This type of attention is first proposed in

    .. Bahdanau et al., Neural Machine Translation by Jointly Learning to Align and Translate.
       ICLR 2015

    Parameters
    ----------
    units : int
    act : Activation, default nn.Activation('tanh')
    normalized : bool, default False
        Whether to normalize the weight that maps the embedded
        hidden states to the final score. This strategy can be interpreted as a type of
        "[NIPS2016] Weight Normalization".
    dropout : float, default 0.0
        Attention dropout.
    weight_initializer : str or `Initializer` or None, default None
        Initializer of the weights.
    bias_initializer : str or `Initializer`, default 'zeros'
        Initializer of the bias.
    prefix : str or None, default None
        See document of `Block`.
    params : ParameterDict or None, default None
        See document of `Block`.
    """

    def __init__(self, units, act=nn.Activation('tanh'), normalized=False, dropout=0.0,
                 weight_initializer=None, bias_initializer='zeros', prefix=None, params=None):
        # Define a temporary class to implement the normalized version
        # TODO(sxjscience) Find a better solution
        class _NormalizedScoreProj(HybridBlock):
            def __init__(self, in_units, weight_initializer=None, prefix=None, params=None):
                super(_NormalizedScoreProj, self).__init__(prefix=prefix, params=params)
                self.g = self.params.get('g', shape=(1,),
                                         init=mx.init.Constant(1.0 / math.sqrt(in_units)),
                                         allow_deferred_init=True)
                self.v = self.params.get('v', shape=(1, in_units),
                                         init=weight_initializer,
                                         allow_deferred_init=True)

            def hybrid_forward(self, F, x, g, v):  # pylint: disable=arguments-differ
                v = F.broadcast_div(v, F.sqrt(F.dot(v, v, transpose_b=True)))
                weight = F.broadcast_mul(g, v)
                out = F.FullyConnected(x, weight, None, no_bias=True, num_hidden=1,
                                       flatten=False, name='fwd')
                return out

        super(MLPAttentionCell, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._act = act
        self._normalized = normalized
        self._dropout = dropout
        with self.name_scope():
            self._dropout_layer = nn.Dropout(dropout)
            self._query_mid_layer = nn.Dense(units=self._units, flatten=False, use_bias=True,
                                             weight_initializer=weight_initializer,
                                             bias_initializer=bias_initializer,
                                             prefix='query_')
            self._key_mid_layer = nn.Dense(units=self._units, flatten=False, use_bias=False,
                                           weight_initializer=weight_initializer,
                                           prefix='key_')
            if self._normalized:
                self._attention_score = \
                    _NormalizedScoreProj(in_units=units,
                                         weight_initializer=weight_initializer,
                                         prefix='score_')
            else:
                self._attention_score = nn.Dense(units=1, in_units=self._units,
                                                 flatten=False, use_bias=False,
                                                 weight_initializer=weight_initializer,
                                                 prefix='score_')

    def _compute_score(self, F, query, key, mask=None):
        mapped_query = self._query_mid_layer(query)
        mapped_key = self._key_mid_layer(key)
        mid_feat = F.broadcast_add(F.expand_dims(mapped_query, axis=2),
                                   F.expand_dims(mapped_key, axis=1))
        mid_feat = self._act(mid_feat)
        att_score = self._attention_score(mid_feat).reshape(shape=(0, 0, 0))
        if mask is not None:
            att_score = _apply_mask(F, att_score, mask, self._dtype)
        return att_score

    def _compute_weight(self, F, query, key, mask=None):
        att_score = self._compute_score(F, query, key, mask)
        att_weights = F.softmax(att_score, axis=-1)
        if mask is not None:
            att_weights = att_weights * mask
        att_weights = self._dropout_layer(att_weights)
        return att_weights


class DotProductAttentionCell(AttentionCell):
    r"""Dot product attention between the query and the key.

    Depending on parameters, defined as::

        units is None:
            score = <h_q, h_k>
        units is not None and luong_style is False:
            score = <W_q h_q, W_k h_k>
        units is not None and luong_style is True:
            score = <W h_q, h_k>

    Parameters
    ----------
    units: int or None, default None
        Project the query and key to vectors with `units` dimension
        before applying the attention. If set to None,
        the query vector and the key vector are directly used to compute the attention and
        should have the same dimension::

            If the units is None,
                score = <h_q, h_k>
            Else if the units is not None and luong_style is False:
                score = <W_q h_q, W_k h_k>
            Else if the units is not None and luong_style is True:
                score = <W h_q, h_k>

    luong_style: bool, default False
        If turned on, the score will be::

            score = <W h_q, h_k>

        `units` must be the same as the dimension of the key vector
    scaled: bool, default True
        Whether to divide the attention weights by the sqrt of the query dimension.
        This is first proposed in "[NIPS2017] Attention is all you need."::

            score = <h_q, h_k> / sqrt(dim_q)

    normalized: bool, default False
        If turned on, the cosine distance is used, i.e::

            score = <h_q / ||h_q||, h_k / ||h_k||>

    use_bias : bool, default True
        Whether to use bias in the projection layers.
    dropout : float, default 0.0
        Attention dropout
    weight_initializer : str or `Initializer` or None, default None
        Initializer of the weights
    bias_initializer : str or `Initializer`, default 'zeros'
        Initializer of the bias
    prefix : str or None, default None
        See document of `Block`.
    params : str or None, default None
        See document of `Block`.
    """
    def __init__(self, units=None, luong_style=False, scaled=True, normalized=False, use_bias=True,
                 dropout=0.0, weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(DotProductAttentionCell, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._scaled = scaled
        self._normalized = normalized
        self._use_bias = use_bias
        self._luong_style = luong_style
        self._dropout = dropout
        if self._luong_style:
            assert units is not None, 'Luong style attention is not available without explicitly ' \
                                      'setting the units'
        with self.name_scope():
            self._dropout_layer = nn.Dropout(dropout)
        if units is not None:
            with self.name_scope():
                self._proj_query = nn.Dense(units=self._units, use_bias=self._use_bias,
                                            flatten=False, weight_initializer=weight_initializer,
                                            bias_initializer=bias_initializer, prefix='query_')
                if not self._luong_style:
                    self._proj_key = nn.Dense(units=self._units, use_bias=self._use_bias,
                                              flatten=False, weight_initializer=weight_initializer,
                                              bias_initializer=bias_initializer, prefix='key_')
        if self._normalized:
            with self.name_scope():
                self._l2_norm = L2Normalization(axis=-1)

    def _compute_score(self, F, query, key, mask=None):
        if self._units is not None:
            query = self._proj_query(query)
            if not self._luong_style:
                key = self._proj_key(key)
            elif F == mx.nd:
                assert query.shape[-1] == key.shape[-1], 'Luong style attention requires key to ' \
                                                         'have the same dim as the projected ' \
                                                         'query. Received key {}, query {}.'.format(
                                                             key.shape, query.shape)
        if self._normalized:
            query = self._l2_norm(query)
            key = self._l2_norm(key)
        if self._scaled:
            query = F.contrib.div_sqrt_dim(query)

        att_score = F.batch_dot(query, key, transpose_b=True)
        if mask is not None:
            att_score = _apply_mask(F, att_score, mask, self._dtype)
        return att_score

    def _compute_weight(self, F, query, key, mask=None):
        att_score = self._compute_score(F, query, key, mask)
        att_weights = F.softmax(att_score, axis=-1)
        if mask is not None:
            att_weights = att_weights * mask
        att_weights = self._dropout_layer(att_weights)
        return att_weights
