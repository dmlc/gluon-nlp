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
import math
import mxnet as mx
from mxnet import np, npx
import numpy as _np
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from .op import l2_normalize
from .layers import SinusoidalPositionalEmbedding,\
                    BucketPositionalEmbedding,\
                    LearnedPositionalEmbedding
from typing import Optional


# TODO(sxjscience)
#  We can optimize the whole function by writing a custom-op,
#  or automatically fuse these operators.
def gen_self_attn_mask(data,
                       valid_length=None,
                       dtype: type = np.float32,
                       attn_type: str = 'full',
                       layout: str = 'NT'):
    """Generate the mask used for the encoder, i.e, self-attention.

    In our implementation, 1 --> not masked, 0 --> masked

    Let's consider the data with two samples:

    .. code-block:: none

        data =
            [['I',   'can', 'now',   'use', 'numpy', 'in',  'Gluon@@', 'NLP'  ],
             ['May', 'the', 'force', 'be',  'with',  'you', '<PAD>',   '<PAD>']]
        valid_length =
            [8, 6]

    - attn_type = 'causal'
        Each token will attend to itself + the tokens before.
        It will not attend to tokens in the future.

        For our example, the mask of the first sample is

        .. code-block:: none

                       ['I', 'can', 'now', 'use', 'numpy', 'in', 'Gluon@@', 'NLP']
            'I':         1,    0,     0,     0,      0,     0,      0,      0
            'can':       1,    1,     0,     0,      0,     0,      0,      0
            'now':       1,    1,     1,     0,      0,     0,      0,      0
            'use':       1,    1,     1,     1,      0,     0,      0,      0
            'numpy':     1,    1,     1,     1,      1,     0,      0,      0
            'in':        1,    1,     1,     1,      1,     1,      0,      0
            'Gluon@@':   1,    1,     1,     1,      1,     1,      1,      0
            'NLP':       1,    1,     1,     1,      1,     1,      1,      1

        The mask of the second sample is

        .. code-block:: none

                       ['May', 'the', 'force', 'be', 'with', 'you', '<PAD>', '<PAD>']
            'May':        1,    0,     0,     0,      0,     0,      0,      0
            'the':        1,    1,     0,     0,      0,     0,      0,      0
            'force':      1,    1,     1,     0,      0,     0,      0,      0
            'be':         1,    1,     1,     1,      0,     0,      0,      0
            'with':       1,    1,     1,     1,      1,     0,      0,      0
            'you':        1,    1,     1,     1,      1,     1,      0,      0
            '<PAD>':      0,    0,     0,     0,      0,     0,      0,      0
            '<PAD>':      0,    0,     0,     0,      0,     0,      0,      0


    - attn_type = 'full'
        Each token will attend to both the tokens before and in the future

        For our example, the mask of the first sample is

        .. code-block:: none

                       ['I', 'can', 'now', 'use', 'numpy', 'in', 'Gluon@@', 'NLP']
            'I':         1,    1,     1,     1,      1,     1,      1,      1
            'can':       1,    1,     1,     1,      1,     1,      1,      1
            'now':       1,    1,     1,     1,      1,     1,      1,      1
            'use':       1,    1,     1,     1,      1,     1,      1,      1
            'numpy':     1,    1,     1,     1,      1,     1,      1,      1
            'in':        1,    1,     1,     1,      1,     1,      1,      1
            'Gluon@@':   1,    1,     1,     1,      1,     1,      1,      1
            'NLP':       1,    1,     1,     1,      1,     1,      1,      1

        The mask of the second sample is

        .. code-block:: none

                       ['May', 'the', 'force', 'be', 'with', 'you', '<PAD>', '<PAD>']
            'May':        1,    1,     1,     1,      1,     1,      0,      0
            'the':        1,    1,     1,     1,      1,     1,      0,      0
            'force':      1,    1,     1,     1,      1,     1,      0,      0
            'be':         1,    1,     1,     1,      1,     1,      0,      0
            'with':       1,    1,     1,     1,      1,     1,      0,      0
            'you':        1,    1,     1,     1,      1,     1,      0,      0
            '<PAD>':      0,    0,     0,     0,      0,     0,      0,      0
            '<PAD>':      0,    0,     0,     0,      0,     0,      0,      0

    Parameters
    ----------
    data
        The data.

        - layout = 'NT'
            Shape (batch_size, seq_length, C)
        - layout = 'TN'
            Shape (seq_length, batch_size, C)

    valid_length
        Shape (batch_size,)
    dtype
        Data type of the mask
    attn_type
        Can be 'full' or 'causal'
    layout
        The layout of the data

    Returns
    -------
    mask
        Shape (batch_size, seq_length, seq_length)
    """
    if layout == 'NT':
        batch_axis, time_axis = 0, 1
    elif layout == 'TN':
        batch_axis, time_axis = 1, 0
    else:
        raise NotImplementedError('Unsupported layout={}'.format(layout))
    if attn_type == 'full':
        if valid_length is not None:
            valid_length = valid_length.astype(dtype)
            steps = npx.arange_like(data, axis=time_axis)  # (seq_length,)
            mask1 = (npx.reshape(steps, (1, 1, -1))
                     < npx.reshape(valid_length, (-2, 1, 1)))
            mask2 = (npx.reshape(steps, (1, -1, 1))
                     < npx.reshape(valid_length, (-2, 1, 1)))
            mask = mask1 * mask2
        else:
            # TODO(sxjscience) optimize
            seq_len_ones = np.ones_like(npx.arange_like(data, axis=time_axis))  # (seq_length,)
            batch_ones = np.ones_like(npx.arange_like(data, axis=batch_axis))   # (batch_size,)
            mask = batch_ones.reshape((-1, 1, 1)) * seq_len_ones.reshape((1, -1, 1))\
                   * seq_len_ones.reshape((1, 1, -1))
    elif attn_type == 'causal':
        steps = npx.arange_like(data, axis=time_axis)
        # mask: (seq_length, seq_length)
        # batch_mask: (batch_size, seq_length)
        mask = (np.expand_dims(steps, axis=0) <= np.expand_dims(steps, axis=1)).astype(dtype)
        if valid_length is not None:
            valid_length = valid_length.astype(dtype)
            batch_mask = (np.expand_dims(steps, axis=0) < np.expand_dims(valid_length, axis=-1)).astype(dtype)
            mask = mask * np.expand_dims(batch_mask, axis=-1)
        else:
            batch_ones = np.ones_like(npx.arange_like(data, axis=batch_axis),
                                        dtype=dtype)  # (batch_size,)
            mask = mask * batch_ones.reshape((-1, 1, 1))
    else:
        raise NotImplementedError
    return mask.astype(np.bool)


def gen_mem_attn_mask(mem, mem_valid_length, data, data_valid_length=None,
                      dtype=np.float32, layout: str = 'NT'):
    """Generate the mask used for the decoder. All query slots are attended to the memory slots.

    In our implementation, 1 --> not masked, 0 --> masked

    Let's consider the data + mem with a batch of two samples:

    .. code-block:: none

        mem = [['I',   'can', 'now',   'use'],
               ['May', 'the', 'force', '<PAD>']]
        mem_valid_length =
            [4, 3]
        data =
            [['numpy', 'in',    'Gluon@@', 'NLP'  ],
             ['be',    'with',  'you',     '<PAD>']]
        data_valid_length =
            [4, 3]

    For our example, the mask of the first sample is

    .. code-block:: none

                   ['I', 'can', 'now', 'use']
        'numpy':     1,    1,     1,     1
        'in':        1,    1,     1,     1
        'Gluon@@':   1,    1,     1,     1
        'NLP':       1,    1,     1,     1

    The mask of the second sample is

    .. code-block:: none

                   ['be', 'with', 'you', '<PAD>']
        'May':        1,    1,     1,     0
        'the':        1,    1,     1,     0
        'force':      1,    1,     1,     0
        '<PAD>':      0,    0,     0,     0


    Parameters
    ----------
    mem
       - layout = 'NT'
            Shape (batch_size, mem_length, C_mem)
       - layout = 'TN'
            Shape (mem_length, batch_size, C_mem)

    mem_valid_length :
        Shape (batch_size,)
    data
        - layout = 'NT'
            Shape (batch_size, query_length, C_data)
        - layout = 'TN'
            Shape (query_length, batch_size, C_data)

    data_valid_length :
        Shape (batch_size,)
    dtype
        Data type of the mask
    layout
        Layout of the data + mem tensor

    Returns
    -------
    mask :
        Shape (batch_size, query_length, mem_length)
    """
    if layout == 'NT':
        batch_axis, time_axis = 0, 1
    elif layout == 'TN':
        batch_axis, time_axis = 1, 0
    else:
        raise NotImplementedError('Unsupported layout={}'.format(layout))
    mem_valid_length = mem_valid_length.astype(dtype)
    mem_steps = npx.arange_like(mem, axis=time_axis)  # (mem_length,)
    data_steps = npx.arange_like(data, axis=time_axis)  # (query_length,)
    mem_mask = (npx.reshape(mem_steps, (1, 1, -1))
                < npx.reshape(mem_valid_length, (-2, 1, 1))).astype(dtype)  # (B, 1, mem_length)
    if data_valid_length is not None:
        data_valid_length = data_valid_length.astype(dtype)
        data_mask = (npx.reshape(data_steps, (1, -1, 1))
                     < npx.reshape(data_valid_length, (-2, 1, 1))).astype(dtype)  # (B, query_length, 1)
        mask = mem_mask * data_mask
    else:
        query_length_ones = np.ones_like(data_steps)
        mask = query_length_ones.reshape((1, -1, 1)) * mem_mask
    return mask.astype(np.bool)


def masked_softmax(att_score, mask, axis: int = -1, temperature=None):
    """Ignore the masked elements when calculating the softmax. The mask can be broadcastable.

    Parameters
    ----------
    att_score : Symbol or NDArray
        Shape (..., length, ...)
    mask : Symbol or NDArray or None
        Shape (..., length, ...)
        1 --> The element is not masked
        0 --> The element is masked
    axis
        The axis to calculate the softmax. att_score.shape[axis] must be the same as mask.shape[axis]
    temperature
        The temperature. It scales down the scores before applying the softmax.

    Returns
    -------
    att_weights : Symborl or NDArray
        Shape (..., length, ...)
    """
    if mask is None:
        return npx.softmax(att_score, axis=axis, temperature=temperature)
    else:
        return npx.masked_softmax(att_score, mask=mask.astype(np.bool),
                                  axis=axis, temperature=temperature)


def masked_logsoftmax(att_score, mask, axis: int = -1):
    """Ignore the masked elements when calculating the softmax. The mask can be broadcastable.

    Parameters
    ----------
    att_score : Symborl or NDArray
        Shape (..., length, ...)
    mask : Symbol or NDArray or None
        Shape (..., length, ...)
        mask = 1 --> not masked
        mask = 0 --> masked
    axis
        The axis to calculate the softmax. att_score.shape[axis] must be the same as mask.shape[axis]

    Returns
    -------
    logits : Symborl or NDArray
        Shape (..., length, ...)
        The masked values will be all zero
    """
    if mask is None:
        return npx.log_softmax(att_score, axis=axis)
    else:
        mask = mask.astype(np.bool)
        return np.where(mask, npx.masked_log_softmax(att_score, mask, axis=axis), -np.inf)


def multi_head_dot_attn(query, key, value,
                        mask=None,
                        edge_scores=None,
                        dropout: float = 0.0,
                        scaled: bool = True, normalized: bool = False,
                        eps: float = 1E-6, query_head_units: Optional[int] = None,
                        layout: str = 'NKT',
                        use_einsum: bool = False):
    """Multihead dot product attention between the query, key, value.

    scaled is False, normalized is False:
        D(h_q, h_k) = <h_q, h_k>
    scaled is True, normalized is False:
        D(h_q, h_k) = <h_q, h_k> / sqrt(dim_q)
    scaled is False, normalized is True:
        D(h_q, h_k) = <h_q / ||h_q||, h_k / ||h_k||>
    scaled is True, normalized is True:
        D(h_q, h_k) = <h_q / ||h_q||, h_k / ||h_k||> / sqrt(dim_q)

    If edge_scores is provided, we will calcualte the attention as
        scores = D(h_q, h_k) + EdgeScore_{q, k}

    Parameters
    ----------
    query
        Query. The shape depends on the layout

        - layout is 'NKT'
            Shape (batch_size, num_heads, query_length, key_dim)
        - layout is 'NTK'
            Shape (batch_size, query_length, num_heads, key_dim)
        - layout is 'TNK'
            Shape (query_length, batch_size, num_heads, key_dim)

    key
        Key. The shape depends on the layout

        - layout is 'NKT'
            Shape (batch_size, num_heads, mem_length, key_dim)
        - layout is 'NTK'
            Shape (batch_size, mem_length, num_heads, key_dim)
        - layout is 'TNK'
            Shape (mem_length, batch_size, num_heads, key_dim)

    value
        Value. The shape depends on the layout

        - layout is 'NKT'
            Shape (batch_size, num_heads, mem_length, value_dim)
        - layout is 'NTK'
            Shape (batch_size, mem_length, num_heads, value_dim)
        - layout is 'TNK'
            Shape (mem_length, batch_size, num_heads, value_dim)

    mask
        Mask between query and memory. Shape (batch_size, query_length, mem_length)
    edge_scores
        The edge attention score. Shape can be any shape that is broadcastable to
        (batch_size, num_heads, query_length, mem_length)
    dropout
        Dropout rate
    scaled
        Whether to divide the attention weights by the sqrt of the query dimension.
        This is first proposed in "[NIPS2017] Attention is all you need."::

        .. code-block:: none

            score = <h_q, h_k> / sqrt(dim_q)

    normalized
        If turned on, the cosine distance is used, i.e::

        .. code-block:: none

            score = <h_q / ||h_q||, h_k / ||h_k||>

    eps
        The epsilon value used in L2 normalization
    query_head_units
        The units of each query head. If it's empty, we will estimate it via the
        shape_array of the query.
    layout
        This stands for the layout of the attention cell. The shape of the input/output will depend
        on the layout. Currently, we support 'NKT', 'NTK' and 'TNK' in which
        'N' means the batch_size, 'K' means the head, and 'T' means the length dimension.
    use_einsum
        Whether to use einsum for the computation

    Returns
    -------
    context_vec
        - layout is 'NKT' or 'NTK'
            Shape (batch_size, query_length, num_heads * value_units)
        - layout is 'TNK'
            Shape (query_length, batch_size, num_heads * value_units)

    additional_info
        scores:
            Shape (batch_size, num_head, query_length, mem_length)
        attn_weight:
            Shape (batch_size, num_head, query_length, mem_length)
    """
    # TODO(sxjscience) Profile layout
    if normalized:
        query = l2_normalize(query, axis=-1, eps=eps)
        key = l2_normalize(key, axis=-1, eps=eps)
    if scaled:
        if query_head_units is None:
            raise NotImplementedError('You will need to specify query_head_units!')
        else:
            scale = math.sqrt(query_head_units)
    else:
        scale = None
    if layout == 'NKT':
        # 1. Expand the dimension of the mask:
        #   (B, L_query, L_mem) --> (B, 1, L_query, L_mem)
        if mask is not None:
            mask = np.expand_dims(mask, axis=1).astype(np.bool)
        # 2. Calculate the attention weights
        #   Score: (B, N, L_query, C_Q) X (B, N, L_mem, C_Q) --> (B, N, L_query, L_mem)
        scores = npx.batch_dot(query, key, transpose_b=True)
        if edge_scores is not None:
            scores = scores + edge_scores
        attn_weights = masked_softmax(scores, mask, axis=-1, temperature=scale)
        attn_weights = npx.dropout(attn_weights, p=dropout)
        # 3. Calculate the context vector
        # (B, N, L_query, L_mem) X (B, N, L_mem, C_V) --> (B, L_query, N * C_V)
        if use_einsum:
            context_vec = np.einsum('bnij,bnjc->binc', attn_weights, value)
        else:
            context_vec = npx.batch_dot(attn_weights, value).transpose((0, 2, 1, 3))
        context_vec = npx.reshape(context_vec, (-2, -2, -1))
    elif layout == 'NTK':
        # 1. Expand the dimension of the mask:
        #   (B, L_query, L_mem) --> (B, 1, L_query, L_mem)
        if mask is not None:
            mask = np.expand_dims(mask, axis=1).astype(np.bool)
        # 2. Calculate the attention weights
        #   Score: (B, L_query, N, C_Q) X (B, L_mem, N, C_Q) --> (B, N, L_query, L_mem)
        if use_einsum:
            scores = np.einsum('binc,bjnc->bnij', query, key)
        else:
            scores = npx.batch_dot(np.swapaxes(query, 1, 2), np.swapaxes(key, 1, 2),
                                   transpose_b=True)
        if edge_scores is not None:
            scores = scores + edge_scores
        attn_weights = masked_softmax(scores, mask, axis=-1, temperature=scale)
        attn_weights = npx.dropout(attn_weights, p=dropout)
        # 3. Calculate the context vector
        # (B, N, L_query, L_mem) X (B, L_mem, N, C_V) --> (B, L_query, N * C_V)
        if use_einsum:
            context_vec = np.einsum('bnij,bjnc->binc', attn_weights, value)
        else:
            context_vec = npx.batch_dot(attn_weights,
                                          np.swapaxes(value, 1, 2)).transpose((0, 2, 1, 3))
        context_vec = npx.reshape(context_vec, (-2, -2, -1))
    elif layout == 'TNK':
        # 1. Expand the dimension of the mask:
        #   (B, L_query, L_mem) --> (B, 1, L_query, L_mem)
        if mask is not None:
            mask = np.expand_dims(mask, axis=1).astype(np.bool)
        # 2. Calculate the attention weights
        #   Score: (L_query, B, N, C_Q) X (L_mem, B, N, C_Q) --> (B, N, L_query, L_mem)
        #   This layout structure can be implemented very efficiently because B, N are consecutive
        #   to each other. To have a clear picture of what's happening, we may consider the
        #   (i, j)th element of the output
        #       out[i, j, :, :] = query[:, i, j, :] X key[:, i, j, :].T, which is just one GEMM call
        #   We can thus implement the whole kernel via a single call of batched GEMM with stride.
        if use_einsum:
            scores = np.einsum('ibnc,jbnc->bnij', query, key)
        else:
            scores = npx.batch_dot(query.transpose((1, 2, 0, 3)),
                                     key.transpose((1, 2, 3, 0)))
        if edge_scores is not None:
            scores = scores + edge_scores
        attn_weights = masked_softmax(scores, mask, axis=-1, temperature=scale)
        attn_weights = npx.dropout(attn_weights, p=dropout)
        # 3. Calculate the context vector
        # (B, N, L_query, L_mem) X (L_mem, B, N, C_V) --> (L_query, B, N * C_V)
        # Again, we can implement it via a single call to batched GEMM with stride.

        # Shape (B, N, L_query, C_V)
        if use_einsum:
            context_vec = np.einsum('bnij,jbnc->ibnc', attn_weights, value)
        else:
            context_vec = npx.batch_dot(attn_weights,
                                          value.transpose((1, 2, 0, 3))).transpose((2, 0, 1, 3))
        context_vec = npx.reshape(context_vec, (-2, -2, -1))
    else:
        raise NotImplementedError('layout="{}" is not supported! '
                                  'We only support layout = "NKT", "NTK", and "TNK".'
                                  .format(layout))
    return context_vec, [scores, attn_weights]


class MultiHeadAttentionCell(HybridBlock):
    """The multi-head attention

    out = softmax(<Q_i, K_j> + R_{i, j}) V

    We support multiple layouts

    Let's denote batch_size as B, num_heads as K,
     query_length as L_q, mem_length as L_m, key_dim as C_k, value_dim as C_v

    - layout="NKT"
        query: (B, K, L_q, C_k)
        key: (B, K, L_m, C_k)
        value: (B, K, L_m, C_v)
        out: (B, L_q, K * C_v)
    - layout="NTK"
        query: (B, L_q, K, C_k)
        key: (B, L_m, K, C_k)
        value: (B, L_m, K, C_v)
        out: (B, L_q, K * C_v)
    - layout="TNK"
        query: (L_q, B, K, C_k)
        key: (L_m, B, K, C_k)
        value: (L_m, B, K, C_v)
        out: (L_q, B, K * C_v)

    """
    def __init__(self, query_units=None, num_heads=None, attention_dropout=0.0,
                 scaled: bool = True, normalized: bool = False, eps: float = 1E-6,
                 dtype='float32', layout='NTK', use_einsum=False):
        super().__init__()
        self._query_units = query_units
        self._num_heads = num_heads
        self._attention_dropout = attention_dropout
        self._scaled = scaled
        self._normalized = normalized
        self._eps = eps
        self._dtype = dtype
        assert layout in ['NTK', 'NKT', 'TNK']
        self._layout = layout
        self._use_einsum = use_einsum
        if self._query_units is not None:
            assert self._num_heads is not None
            assert self._query_units % self._num_heads == 0,\
                'The units must be divisible by the number of heads.'
            self._query_head_units = self._query_units // self._num_heads
        else:
            self._query_head_units = None

    @property
    def layout(self):
        return self._layout

    def forward(self, query, key, value, mask=None, edge_scores=None):
        return multi_head_dot_attn(query=query, key=key, value=value,
                                   mask=mask, edge_scores=edge_scores,
                                   dropout=self._attention_dropout,
                                   scaled=self._scaled, normalized=self._normalized,
                                   eps=self._eps,
                                   query_head_units=self._query_head_units,
                                   layout=self._layout, use_einsum=self._use_einsum)

    def __repr__(self):
        s = '{name}(\n' \
            '   query_units={query_units},\n' \
            '   num_heads={num_heads},\n' \
            '   attention_dropout={attention_dropout},\n' \
            '   scaled={scaled},\n' \
            '   normalized={normalized},\n' \
            '   layout="{layout}",\n' \
            '   use_einsum={use_einsum},\n' \
            '   dtype={dtype}\n' \
            ')'
        return s.format(name=self.__class__.__name__,
                        query_units=self._query_units,
                        num_heads=self._num_heads,
                        attention_dropout=self._attention_dropout,
                        scaled=self._scaled,
                        normalized=self._normalized,
                        layout=self._layout,
                        use_einsum=self._use_einsum,
                        dtype=self._dtype)


def gen_rel_position(data, past_data=None, dtype=np.int32, layout='NT'): 
    """Create a matrix of relative position for RelAttentionScoreCell. 
    
    The relative position is defined as the index difference: `mem_i` - `query_j`. 
    Note, though, that the implementation here makes sense in self-attention's setting, 
    but not in cross-attention's. Hence, both `mem_i` and `query_j` are time indices from 
    `data` (or, in incremental decoding's case, the concatenated sequence from the current 
    stepwise `data` and the previous steps `past_data`). 

    Parameters
    ----------
    data
        The data. Under incremental decoding, seq_length = 1. 

        - layout = 'NT'
            Shape (batch_size, seq_length, C)
        - layout = 'TN'
            Shape (seq_length, batch_size, C)
    past_data
        This is only used under incremental decoding. Stacked data from previous steps. 
    dtype
        Data type of the mask
    layout
        Layout of the data + past_data

    Returns
    -------
    relative_position :
        Shape (query_length, mem_length) where query_length = mem_length = seq_length
    """
    time_axis = 1 if layout == 'NT' else 0
    if past_data is None: 
        position = npx.arange_like(data, axis=time_axis)
    else: 
        # for incremental decoding only, where past data is of the shape: 
        # NT(NTK): (B, L_seq, num_heads, n_kv) -> (B, L_seq, inner_dim)
        # TN(TNK): (L_seq, B, num_heads, n_kv) -> (L_seq, B, inner_dim)
        past_data = npx.reshape(past_data, (-2, -2, -5))
        position = npx.arange_like(
            np.concatenate([past_data, data], axis=time_axis), 
            axis=time_axis
        )
    query_position = np.expand_dims(position, axis=-1)
    mem_position = np.expand_dims(position, axis=0)
    relative_position = mem_position - query_position
    return relative_position.astype(np.int32) # shape (L_seq, L_seq)


class RelAttentionScoreCell(HybridBlock):
    r"""Get the score based on the query and relative position index. This is used for implementing
     relative attention.

    For the multi-head attention with relative positional encoding, we have the formula

    .. math::

        out = \text{softmax}(\frac{Q K^T + R}{\sqrt{d}}) V


    Here, :math:`R` is the relative positional encoding matrix.

    Usually, :math:`R_{i, j}` is calculate based on the
    relative positional difference :math:`i - j`.

    This function aims at generating the R matrix given the query and the relative positions.
    We support the following methods:

    - method = 'transformer_xl'
        :math:`R_{i, j} = <Q, W S_{i - j}>`, in which :math:`S_{i, j}` is the sinusoidal embedding and
        :math:`W` is a Dense layer that maps :math:`S_{i - j}` to the same dimension as the query.
        This is proposed in paper:

            [ACL2019] Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

    - method = 'shaw'
        :math:`R_{i, j} = < Q, E_{i - j}>`, in which :math:`E_{i - j}` is the learned positional embedding
        This is proposed in paper:

            [NAACL2018] Self-Attention with Relative Position Representations

    - method = 't5'
        :math:`R_{i, j} = E_{i - j}`, in which :math:`E_{i - j}` is the bucket positional embedding.
        This is proposed in paper:

            [Arxiv2019] Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer

    Like in MultiHeadAttentionCell, we support different layouts to cope with the query matrix.

    - layout="NKT"
        query: (B, K, L_q, C_k)
    - layout="NTK"
        query: (B, L_q, K, C_k)
    - layout="TNK"
        query: (L_q, B, K, C_k)

    """
    def __init__(self, query_units,
                 num_heads,
                 pos_embed_units: Optional[int] = None,
                 max_distance=None,
                 bidirectional=False,
                 num_buckets=None,
                 method='transformer_xl',
                 dropout: float = 0.0,
                 dtype='float32',
                 layout='NTK',
                 use_einsum=False, 
                 embed_initializer=None):
        """

        Parameters
        ----------
        query_units
        num_heads
        pos_embed_units
        max_distance
        bidirectional
        num_buckets
        method
        dropout
        dtype
        layout
        use_einsum
        """
        super().__init__()
        self._dropout = dropout
        self._method = method
        self._query_units = query_units
        self._num_heads = num_heads
        self._bidirectional = bidirectional
        self._num_buckets = num_buckets
        assert query_units % num_heads == 0, 'The units must be divisible by the number of heads.'
        self._head_query_units = query_units // num_heads
        self._max_distance = max_distance
        self._pos_embed_units = pos_embed_units
        self._dtype = dtype
        self._use_einsum = use_einsum
        self._layout = layout
        if self._layout not in ['NKT', 'NTK', 'TNK']:
            raise ValueError('layout="{}" is not supported'.format(self._layout))
        if method == 'transformer_xl':
            if pos_embed_units is None:
                pos_embed_units = self._num_heads * self._head_query_units
            self._rel_pos_embed = SinusoidalPositionalEmbedding(units=pos_embed_units,
                                                                dtype=self._dtype)
            self._rel_proj = nn.Dense(units=query_units,
                                      in_units=pos_embed_units,
                                      flatten=False,
                                      use_bias=False,
                                      dtype=self._dtype)
            self._dropout_layer = nn.Dropout(dropout)
        elif method == 'shaw':
            assert self._max_distance is not None, 'Must set max_distance when method="shaw".'
            if self._bidirectional:
                vocab_size = self._max_distance * 2 + 1
            else:
                vocab_size = self._max_distance + 1
            self._rel_pos_embed = LearnedPositionalEmbedding(
                units=self._num_heads * self._head_query_units,
                max_length=vocab_size,
                weight_initializer=mx.init.Xavier(rnd_type="gaussian",
                                                  factor_type="in",
                                                  magnitude=1),
                mode='wrap' if self._bidirectional else 'raise',
                dtype=self._dtype)
        elif method == 't5':
            if self._num_buckets is None:
                self._num_buckets = 32
            if self._max_distance is None:
                self._max_distance = 128
            self._rel_pos_embed = BucketPositionalEmbedding(
                units=num_heads,
                num_buckets=self._num_buckets,
                max_distance=self._max_distance,
                bidirectional=self._bidirectional,
                embed_initializer=embed_initializer,
                dtype=self._dtype)
        else:
            raise NotImplementedError('method="{}" is currently not supported!'.format(method))

    @property
    def layout(self) -> str:
        """Layout of the cell"""
        return self._layout

    def forward(self, rel_positions, query=None):
        """Forward function

        Parameters
        ----------
        rel_positions
            The relative shifts. Shape (query_length, mem_length).
            Each element represents the shift between the :math:`i-th` element of query and
            the :math:`j-th` element of memory.
        query
            The query for computing the relative scores. The shape depends on the layout.
            If we use T5 attention, the query will not be used.

        Returns
        -------
        rel_scores
            The relative attention scores
            Can have shape (batch_size, num_heads, query_length, mem_length)
            or (num_heads, query_length, mem_length)
        """
        if self._method == 'transformer_xl' or self._method == 'shaw':
            assert query is not None, 'Must specify query if method={}'.format(self._method)
            if self._bidirectional:
                if self._max_distance is not None:
                    rel_positions = np.clip(rel_positions,
                                              a_min=-self._max_distance, a_max=self._max_distance)
            else:
                if self._max_distance is not None:
                    rel_positions = np.clip(rel_positions,
                                              a_min=0, a_max=self._max_distance)
            # uniq_rel.shape = (#uniq,), rev_index.shape = (L_q, L_m)
            uniq_rel, rev_index = np.unique(rel_positions, return_inverse=True)

            uniq_rel_pos_embed = self._rel_pos_embed(uniq_rel)
            if self._method == 'transformer_xl':
                uniq_rel_pos_embed = self._rel_proj(self._dropout_layer(uniq_rel_pos_embed))
            # Shape (#uniq, K, C_q)
            uniq_rel_pos_embed = npx.reshape(uniq_rel_pos_embed,
                                               (-2, self._num_heads, self._head_query_units))
            # Calculate the dot-product between query and the relative positional embeddings.
            # After the calculation, rel_score.shape = (L_q, #uniq, N, K)
            if self._layout == 'NKT':
                # query_for_rel: (N, K, L_q, C_q)
                if self._use_einsum:
                    rel_score = np.einsum('bnid,jnd->ijbn', query, uniq_rel_pos_embed)
                else:
                    rel_score = np.transpose(
                        np.matmul(query,
                                    np.transpose(uniq_rel_pos_embed, (1, 2, 0))),
                        (2, 3, 0, 1)
                    )
            elif self._layout == 'NTK':
                # query_for_rel: (N, L_q, K, C_q)
                if self._use_einsum:
                    rel_score = np.einsum('bind,jnd->ijbn', query, uniq_rel_pos_embed)
                else:
                    rel_score = np.transpose(
                        np.matmul(np.swapaxes(query, 1, 2),
                                    np.transpose(uniq_rel_pos_embed, (1, 2, 0))),
                        (2, 3, 0, 1)
                    )
            elif self._layout == 'TNK':
                # query_for_rel: (L_q, N, K, C_q)
                if self._use_einsum:
                    rel_score = np.einsum('ibnd,jnd->ijbn', query, uniq_rel_pos_embed)
                else:
                    rel_score = np.transpose(
                        np.matmul(np.transpose(query, (1, 2, 0, 3)),
                                    np.transpose(uniq_rel_pos_embed, (1, 2, 0))),
                        (2, 3, 0, 1)
                    )
            else:
                raise NotImplementedError
            # We use gather_nd to select the elements
            # TODO(sxjscience) Use advanced indexing once available
            rev_index = npx.reshape_like(rev_index, rel_positions).astype(np.int32)
            query_idx = np.expand_dims(npx.arange_like(rel_positions, axis=0).astype(np.int32),
                                         axis=-1) + np.zeros_like(rev_index)
            rel_score = npx.gather_nd(rel_score, np.stack([query_idx, rev_index]))
            rel_score = np.transpose(rel_score, (2, 3, 0, 1))
        elif self._method == 't5':
            # shape is (K, L_q, L_m)
            rel_score = self._rel_pos_embed(rel_positions).transpose((2, 0, 1))
        else:
            raise NotImplementedError
        return rel_score
