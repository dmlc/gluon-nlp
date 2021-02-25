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
import torch as th
from torch import nn
import torch.nn.functional as F
from ..base import use_einsum_optimization
from typing import Optional


def gen_self_attn_mask(data,
                       valid_length=None,
                       attn_type: str = 'full',
                       layout: str = 'NT'):
    """Generate the mask used for the encoder, i.e, self-attention.

    In our implementation, 1 --> not masked, 0 --> masked
    Let's consider the data with two samples:
    data =
        [['I',   'can', 'now',   'use', 'numpy', 'in',  'Gluon@@', 'NLP'  ],
         ['May', 'the', 'force', 'be',  'with',  'you', '<PAD>',   '<PAD>']]
    valid_length =
        [8, 6]
    - attn_type = 'causal'
        Each token will attend to itself + the tokens before.
        It will not attend to tokens in the future.
        For our example, the mask of the first sample is
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
    attn_type
        Can be 'full' or 'causal'
    layout
        The layout of the data

    Returns
    -------
    mask
        Shape (batch_size, seq_length, seq_length)
    """
    device = data.device
    if layout == 'NT':
        batch_axis, time_axis = 0, 1
    elif layout == 'TN':
        batch_axis, time_axis = 1, 0
    else:
        raise NotImplementedError('Unsupported layout={}'.format(layout))
    if attn_type == 'full':
        if valid_length is not None:
            steps = th.arange(data.shape[time_axis], device=device)  # (seq_length,)
            mask1 = (steps.view((1, 1, -1))
                     < valid_length.view((valid_length.shape[0], 1, 1)))
            mask2 = (steps.view((1, -1, 1))
                     < valid_length.view((valid_length.shape[0], 1, 1)))
            mask = mask1 * mask2
        else:
            seq_len_ones = th.ones((data.shape[time_axis],), device=device)  # (seq_length,)
            batch_ones = th.ones((data.shape[batch_axis],), device=device)   # (batch_size,)
            mask = batch_ones.view((-1, 1, 1)) * seq_len_ones.view((1, -1, 1))\
                   * seq_len_ones.view((1, 1, -1))
    elif attn_type == 'causal':
        steps = th.arange(data.shape[time_axis], device=device)
        # mask: (seq_length, seq_length)
        # batch_mask: (batch_size, seq_length)
        mask = th.unsqueeze(steps, dim=0) <= th.unsqueeze(steps, dim=1)
        if valid_length is not None:
            batch_mask = th.unsqueeze(steps, dim=0) < th.unsqueeze(valid_length, dim=-1)
            mask = mask * th.unsqueeze(batch_mask, dim=-1)
        else:
            batch_ones = th.ones(data.shape[batch_axis], device=device)
            mask = mask * batch_ones.view((-1, 1, 1))
    else:
        raise NotImplementedError
    return mask.type(th.bool)


def gen_mem_attn_mask(mem, mem_valid_length, data, data_valid_length=None,
                      layout: str = 'NT'):
    """Generate the mask used for the decoder. All query slots are attended to the memory slots.

    In our implementation, 1 --> not masked, 0 --> masked
    Let's consider the data + mem with a batch of two samples:

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
                   ['I', 'can', 'now', 'use']
        'numpy':     1,    1,     1,     1
        'in':        1,    1,     1,     1
        'Gluon@@':   1,    1,     1,     1
        'NLP':       1,    1,     1,     1
    The mask of the second sample is
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
    layout
        Layout of the data + mem tensor

    Returns
    -------
    mask
        Shape (batch_size, query_length, mem_length)
    """
    device = mem.device
    if layout == 'NT':
        batch_axis, time_axis = 0, 1
    elif layout == 'TN':
        batch_axis, time_axis = 1, 0
    else:
        raise NotImplementedError('Unsupported layout={}'.format(layout))
    batch_size = mem.shape[batch_axis]
    mem_length = mem.shape[time_axis]
    query_length = data[time_axis]
    mem_steps = th.arange(mem_length, device=device)  # (mem_length,)
    data_steps = th.arange(data.shape[time_axis], device=device)  # (query_length,)
    # mem_mask will have shape (B, 1, mem_length)
    mem_mask = mem_steps.view((1, 1, mem_length)) < mem_valid_length.view((batch_size, 1, 1))
    if data_valid_length is not None:
        # (B, query_length, 1)
        data_mask = (data_steps.view((1, -1, 1))
                     < data_valid_length.view((batch_size, 1, 1)))
        mask = mem_mask * data_mask
    else:
        mask = mem_mask.expand(batch_size, query_length, -1)
    return mask.type(th.bool)


def masked_softmax(att_score, mask, axis: int = -1):
    """Ignore the masked elements when calculating the softmax.
     The mask can be broadcastable.

    Parameters
    ----------
    att_score : Symborl or NDArray
        Shape (..., length, ...)
    mask : Symbol or NDArray or None
        Shape (..., length, ...)
        1 --> The element is not masked
        0 --> The element is masked
    axis
        The axis to calculate the softmax. att_score.shape[axis] must be the same as mask.shape[axis]

    Returns
    -------
    att_weights : Symborl or NDArray
        Shape (..., length, ...)
    """
    if mask is not None:
        # Fill in the masked scores with a very small value
        if att_score.dtype == th.float16:
            att_score = att_score.masked_fill(th.logical_not(mask), -1E4)
        else:
            att_score = att_score.masked_fill(th.logical_not(mask), -1E18)
        att_weights = th.softmax(att_score, dim=axis) * mask
    else:
        att_weights = th.softmax(att_score, dim=axis)
    return att_weights


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
    if mask is not None:
        # Fill in the masked scores with a very small value
        inv_mask = th.logical_not(mask)
        if att_score.dtype == th.float16:
            att_score = att_score.masked_fill(inv_mask, -1E4)
        else:
            att_score = att_score.masked_fill(inv_mask, -1E18)
        logits = th.log_softmax(att_score, dim=axis)
        logits.masked_fill(inv_mask, float('-inf'))
    else:
        logits = th.log_softmax(att_score, dim=axis)
    return logits


def multi_head_dot_attn(query, key, value,
                        mask=None,
                        edge_scores=None,
                        dropout: float = 0.0,
                        scaled: bool = True, normalized: bool = False,
                        eps: float = 1E-6,
                        layout: str = 'NKT',
                        use_einsum: bool = None, *, training: bool = True):
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

            score = <h_q, h_k> / sqrt(dim_q)

    normalized
        If turned on, the cosine distance is used, i.e::

            score = <h_q / ||h_q||, h_k / ||h_k||>

    eps
        The epsilon value used in L2 normalization
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
    if use_einsum is None:
        use_einsum = use_einsum_optimization()
    # TODO(sxjscience) Profile layout
    if normalized:
        query = F.normalize(query, p=2, dim=-1, eps=eps)
        key = F.normalize(key, p=2, dim=-1, eps=eps)

    if scaled:
        scale = math.sqrt(query.shape[-1])
    else:
        scale = None

    if layout == 'NKT':
        # 1. Expand the dimension of the mask:
        #   (B, L_query, L_mem) --> (B, 1, L_query, L_mem)
        if mask is not None:
            mask = th.unsqueeze(mask, dim=1)
        # 2. Calculate the attention weights
        #   Score: (B, N, L_query, C_Q) X (B, N, L_mem, C_Q) --> (B, N, L_query, L_mem)
        scores = th.matmul(query, th.transpose(key, -2, -1))

        if edge_scores is not None:
            scores = scores + edge_scores
        attn_weights = masked_softmax(scores / scale if scale is not None else scores, mask, axis=-1)
        attn_weights = th.nn.functional.dropout(attn_weights, p=dropout, training=training)
        # 3. Calculate the context vector
        # (B, N, L_query, L_mem) X (B, N, L_mem, C_V) --> (B, L_query, N * C_V)
        if use_einsum:
            context_vec = th.einsum('bnij,bnjc->binc', attn_weights, value)
        else:
            context_vec = th.transpose(th.matmul(attn_weights, value), 1, 2)
        context_vec = th.reshape(context_vec,
                                 (context_vec.shape[0], context_vec.shape[1], -1))
    elif layout == 'NTK':
        # 1. Expand the dimension of the mask:
        #   (B, L_query, L_mem) --> (B, 1, L_query, L_mem)
        if mask is not None:
            mask = th.unsqueeze(mask, dim=1)
        # 2. Calculate the attention weights
        #   Score: (B, L_query, N, C_Q) X (B, L_mem, N, C_Q) --> (B, N, L_query, L_mem)
        if use_einsum:
            scores = th.einsum('binc,bjnc->bnij', query, key)
        else:
            scores = th.matmul(th.transpose(query, 1, 2), key.permute(0, 2, 3, 1))
        if edge_scores is not None:
            scores = scores + edge_scores
        attn_weights = masked_softmax(scores / scale if scale is not None else scores, mask)
        attn_weights = th.nn.functional.dropout(attn_weights, p=dropout, training=training)
        # 3. Calculate the context vector
        # (B, N, L_query, L_mem) X (B, L_mem, N, C_V) --> (B, L_query, N * C_V)
        if use_einsum:
            context_vec = th.einsum('bnij,bjnc->binc', attn_weights, value)
        else:
            context_vec = th.matmul(attn_weights, th.transpose(value, 1, 2)).permute(0, 2, 1, 3)
        context_vec = th.reshape(context_vec, (context_vec.shape[0], context_vec.shape[1], -1))
    elif layout == 'TNK':
        # 1. Expand the dimension of the mask:
        #   (B, L_query, L_mem) --> (B, 1, L_query, L_mem)
        if mask is not None:
            mask = th.unsqueeze(mask, dim=1)
        # 2. Calculate the attention weights
        #   Score: (L_query, B, N, C_Q) X (L_mem, B, N, C_Q) --> (B, N, L_query, L_mem)
        #   This layout structure can be implemented very efficiently because B, N are consecutive
        #   to each other. To have a clear picture of what's happening, we may consider the
        #   (i, j)th element of the output
        #       out[i, j, :, :] = query[:, i, j, :] X key[:, i, j, :].T, which is just one GEMM call
        #   We can thus implement the whole kernel via a single call of batched GEMM with stride.
        if use_einsum:
            scores = th.einsum('ibnc,jbnc->bnij', query, key)
        else:
            scores = th.matmul(query.permute(1, 2, 0, 3),
                               key.permute(1, 2, 3, 0))
        if edge_scores is not None:
            scores = scores + edge_scores
        attn_weights = masked_softmax(scores / scale if scale is not None else scores, mask)
        attn_weights = th.nn.functional.dropout(attn_weights, p=dropout, training=training)
        # 3. Calculate the context vector
        # (B, N, L_query, L_mem) X (L_mem, B, N, C_V) --> (L_query, B, N * C_V)
        # Again, we can implement it via a single call to batched GEMM with stride.

        # Shape (B, N, L_query, C_V)
        if use_einsum:
            context_vec = th.einsum('bnij,jbnc->ibnc', attn_weights, value)
        else:
            context_vec = th.matmul(attn_weights,
                                    value.permute(1, 2, 0, 3)).permute(2, 0, 1, 3)
        context_vec = th.reshape(context_vec, (context_vec.shape[0], context_vec.shape[1], -1))
    else:
        raise NotImplementedError('layout="{}" is not supported! '
                                  'We only support layout = "NKT", "NTK", and "TNK".'
                                  .format(layout))
    return context_vec, [scores, attn_weights]


class MultiHeadAttentionCell(nn.Module):
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
                 layout='NTK', use_einsum=None):
        super(MultiHeadAttentionCell, self).__init__()
        self._query_units = query_units
        self._num_heads = num_heads
        self._attention_dropout = attention_dropout
        self._scaled = scaled
        self._normalized = normalized
        self._eps = eps
        assert layout in ['NTK', 'NKT', 'TNK']
        self._layout = layout
        if use_einsum is None:
            use_einsum = use_einsum_optimization()
        self._use_einsum = use_einsum
        if self._query_units is not None:
            assert self._num_heads is not None
            assert self._query_units % self._num_heads == 0,\
                'The units must be divisible by the number of heads.'

    @property
    def layout(self):
        return self._layout

    def forward(self, query, key, value, mask=None, edge_scores=None):
        return multi_head_dot_attn(query=query, key=key, value=value,
                                   mask=mask, edge_scores=edge_scores,
                                   dropout=self._attention_dropout,
                                   scaled=self._scaled, normalized=self._normalized,
                                   eps=self._eps,
                                   layout=self._layout, use_einsum=self._use_einsum,
                                   training=self.training)

    def __repr__(self):
        s = '{name}(\n' \
            '   query_units={query_units},\n' \
            '   num_heads={num_heads},\n' \
            '   attention_dropout={attention_dropout},\n' \
            '   scaled={scaled},\n' \
            '   normalized={normalized},\n' \
            '   layout="{layout}",\n' \
            '   use_einsum={use_einsum}\n' \
            ')'
        return s.format(name=self.__class__.__name__,
                        query_units=self._query_units,
                        num_heads=self._num_heads,
                        attention_dropout=self._attention_dropout,
                        scaled=self._scaled,
                        normalized=self._normalized,
                        layout=self._layout,
                        use_einsum=self._use_einsum)
