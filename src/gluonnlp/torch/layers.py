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
"""Layers."""
import math
from collections import OrderedDict
import torch as th
import numpy as np
from torch import nn
from typing import Union
from .utils import to_torch_dtype


def sequence_mask(X, valid_len, value=0, axis=0):
    """Mask irrelevant entries in sequences."""
    assert axis in (0, 1)
    maxlen = X.size(axis)
    if axis == 1:
        mask = th.arange((maxlen), dtype=th.float32, device=X.device)[None, :] < valid_len[:, None]
    else:
        mask = th.arange((maxlen), dtype=th.float32, device=X.device)[:, None] < valid_len[None, :]
    X[~mask] = value
    return X


# TODO(sxjscience)
# Fix it to be the same as mesh-transformer
def relative_position_bucket(relative_position, bidirectional: bool = True, num_buckets: int = 32,
                             max_distance: int = 128):
    """Map the relative position to buckets.

    The major difference between our implementation and
    that in [mesh_tensorflow](https://github.com/tensorflow/mesh/blob/c59988047e49b4d2af05603e3170724cdbadc467/mesh_tensorflow/transformer/transformer_layers.py#L595-L637)
    is that we use 'query_i - mem_j' as the (i, j)-th location in relative_position.
    Thus, a positive value means that the query slot is in a later timestamp than the memory slot.
    However, in mesh transformer, it is treated as `mem_i - query_j` (reversed).
    The implementation uses the first half of the bucket (num_buckets // 2) to store the
    exact increments in positions and the second half of the bucket
    (num_buckets - num_buckets // 2) to store the bucketing values in the logarithm order.

    Parameters
    ----------
    relative_position
        Shape (...,)
    bidirectional
        Whether we are dealing with bidirectional attention.
        If it's bidirectional, we will use the first half to map the positions of the
        positive shifts and the second half to map the positions of the negative shifts.
    num_buckets
        The number of buckets.
    max_distance
        Maximum distance. Positions that fall outside of 'max_distance' will be trimmed.

    Returns
    -------
    buckets
        Shape (...,).
        It has the same shape as the `relative_position`. It will have int32 type.
    """
    ret = 0

    if bidirectional:
        assert num_buckets % 2 == 0, 'When bidirectional is True, the number of buckets must be ' \
                                     'divisible by 2.'
        num_buckets //= 2
        ret = ret + (relative_position < 0).astype(th.int32) * num_buckets
        relative_position = th.abs(relative_position)
    else:
        # Clip all the negative values to 0
        relative_position = th.clip(relative_position, min=0, max=None)
    # Now, the relative_position is in the range [0, inf)

    # Half of the buckets deal with the exact increments,
    # i.e., 0, 1, 2, ..., max_exact - 1, where max_exact = num_buckets // 2
    max_exact = num_buckets // 2
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to
    # max_distance
    val_if_large = max_exact + (th.log(relative_position.type(th.float32) / max_exact) /
                                math.log(max_distance / max_exact) *
                                (num_buckets - max_exact)).astype(th.int32)
    val_if_large = th.minimum(val_if_large, th.tensor(num_buckets - 1))
    ret = ret + th.where(is_small, relative_position, val_if_large)
    return ret


def get_activation(act, inplace=False):
    """

    Parameters
    ----------
    act
        Name of the activation
    inplace
        Whether to perform inplace activation

    Returns
    -------
    activation_layer
        The activation
    """
    if act is None:
        return lambda x: x
    if isinstance(act, str):
        if act == 'leaky':
            # TODO(sxjscience) Add regex matching here to parse `leaky(0.1)`
            return nn.LeakyReLU(0.1, inplace=inplace)
        elif act == 'identity':
            return nn.Identity()
        elif act == 'elu':
            return nn.ELU(inplace=inplace)
        elif act == 'gelu':
            return nn.GELU()
        elif act == 'gelu(tanh)':
            return GELU_TANH()
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'sigmoid':
            return nn.Sigmoid()
        elif act == 'tanh':
            return nn.Tanh()
        elif act == 'softrelu' or act == 'softplus':
            return nn.Softplus()
        elif act == 'softsign':
            return nn.Softsign()
        else:
            raise NotImplementedError('act="{}" is not supported. '
                                      'Try to include it if you can find that in '
                                      'https://pytorch.org/docs/stable/nn.html'.format(act))
    else:
        return act


def get_norm_layer(normalization: str = 'layer_norm', axis: int = -1, epsilon: float = 1e-5,
                   in_channels: int = 0, **kwargs):
    """Get the normalization layer based on the provided type

    Parameters
    ----------
    normalization
        The type of the layer normalization from ['layer_norm']
    axis
        The axis to normalize the
    epsilon
        The epsilon of the normalization layer
    in_channels
        Input channel

    Returns
    -------
    norm_layer
        The layer normalization layer
    """
    if isinstance(normalization, str):
        if normalization == 'layer_norm':
            assert in_channels > 0
            assert axis == -1
            norm_layer = nn.LayerNorm(normalized_shape=in_channels, eps=epsilon, **kwargs)
        else:
            raise NotImplementedError('normalization={} is not supported'.format(normalization))
        return norm_layer
    else:
        raise NotImplementedError('The type of normalization must be str')


class GELU_TANH(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """
    def forward(self, x):
        return 0.5 * x * (1 + th.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * th.pow(x, 3))))


class PositionwiseFFN(nn.Module):
    """The Position-wise FFN layer used in Transformer-like architectures

    If pre_norm is True:
        norm(data) -> fc1 -> act -> act_dropout -> fc2 -> dropout -> res(+data)
    Else:
        data -> fc1 -> act -> act_dropout -> fc2 -> dropout -> norm(res(+data))
    Also, if we use gated projection. We will use
        fc1_1 * act(fc1_2(data)) to map the data
    """
    def __init__(self, units: int = 512, hidden_size: int = 2048, activation_dropout: float = 0.0,
                 dropout: float = 0.1, gated_proj: bool = False, activation='relu',
                 normalization: str = 'layer_norm', layer_norm_eps: float = 1E-5,
                 pre_norm: bool = False):
        """
        Parameters
        ----------
        units
        hidden_size
        activation_dropout
        dropout
        activation
        normalization
            layer_norm or no_norm
        layer_norm_eps
        pre_norm
            Pre-layer normalization as proposed in the paper:
            "[ACL2018] The Best of Both Worlds: Combining Recent Advances in
             Neural Machine Translation"
            This will stabilize the training of Transformers.
            You may also refer to
            "[Arxiv2020] Understanding the Difficulty of Training Transformers"
        """
        super().__init__()
        self._pre_norm = pre_norm
        self._gated_proj = gated_proj
        self._kwargs = OrderedDict([
            ('units', units),
            ('hidden_size', hidden_size),
            ('activation_dropout', activation_dropout),
            ('activation', activation),
            ('dropout', dropout),
            ('normalization', normalization),
            ('layer_norm_eps', layer_norm_eps),
            ('gated_proj', gated_proj),
            ('pre_norm', pre_norm),
        ])
        self.dropout_layer = nn.Dropout(dropout)
        self.activation_dropout_layer = nn.Dropout(activation_dropout)
        self.ffn_1 = nn.Linear(in_features=units, out_features=hidden_size, bias=True)
        if self._gated_proj:
            self.ffn_1_gate = nn.Linear(in_features=units, out_features=hidden_size, bias=True)
        self.activation = get_activation(activation)
        self.ffn_2 = nn.Linear(in_features=hidden_size, out_features=units, bias=True)
        self.layer_norm = get_norm_layer(normalization=normalization, in_channels=units,
                                         epsilon=layer_norm_eps)
        self.init_weights()

    def init_weights(self):
        for module in self.children():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, data):
        """

        Parameters
        ----------
        data :
            Shape (B, seq_length, C_in)

        Returns
        -------
        out :
            Shape (B, seq_length, C_out)
        """
        residual = data
        if self._pre_norm:
            data = self.layer_norm(data)
        if self._gated_proj:
            out = self.activation(self.ffn_1_gate(data)) * self.ffn_1(data)
        else:
            out = self.activation(self.ffn_1(data))
        out = self.activation_dropout_layer(out)
        out = self.ffn_2(out)
        out = self.dropout_layer(out)
        out = out + residual
        if not self._pre_norm:
            out = self.layer_norm(out)
        return out


def get_positional_embedding(units, max_length=None, embed_method='learn_sinusoidal'):
    if embed_method == 'sinusoidal':
        return SinusoidalPositionalEmbedding(units, learnable=False)
    elif embed_method == 'learned_sinusoidal':
        return SinusoidalPositionalEmbedding(units, learnable=True)
    elif embed_method == 'learned':
        return nn.Embedding(num_embeddings=max_length, embedding_dim=units)


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, units: int, learnable=False):
        """Use a geometric sequence of timescales.

        It is calculated as
        [sin(wi x), cos(wi x), sin(wi x), cos(wi x), ...]

        By default, we initialize wi to be (1 / 10000) ^ (1 / (units//2 - 1))


        Parameters
        ----------
        units
            The number of units for positional embedding
        learnable
            Whether to make the Sinusoidal positional embedding learnable.
            If it is turned on, we will also update the frequency of this layer.
            See "[ICLR2021] On Position Embeddings in BERT" for more detail.

        """
        super().__init__()

        def _init_sinusoidal_base(units):
            half_units = units // 2
            val = np.log(10000) / (half_units - 1)
            val = np.exp(np.arange(half_units, dtype=np.float32) * -val)
            return val

        default_sinusoidal_base = _init_sinusoidal_base(units)

        self.freq = nn.Parameter(data=th.tensor(default_sinusoidal_base), requires_grad=learnable)
        self._units = units
        self._learnable = learnable

    def forward(self, positions):
        """

        Parameters
        ----------
        positions : th.Tensor
            Shape (..., )

        Returns
        -------
        ret :
            Shape (..., units)
        """
        emb = positions.unsqueeze(-1) * self.freq
        sin_emb = th.sin(emb)
        cos_emb = th.cos(emb)
        if self._units % 2 == 0:
            return th.cat([sin_emb, cos_emb], dim=-1)
        else:
            return th.cat([sin_emb, cos_emb, th.zeros_like(positions).unsqueeze(-1)], dim=-1)

    def __repr__(self):
        s = '{name}(units={units}, learnable={learnable})'
        return s.format(name=self.__class__.__name__, units=self._units, learnable=self._learnable)
