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
"""GPT models."""

__all__ = ['GPT2Model', 'GPT2SelfAttentionLayer', 'GPT2FFNLayer',
           'gpt2_117m', 'gpt2_345m']

import os
import numpy as np
import mxnet as mx
from mxnet.gluon import nn, Block, HybridBlock
from mxnet.gluon.model_zoo import model_store
from gluonnlp.model.attention_cell import DotProductAttentionCell
from gluonnlp.model.block import GELU
from gluonnlp.model.utils import _load_vocab, _load_pretrained_params
from gluonnlp.base import get_home_dir


class GPT2SelfAttentionLayer(Block):
    """Self-attention layer used in OpenAI GPT-2.

    Parameters
    ----------
    units : int
        Number of units for the output.
    num_heads : int
        Number of heads in multi-head attention
    dropout : float
        Dropout probability of the attention probabilities.
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default None.
        Prefix for name of `Block`s. (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells. Created if `None`.

    Inputs:
        - **inputs** : input sequence of shape (batch_size, length, in_dim).
        - **states** : None, or list of tensors
            The states, for initial states and masks that contains
            the previous encoded key/values
            prev_key (batch_size, num_heads, past_length, mem_length),
            prev_value (batch_size, num_heads, past_length, mem_length)
            None means no previous states.

    Outputs:
        - **outputs** : output encoding of shape (batch_size, length, C_out).
        - **additional_outputs** : list of tensors.
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, num_heads, length, mem_length)
    """
    def __init__(self, units, num_heads, dropout=0.0,
                 weight_initializer=mx.init.Normal(0.02), bias_initializer='zeros',
                 prefix=None, params=None):
        super(GPT2SelfAttentionLayer, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._num_heads = num_heads
        assert units % num_heads == 0
        with self.name_scope():
            self._multi_head_qkv_proj = nn.Dense(units=units * 3, flatten=False, use_bias=True,
                                                 weight_initializer=weight_initializer,
                                                 bias_initializer=bias_initializer,
                                                 prefix='qkv_proj_')
            self._base_attn_cell = DotProductAttentionCell(
                scaled=True, dropout=dropout, prefix='attn_')
            self._dropout_layer = nn.Dropout(dropout)
            self._out_proj = nn.Dense(units=units, flatten=False, use_bias=True,
                                      weight_initializer=weight_initializer,
                                      bias_initializer=bias_initializer,
                                      prefix='out_proj_')

    def forward(self, data, states=None): # pylint: disable=arguments-differ
        batch_size = data.shape[0]
        seq_len = data.shape[1]
        # Generate mask
        if states is not None:
            prev_key, prev_value = states
            prev_len = prev_key.shape[2]
        else:
            prev_key, prev_value = None, None
            prev_len = 0
        data_pos = mx.nd.arange(prev_len, prev_len + seq_len, ctx=data.context, dtype=data.dtype)
        all_pos = mx.nd.arange(seq_len + prev_len, ctx=data.context, dtype=data.dtype)
        mask = mx.nd.broadcast_lesser_equal(all_pos.reshape((1, -1)), data_pos.reshape((-1, 1)))
        mask = mx.nd.broadcast_axes(mx.nd.expand_dims(mask, axis=0), axis=0,
                                    size=batch_size * self._num_heads)

        # Multi-head attention
        qkv = self._multi_head_qkv_proj(data)  # Shape (batch_size, seq_len, 3 * units)
        qkv = mx.nd.swapaxes(qkv, 1, 2)  # Shape (batch_size, 3 * units, seq_len)

        # Each has shape (batch_size, units, seq_len)
        query, key, value = mx.nd.split(qkv, num_outputs=3, axis=1)
        # Map each to have shape (batch_size * num_head, ele_units, seq_len)
        query = query.reshape(shape=(0, -4, self._num_heads, -1, 0)).reshape(
            shape=(-1, 0, 0), reverse=True)
        key = key.reshape(shape=(0, -4, self._num_heads, -1, 0)).reshape(
            shape=(-1, 0, 0), reverse=True)
        value = value.reshape(shape=(0, -4, self._num_heads, -1, 0)).reshape(
            shape=(-1, 0, 0), reverse=True)
        query = mx.nd.swapaxes(query, 1, 2)
        key = mx.nd.swapaxes(key, 1, 2)
        value = mx.nd.swapaxes(value, 1, 2)
        if prev_key is not None:
            key = mx.nd.concat(prev_key.reshape((-1, 0, 0), reverse=True),
                               key, dim=1)  # Shape (batch_size * num_heads, all_len, ele_units)
        if prev_value is not None:
            value = mx.nd.concat(prev_value.reshape((-1, 0, 0), reverse=True),
                                 value, dim=1)

        # Shape (batch_size * num_heads, all_len, ele_units)
        out, _ = self._base_attn_cell(query, key, value, mask)
        out = mx.nd.transpose(out.reshape((-1, self._num_heads, 0, 0), reverse=True),
                              axes=(0, 2, 1, 3)).reshape((0, 0, -1))
        out = self._out_proj(out)
        return out, [key.reshape((-1, self._num_heads, 0, 0), reverse=True),
                     value.reshape((-1, self._num_heads, 0, 0), reverse=True)]


class GPT2FFNLayer(HybridBlock):
    """Feed-forward network (FFN) layer used in OpenAI GPT-2.

    Parameters
    ----------
    units : int
        Number of units for the output.
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    num_heads : int
        Number of heads in multi-head attention
    dropout : float
        Dropout probability of the attention probabilities.
    prefix : str, default None.
        Prefix for name of `Block`s. (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells. Created if `None`.


    Inputs:
        - **inputs** : input sequence of shape (batch_size, length, C_in)

    Outputs:
        - **outputs** : the output of the encoder. Shape is (batch_size, length, C_out)
    """
    def __init__(self, units, hidden_size,
                 weight_initializer=mx.init.Normal(0.02), bias_initializer='zeros',
                 prefix=None, params=None):
        super(GPT2FFNLayer, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._hidden_size = hidden_size
        with self.name_scope():
            self._hidden_map = nn.Dense(flatten=False, units=hidden_size,
                                        weight_initializer=weight_initializer,
                                        bias_initializer=bias_initializer)
            self._out_map = nn.Dense(flatten=False, units=units,
                                     weight_initializer=weight_initializer,
                                     bias_initializer=bias_initializer)
            self._act = GELU()

    def hybrid_forward(self, F, data): # pylint: disable=arguments-differ
        out = self._out_map(self._act(self._hidden_map(data)))
        return out


class GPT2Model(Block):
    """Generic Model for GPT-2.

    Parameters
    ----------
    units : int
        Number of units for the output.
    vocab_size : int or None, default None
        The size of the vocabulary.
    max_length : int
        Maximum length of the input sequence
    num_layers : int
        Number of attention layers.
    num_heads : int
        Number of heads in multi-head attention
    dropout : float
        Dropout probability of the attention probabilities.
    prefix : str, default None.
        Prefix for name of `Block`s. (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells. Created if `None`.
    """
    def __init__(self, units, vocab_size, max_length, num_layers, num_heads, dropout=0.0,
                 prefix=None, params=None):
        super(GPT2Model, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._max_length = max_length
        self._num_layers = num_layers
        self._num_heads = num_heads
        with self.name_scope():
            self._pos_embed = nn.Embedding(input_dim=max_length, output_dim=units,
                                           weight_initializer=mx.init.Normal(0.01),
                                           prefix='pos_embed_')
            self._embed = nn.Embedding(input_dim=vocab_size, output_dim=units, prefix='embed_',
                                       weight_initializer=mx.init.Normal(0.02))
            self._logits_proj = nn.Dense(units=vocab_size, in_units=units, use_bias=False,
                                         flatten=False, params=self._embed.params)
            self._self_attention_layers = nn.Sequential()
            self._ffn_layers = nn.HybridSequential()
            self._attn_ln = nn.HybridSequential()
            self._ffn_ln = nn.HybridSequential()
            for i in range(num_layers):
                self._self_attention_layers.add(GPT2SelfAttentionLayer(
                    units=units, num_heads=num_heads, dropout=dropout,
                    prefix='self_attn{}_'.format(i)))
                self._ffn_layers.add(GPT2FFNLayer(
                    units=units, hidden_size=units * 4, prefix='ffn{}_'.format(i)))
                self._attn_ln.add(nn.LayerNorm(prefix='attn_ln{}_'.format(i)))
                self._ffn_ln.add(nn.LayerNorm(prefix='ffn_ln{}_'.format(i)))
                self._final_ln = nn.LayerNorm(prefix='final_ln{}_'.format(i))

    def forward(self, data, states=None): # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        data : NDArray
            Shape (batch_size, seq_len)
        states : list of NDArray or None

        Returns
        -------
        out : NDArray
            Shape (batch_size, seq_len, vocab_size)
        new_states : list of NDArray
        """
        new_states = []
        batch_size, seq_len = data.shape[0], data.shape[1]
        if states is not None:
            prev_len = states[0].shape[1]
        else:
            prev_len = 0
        assert seq_len + prev_len <= self._max_length
        data_pos = mx.nd.arange(prev_len, prev_len + seq_len, ctx=data.context, dtype=np.float32)
        data_pos = mx.nd.broadcast_axes(mx.nd.expand_dims(data_pos, axis=0),
                                        axis=0, size=batch_size)
        out = self._embed(data) + self._pos_embed(data_pos)
        for i in range(self._num_layers):
            attn_layer = self._self_attention_layers[i]
            ffn_layer = self._ffn_layers[i]
            attn_ln = self._attn_ln[i]
            ffn_ln = self._ffn_ln[i]
            layer_states = None if states is None else states[2*i:(2*i + 2)]
            h, new_layer_states = attn_layer(attn_ln(out), layer_states)
            out = out + h
            h = ffn_layer(ffn_ln(out))
            out = out + h
            new_states.extend(new_layer_states)
        out = self._final_ln(out)
        logits = self._logits_proj(out)
        return logits, new_states

    def state_info(self, *args, **kwargs): # pylint: disable=unused-argument
        return None

model_store._model_sha1.update(
    {name: checksum for checksum, name in [
        ('26416f2ec2ab0c5f37e74dcec801f3e659546e03', 'gpt2_117m_openai_webtext'),
        ('29173e25d2f3b187745bea6689693bb771862f81', 'gpt2_345m_openai_webtext'),
    ]})

gpt2_117m_hparams = {
    'units': 768,
    'max_length': 1024,
    'num_heads': 12,
    'num_layers': 12,
    'dropout': 0.0,
}

gpt2_345m_hparams = {
    'units': 1024,
    'max_length': 1024,
    'num_heads': 16,
    'num_layers': 24,
    'dropout': 0.0,
}

gpt2_hparams = {
    'gpt2_117m': gpt2_117m_hparams,
    'gpt2_345m': gpt2_345m_hparams,
}

def gpt2_117m(dataset_name=None, vocab=None, pretrained=True, ctx=mx.cpu(),
              root=os.path.join(get_home_dir(), 'models'), **kwargs):
    """Generic GPT-2 model.

    The number of layers (L) is 12, number of units (H) is 768, and the
    number of self-attention heads (A) is 12.

    Parameters
    ----------
    dataset_name : str or None, default None
        If not None, the dataset name is used to load a vocabulary for the
        dataset. If the `pretrained` argument is set to True, the dataset name
        is further used to select the pretrained parameters to load.
        Options include 'book_corpus_wiki_en_uncased' and 'book_corpus_wiki_en_cased'.
    vocab : gluonnlp.vocab.BERTVocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset_name is not
        specified. Ignored if dataset_name is specified.
    pretrained : bool, default True
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
        MXNET_HOME defaults to '~/.mxnet'.

    Returns
    -------
    GPT2Model, gluonnlp.vocab.Vocab
    """
    return _get_gpt2_model('gpt2_117m', dataset_name=dataset_name, vocab=vocab,
                           pretrained=pretrained, ctx=ctx, root=root,
                           **kwargs)


def gpt2_345m(dataset_name=None, vocab=None, pretrained=True, ctx=mx.cpu(),
              root=os.path.join(get_home_dir(), 'models'), **kwargs):
    """Generic GPT-2 model.

    The number of layers (L) is 24, number of units (H) is 1024, and the
    number of self-attention heads (A) is 24.

    Parameters
    ----------
    dataset_name : str or None, default None
        If not None, the dataset name is used to load a vocabulary for the
        dataset. If the `pretrained` argument is set to True, the dataset name
        is further used to select the pretrained parameters to load.
        Options include 'book_corpus_wiki_en_uncased' and 'book_corpus_wiki_en_cased'.
    vocab : gluonnlp.vocab.BERTVocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset_name is not
        specified. Ignored if dataset_name is specified.
    pretrained : bool, default True
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
        MXNET_HOME defaults to '~/.mxnet'.

    Returns
    -------
    GPT2Model, gluonnlp.vocab.Vocab
    """
    return _get_gpt2_model('gpt2_345m', dataset_name=dataset_name, vocab=vocab,
                           pretrained=pretrained, ctx=ctx, root=root,
                           **kwargs)


def _get_gpt2_model(model_name=None, dataset_name=None, vocab=None, pretrained=True, ctx=mx.cpu(),
                    root=os.path.join(get_home_dir(), 'models'), **kwargs):
    """Any predefined GPT-2 model.

    Parameters
    ----------
    model_name : str or None, default None
        Options include 'gpt2_117m' and 'gpt2_345m'.
    dataset_name : str or None, default None
        If not None, the dataset name is used to load a vocabulary for the
        dataset. If the `pretrained` argument is set to True, the dataset name
        is further used to select the pretrained parameters to load.
        The supported datasets for model_name of either bert_24_1024_16 and
        bert_12_768_12 are 'openai_webtext'.
    vocab : gluonnlp.vocab.BERTVocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset_name is not
        specified. Ignored if dataset_name is specified.
    pretrained : bool, default True
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
        MXNET_HOME defaults to '~/.mxnet'.

    Returns
    -------
    GPT2Model, gluonnlp.vocab.Vocab
    """
    predefined_args = gpt2_hparams[model_name]
    mutable_args = ['dropout']
    mutable_args = frozenset(mutable_args)
    assert all((k not in kwargs or k in mutable_args) for k in predefined_args), \
        'Cannot override predefined model settings.'
    predefined_args.update(kwargs)
    vocab = _load_vocab(dataset_name, vocab, root)
    # BERT
    net = GPT2Model(units=predefined_args['units'],
                    vocab_size=len(vocab),
                    max_length=predefined_args['max_length'],
                    num_layers=predefined_args['num_layers'],
                    num_heads=predefined_args['num_heads'],
                    dropout=predefined_args['dropout'],
                    **kwargs)
    if pretrained:
        _load_pretrained_params(net, model_name, dataset_name, root, ctx)
    for i in range(net._num_layers):
        net._ffn_layers[i]._act._support_erf = False
    return net, vocab
