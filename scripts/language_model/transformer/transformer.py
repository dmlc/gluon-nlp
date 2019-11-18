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

__all__ = ['TransformerXLCell', 'TransformerXL', 'XLNet']

import typing

import numpy as np
import mxnet as mx
from mxnet.gluon import nn

import gluonnlp as nlp

from .attention_cell import PositionalEmbeddingMultiHeadAttentionCell, \
                             RelativeSegmentEmbeddingPositionalEmbeddingMultiHeadAttentionCell
from .embedding import AdaptiveEmbedding, ProjectedEmbedding
from .softmax import AdaptiveLogSoftmaxWithLoss, ProjectedLogSoftmaxWithLoss


class PositionalEmbedding(mx.gluon.HybridBlock):
    """Positional embedding.

    Parameters
    ----------
    embed_size : int
        Dimensionality of positional embeddings.
    """

    def __init__(self, embed_size, **kwargs):
        super().__init__(**kwargs)

        inv_freq = 1 / mx.nd.power(10000, mx.nd.arange(0.0, embed_size, 2.0) / embed_size)
        with self.name_scope():
            self.inv_freq = self.params.get_constant('inv_freq', inv_freq.reshape((1, -1)))

    def hybrid_forward(self, F, pos_seq, inv_freq):  # pylint: disable=arguments-differ
        """Compute positional embeddings.

        Parameters
        ----------
        pos_seq : Symbol or NDArray
            Positions to compute embedding for. Shape (length, )

        Returns
        -------
        pos_emb: Symbol or NDArray
            Positional embeddings for positions secified in pos_seq. Shape
            (length, embed_size).
        """
        inp = F.dot(pos_seq.reshape((-1, 1)), inv_freq)
        pos_emb = F.concat(F.sin(inp), F.cos(inp), dim=-1)
        return pos_emb


class TransformerXLCell(mx.gluon.HybridBlock):
    """Transformer-XL Cell.

    Parameters
    ----------
    attention_cell
        Attention cell to be used.
    units : int
        Number of units for the output
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
    attention_dropout : float
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default None
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """

    def __init__(self, attention_cell: PositionalEmbeddingMultiHeadAttentionCell, units=128,
                 hidden_size=512, num_heads=4, activation='relu', scaled=True, dropout=0.0,
                 use_residual=True, output_attention=False, weight_initializer=None,
                 bias_initializer='zeros', prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self._units = units
        self._num_heads = num_heads
        self._activation = activation
        self._dropout = dropout
        self._use_residual = use_residual
        self._output_attention = output_attention
        self._scaled = scaled
        with self.name_scope():
            if dropout:
                self.dropout_layer = nn.Dropout(rate=dropout)
            assert units % num_heads == 0
            self.attention_cell = attention_cell
            self.proj = nn.Dense(units=units, flatten=False, use_bias=False,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer, prefix='proj_')
            self.ffn = nlp.model.PositionwiseFFN(hidden_size=hidden_size, units=units,
                                                 use_residual=use_residual, dropout=dropout,
                                                 ffn1_dropout=True, activation=activation,
                                                 weight_initializer=weight_initializer,
                                                 bias_initializer=bias_initializer,
                                                 layer_norm_eps=1e-12)
            self.layer_norm = nn.LayerNorm(in_channels=units, epsilon=1e-12)

    def hybrid_forward(self, F, inputs, pos_emb, mem_value, mask):
        #  pylint: disable=arguments-differ
        """Transformer Decoder Attention Cell.

        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (batch_size, length, C_in)
        mem_value : Symbol or NDArray
            Memory value, i.e. output of the encoder. Shape (batch_size, mem_length, C_in)
        pos_emb : Symbol or NDArray
            Positional embeddings. Shape (mem_length, C_in)
        mask : Symbol or NDArray or None
            Attention mask of shape (batch_size, length, length + mem_length)

        Returns
        -------
        decoder_cell_outputs: list
            Outputs of the decoder cell. Contains:

            - outputs of the transformer decoder cell. Shape (batch_size, length, C_out)
            - additional_outputs of all the transformer decoder cell
        """
        key_value = F.concat(mem_value, inputs, dim=1)
        outputs, attention_outputs = self.attention_cell(inputs, key_value, key_value, pos_emb,
                                                         mask)
        outputs = self.proj(outputs)
        if self._dropout:
            outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm(outputs)
        outputs = self.ffn(outputs)
        additional_outputs = [attention_outputs] if self._output_attention else []
        return outputs, additional_outputs


class _BaseTransformerXL(mx.gluon.HybridBlock):
    def __init__(self, vocab_size, embed_size, embed_cutoffs=None, embed_div_val=None, num_layers=2,
                 units=128, hidden_size=2048, num_heads=4, scaled=True, dropout=0.0,
                 attention_dropout=0.0, use_residual=True, clamp_len: typing.Optional[int] = None,
                 project_same_dim: bool = True, tie_input_output_embeddings: bool = False,
                 tie_input_output_projections: typing.Optional[typing.List[bool]] = None,
                 output_attention=False, weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        assert units % num_heads == 0, 'In TransformerDecoder, the units should be divided ' \
                                       'exactly by the number of heads. Received units={}, ' \
                                       'num_heads={}'.format(units, num_heads)

        self._num_layers = num_layers
        self._units = units
        self._embed_size = embed_size
        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._dropout = dropout
        self._use_residual = use_residual
        self._clamp_len = clamp_len
        self._project_same_dim = project_same_dim
        self._tie_input_output_embeddings = tie_input_output_embeddings
        self._tie_input_output_projections = tie_input_output_projections
        if output_attention:
            # Will be implemented when splitting this Block to separate the
            # AdaptiveLogSoftmaxWithLoss used with targets
            raise NotImplementedError()
        self._output_attention = output_attention
        with self.name_scope():
            if embed_cutoffs is not None and embed_div_val != 1:
                self.embedding = AdaptiveEmbedding(vocab_size=vocab_size, embed_size=embed_size,
                                                   units=units, cutoffs=embed_cutoffs,
                                                   div_val=embed_div_val,
                                                   project_same_dim=project_same_dim)
                self.crit = AdaptiveLogSoftmaxWithLoss(vocab_size=vocab_size, embed_size=embed_size,
                                                       units=units, cutoffs=embed_cutoffs,
                                                       div_val=embed_div_val,
                                                       project_same_dim=project_same_dim,
                                                       tie_embeddings=tie_input_output_embeddings,
                                                       tie_projections=tie_input_output_projections,
                                                       params=self.embedding.collect_params())
            else:
                self.embedding = ProjectedEmbedding(vocab_size=vocab_size, embed_size=embed_size,
                                                    units=units, project_same_dim=project_same_dim)
                self.crit = ProjectedLogSoftmaxWithLoss(
                    vocab_size=vocab_size, embed_size=embed_size, units=units,
                    project_same_dim=project_same_dim, tie_embeddings=tie_input_output_embeddings,
                    tie_projections=tie_input_output_projections[0]
                    if tie_input_output_projections is not None else None,
                    params=self.embedding.collect_params())

            self.pos_emb = PositionalEmbedding(embed_size)
            if dropout:
                self.dropout_layer = nn.Dropout(rate=dropout)

            self.transformer_cells = nn.HybridSequential()
            for i in range(num_layers):
                attention_cell = PositionalEmbeddingMultiHeadAttentionCell(
                    d_head=units // num_heads, num_heads=num_heads, scaled=scaled,
                    dropout=attention_dropout)
                self.transformer_cells.add(
                    TransformerXLCell(attention_cell=attention_cell, units=units,
                                      hidden_size=hidden_size, num_heads=num_heads,
                                      weight_initializer=weight_initializer,
                                      bias_initializer=bias_initializer, dropout=dropout,
                                      scaled=scaled, use_residual=use_residual,
                                      output_attention=output_attention,
                                      prefix='transformer%d_' % i))

    def hybrid_forward(self, F, step_input, target, mask, pos_seq, mems):  #pylint: disable=arguments-differ
        """

        Parameters
        ----------
        step_input : NDArray or Symbol
            Input of shape [batch_size, length]
        target : NDArray or Symbol
            Targets of shape [batch_size, length]
        mask : NDArray or Symbol
            Attention mask of shape [length + memory_length]
        pos_seq : NDArray or Symbol
            Array of [length + memory_length] created with arange(length +
            memory_length).
        mems : List of NDArray or Symbol, optional
            Optional memory from previous forward passes containing
            `num_layers` `NDArray`s or `Symbol`s each of shape [batch_size,
            memory_length, units].

        Returns
        -------
        softmax_output : NDArray or Symbol
            Negative log likelihood of targets with shape [batch_size, length]
        hids : List of NDArray or Symbol
            List containing `num_layers` `NDArray`s or `Symbol`s each of shape
            [batch_size, mem_len, units] representing the mememory states at
            the entry of each layer (does not include last_hidden).
        last_hidden

        """
        core_out = self.embedding(step_input)
        if self._clamp_len is not None and self._clamp_len >= 0:
            pos_seq = F.clip(pos_seq, a_min=0, a_max=self._clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        if self._dropout:
            core_out = self.dropout_layer(core_out)
            pos_emb = self.dropout_layer(pos_emb)

        hids = []
        for i, layer in enumerate(self.transformer_cells):
            hids.append(core_out)
            mems_i = None if mems is None else mems[i]
            # inputs, mem_value, emb, mask=None
            core_out, _ = layer(core_out, pos_emb, mems_i, mask)

        if self._dropout:
            core_out = self.dropout_layer(core_out)

        softmax_output = self.crit(core_out, target)

        return softmax_output, hids, core_out


class TransformerXL(mx.gluon.Block):
    """Structure of the Transformer-XL.

    Dai, Z., Yang, Z., Yang, Y., Cohen, W. W., Carbonell, J., Le, Q. V., &
    Salakhutdinov, R. (2019). Transformer-XL: Attentive language models beyond
    a fixed-length context. arXiv preprint arXiv:1901.02860.

    Parameters
    ----------
    attention_cell : None
        Argument reserved for later.
    vocab_size : int or None, default None
        The size of the vocabulary.
    num_layers : int
    units : int
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    tie_input_output_embeddings : boolean, default False
        If True, tie embedding parameters for all clusters between
        AdaptiveEmbedding and AdaptiveLogSoftmaxWithLoss.
    tie_input_output_projections : List[boolean] or None, default None
        If not None, tie projection parameters for the specified clusters
        between AdaptiveEmbedding and AdaptiveLogSoftmaxWithLoss. The number of
        clusters is `len(tie_input_output_projections) == len(cutoffs) + 1`.
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.

    """

    def __init__(self, *args, **kwargs):
        prefix = kwargs.pop('prefix', None)
        params = kwargs.pop('params', None)
        super().__init__(prefix=prefix, params=params)

        with self.name_scope():
            self._net = _BaseTransformerXL(*args, **kwargs)

    def begin_mems(self, batch_size, mem_len, context):
        mems = [
            mx.nd.zeros((batch_size, mem_len, self._net._units), ctx=context)
            for _ in range(len(self._net.transformer_cells))
        ]
        return mems

    def forward(self, step_input, target, mems):  # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        step_input : NDArray or Symbol
            Input of shape [batch_size, length]
        target : NDArray or Symbol
            Input of shape [batch_size, length]
        mems : List of NDArray or Symbol, optional
            Optional memory from previous forward passes containing
            `num_layers` `NDArray`s or `Symbol`s each of shape [batch_size,
            mem_len, units].

        Returns
        -------
        softmax_output : NDArray or Symbol
            Negative log likelihood of targets with shape [batch_size, length]
        mems : List of NDArray or Symbol
            List containing `num_layers` `NDArray`s or `Symbol`s each of shape
            [batch_size, mem_len, units] representing the mememory states at
            the entry of each layer.

        """
        # Uses same number of unmasked memory steps for every step
        batch_size, qlen = step_input.shape[:2]
        mlen = mems[0].shape[1] if mems is not None else 0
        klen = qlen + mlen

        all_ones = np.ones((qlen, klen), dtype=step_input.dtype)
        mask = np.triu(all_ones, 1 + mlen) + np.tril(all_ones, 0)
        mask_nd = (mx.nd.from_numpy(mask, zero_copy=True) == 0).as_in_context(
            step_input.context).expand_dims(0).broadcast_axes(axis=0, size=batch_size)

        pos_seq = mx.nd.arange(start=klen, stop=-qlen, step=-1, ctx=step_input.context)

        softmax_output, hids, last_hidden = self._net(step_input, target, mask_nd, pos_seq, mems)

        # Update memory
        if mems is not None:
            new_mems = [
                # pylint: disable=invalid-sequence-index
                mx.nd.concat(mem_i, hid_i, dim=1)[:, -mem_i.shape[1]:].detach()
                for mem_i, hid_i in zip(mems, hids)
            ]
        else:
            new_mems = None

        return softmax_output, new_mems, last_hidden


class XLNetCell(TransformerXLCell):
    """XLNet Cell.

    Parameters
    ----------
    attention_cell
        Attention cell to be used.
    units : int
        Number of units for the output
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
    attention_dropout : float
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default None
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """

    def hybrid_forward(self, F, inputs, pos_emb, mem_value, mask, segments):
        #  pylint: disable=arguments-differ
        """Transformer Decoder Attention Cell.

        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (batch_size, length, C_in)
        mem_value : Symbol or NDArray
            Memory value, i.e. output of the encoder. Shape (batch_size,
            memory__length, C_in)
        pos_emb : Symbol or NDArray
            Positional embeddings. Shape (mem_length, C_in)
        seg_emb : Symbol or NDArray
            Segment embeddings. Shape (mem_length, C_in)
        mask : Symbol or NDArray
            Attention mask of shape (batch_size, length, length + mem_length)
        segments : Symbol or NDArray
            One-hot vector indicating if a query-key pair is in the same
            segment or not. Shape [batch_size, query_length, query_length +
            memory_length, 2]. `1` indicates that the pair is not in the same
            segment.

        Returns
        -------
        decoder_cell_outputs: list
            Outputs of the decoder cell. Contains:

            - outputs of the transformer decoder cell. Shape (batch_size, length, C_out)
            - additional_outputs of all the transformer decoder cell
        """
        key_value = F.concat(mem_value, inputs, dim=1)
        outputs, attention_outputs = self.attention_cell(inputs, key_value, key_value, pos_emb,
                                                         mask, segments)
        outputs = self.proj(outputs)
        if self._dropout:
            outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm(outputs)
        outputs = self.ffn(outputs)
        additional_outputs = [attention_outputs] if self._output_attention else []
        return outputs, additional_outputs


class _BaseXLNet(mx.gluon.HybridBlock):
    """
    Parameters
    ----------
    vocab_size : int
        The size of the vocabulary.
    num_layers : int
    units : int
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    num_heads : int
        Number of heads in multi-head attention
    activation
        Activation function used for the position-wise feed-forward networks
    two_stream
        If True, use Two-Stream Self-Attention. Typically set to True for
        pre-training and False during finetuning.
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
    attention_dropout : float
    use_residual : bool
    clamp_len : int
        Clamp all relative distances larger than clamp_len
    use_decoder : bool, default True
        Whether to include the decoder for language model prediction.
    tie_decoder_weight : bool, default True
        Whether to tie the decoder weight with the input embeddings
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default None
        Prefix for name of `Block`s (and name of weight if params is `None`).
    params : ParameterDict or None
        Container for weight sharing between cells. Created if `None`.

    """
    def __init__(self, vocab_size, num_layers=2, units=128, hidden_size=2048, num_heads=4,
                 activation='approx_gelu', two_stream: bool = False, scaled=True, dropout=0.0,
                 attention_dropout=0.0, use_residual=True, clamp_len: typing.Optional[int] = None,
                 use_decoder=True, tie_decoder_weight=True, weight_initializer=None,
                 bias_initializer='zeros', prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        assert units % num_heads == 0, 'In TransformerDecoder, the units should be divided ' \
                                       'exactly by the number of heads. Received units={}, ' \
                                       'num_heads={}'.format(units, num_heads)

        self._num_layers = num_layers
        self._units = units
        self._hidden_size = hidden_size
        self._num_heads = num_heads
        self._two_stream = two_stream
        assert not two_stream, 'Not yet implemented.'
        self._dropout = dropout
        self._use_residual = use_residual
        self._clamp_len = clamp_len
        with self.name_scope():
            self.word_embed = nn.Embedding(vocab_size, units)
            self.mask_embed = self.params.get('mask_embed', shape=(1, 1, units))
            self.pos_embed = PositionalEmbedding(units)
            if dropout:
                self.dropout_layer = nn.Dropout(rate=dropout)

            self.transformer_cells = nn.HybridSequential()
            for i in range(num_layers):
                attention_cell = RelativeSegmentEmbeddingPositionalEmbeddingMultiHeadAttentionCell(
                    d_head=units // num_heads, num_heads=num_heads, scaled=scaled,
                    dropout=attention_dropout)
                self.transformer_cells.add(
                    XLNetCell(attention_cell=attention_cell, units=units, hidden_size=hidden_size,
                              num_heads=num_heads, activation=activation,
                              weight_initializer=weight_initializer,
                              bias_initializer=bias_initializer, dropout=dropout, scaled=scaled,
                              use_residual=use_residual, prefix='transformer%d_' % i))
            if use_decoder:
                self.decoder = nn.Dense(
                    vocab_size, flatten=False,
                    params=self.word_embed.params if tie_decoder_weight else None)

    def hybrid_forward(self, F, step_input, segments, mask, pos_seq, mems, mask_embed):  #pylint: disable=arguments-differ
        """
        Parameters
        ----------
        step_input : Symbol or NDArray
            Input of shape [batch_size, query_length]
        segments : Symbol or NDArray
            One-hot vector indicating if a query-key pair is in the same
            segment or not. Shape [batch_size, query_length, query_length +
            memory_length, 2]. `1` indicates that the pair is not in the same
            segment.
        mask : Symbol or NDArray
            Attention mask of shape (batch_size, length, length + mem_length)
        pos_seq : Symbol or NDArray
            Relative distances
        mems : List of NDArray or Symbol, optional
            Memory from previous forward passes containing
            `num_layers` `NDArray`s or `Symbol`s each of shape [batch_size,
            memory_length, units].

        Returns
        -------
        core_out : NDArray or Symbol
            For use_decoder=True, logits. Otherwise output of last layer.
        hids : List of NDArray or Symbol
            Stacking the output of each layer
        """
        if self._clamp_len:
            pos_seq = F.clip(pos_seq, a_min=0, a_max=self._clamp_len)

        # Force use mask_embed in a noop to make HybridBlock happy
        core_out = F.broadcast_add(self.word_embed(step_input), 0 * mask_embed)
        pos_emb = self.pos_embed(pos_seq)

        if self._dropout:
            core_out = self.dropout_layer(core_out)
            pos_emb = self.dropout_layer(pos_emb)

        hids = []
        for i, layer in enumerate(self.transformer_cells):
            hids.append(core_out)
            mems_i = None if mems is None else mems[i]
            core_out, _ = layer(core_out, pos_emb, mems_i, mask, segments)

        if self._dropout:
            core_out = self.dropout_layer(core_out)

        if hasattr(self, 'decoder'):
            return self.decoder(core_out), hids
        return core_out, hids

    def begin_mems(self, batch_size, mem_len, context):
        mems = [
            mx.nd.zeros((batch_size, mem_len, self._units), ctx=context)
            for _ in range(len(self.transformer_cells))
        ]
        return mems


class XLNet((mx.gluon.Block)):
    """XLNet

    Yang, Z., Dai, Z., Yang, Y., Carbonell, J., Salakhutdinov, R., & Le, Q. V.
    (2019). XLNet: Generalized Autoregressive Pretraining for Language
    Understanding. arXiv preprint arXiv:1906.08237.

    Parameters
    ----------
    attention_cell : None
        Argument reserved for later.
    vocab_size : int or None, default None
        The size of the vocabulary.
    num_layers : int
    units : int
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    num_heads : int
        Number of heads in multi-head attention
    activation
        Activation function used for the position-wise feed-forward networks
    two_stream
        If True, use Two-Stream Self-Attention. Typically set to True for
        pre-training and False during finetuning.
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
    use_residual : bool
    use_decoder : bool, default True
        Whether to include the decoder for language model prediction.
    tie_decoder_weight : bool, default True
        Whether to tie the decoder weight with the input embeddings
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells. Created if `None`.

    """

    def __init__(self, *args, **kwargs):
        prefix = kwargs.pop('prefix', None)
        params = kwargs.pop('params', None)
        super().__init__(prefix=prefix, params=params)

        with self.name_scope():
            self._net = _BaseXLNet(*args, **kwargs)

    def begin_mems(self, batch_size, mem_len, context):
        mems = [
            mx.nd.zeros((batch_size, mem_len, self._net._units), ctx=context)
            for _ in range(len(self._net.transformer_cells))
        ]
        return mems

    def forward(self, step_input, token_types, mems=None, mask=None):  # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        step_input : NDArray or Symbol
            Input of shape [batch_size, query_length]
        token_types : NDArray or Symbol
            Token types of the input tokens of shape [batch_size,
            query_length], indicating various portions of the inputs.
        mems : List of NDArray or Symbol, optional
            Optional memory from previous forward passes containing
            `num_layers` `NDArray`s or `Symbol`s each of shape [batch_size,
            memory_length, units].
        mask : Symbol or NDArray
            Attention mask of shape (batch_size, length, length + mem_length)

        Returns
        -------
        output : NDArray or Symbol
            For XLNet(..., use_decoder=True), logits. Otherwise output of last
            XLNetCell layer.
        mems : List of NDArray or Symbol
            List containing `num_layers` `NDArray`s or `Symbol`s each of shape
            [batch_size, mem_len, units] representing the mememory states at
            the entry of each layer.

        """
        # Uses same number of unmasked memory steps for every step
        batch_size, qlen = step_input.shape[:2]
        mlen = mems[0].shape[1] if mems is not None else 0
        klen = qlen + mlen

        if token_types is not None:
            if mlen > 0:
                mem_pad = mx.nd.zeros([batch_size, mlen], dtype=token_types.dtype,
                                      ctx=token_types.context)
                mem_pad_token_types = mx.nd.concat(mem_pad, token_types, dim=1)

            # `1` indicates not in the same segment [qlen x klen x bsz]
            segments = mx.nd.broadcast_not_equal(token_types.expand_dims(2),
                                                 mem_pad_token_types.expand_dims(1))
            segments = mx.nd.one_hot(segments, 2, 1, 0)
        else:
            segments = None

        pos_seq = mx.nd.arange(start=klen, stop=-qlen, step=-1, ctx=step_input.context)

        if mask is None and self._net._active:
            # Hybridized _net does not support `None`-valued parameters
            mask = mx.nd.ones((batch_size, qlen, klen), ctx=step_input.context)
        output, hids = self._net(step_input, segments, mask, pos_seq, mems)

        # Update memory
        if mems is not None:
            new_mems = [
                # pylint: disable=invalid-sequence-index
                mx.nd.concat(mem_i, hid_i, dim=1)[:, -mem_i.shape[1]:].detach()
                for mem_i, hid_i in zip(mems, hids)
            ]
        else:
            new_mems = None

        return output, new_mems
