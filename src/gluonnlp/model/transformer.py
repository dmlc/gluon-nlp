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
# pylint: disable=too-many-lines
"""Encoder and decoder usded in sequence-to-sequence learning."""

__all__ = ['TransformerEncoder', 'PositionwiseFFN', 'TransformerEncoderCell',
           'transformer_en_de_512']

import math
import os

import numpy as np
import mxnet as mx
from mxnet import cpu, gluon
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock
from mxnet.gluon.model_zoo import model_store

from ..base import get_home_dir
from ..utils.parallel import Parallelizable
from .block import GELU
from .seq2seq_encoder_decoder import (Seq2SeqDecoder, Seq2SeqEncoder,
                                      Seq2SeqOneStepDecoder)
from .translation import NMTModel
from .utils import _load_pretrained_params, _load_vocab
from .attention_cell import _get_attention_cell

def _position_encoding_init(max_length, dim):
    """Init the sinusoid position encoding table """
    position_enc = np.arange(max_length).reshape((-1, 1)) \
                   / (np.power(10000, (2. / dim) * np.arange(dim).reshape((1, -1))))
    # Apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    return position_enc


###############################################################################
#                                ENCODER                                      #
###############################################################################

class PositionwiseFFN(HybridBlock):
    """Positionwise Feed-Forward Neural Network.

    Parameters
    ----------
    units : int
        Number of units for the output
    hidden_size : int
        Number of units in the hidden layer of position-wise feed-forward networks
    dropout : float
        Dropout probability for the output
    use_residual : bool
        Add residual connection between the input and the output
    ffn1_dropout : bool, default False
        If True, apply dropout both after the first and second Positionwise
        Feed-Forward Neural Network layers. If False, only apply dropout after
        the second.
    activation : str, default 'relu'
        Activation function
    layer_norm_eps : float, default 1e-5
        Epsilon parameter passed to for mxnet.gluon.nn.LayerNorm
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

    def __init__(self, *, units=512, hidden_size=2048, dropout=0.0, use_residual=True,
                 ffn1_dropout=False, activation='relu', layer_norm_eps=1e-5,
                 weight_initializer=None, bias_initializer='zeros', prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self._use_residual = use_residual
        self._dropout = dropout
        self._ffn1_dropout = ffn1_dropout
        with self.name_scope():
            self.ffn_1 = nn.Dense(units=hidden_size, flatten=False,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  prefix='ffn_1_')
            self.activation = self._get_activation(activation) if activation else None
            self.ffn_2 = nn.Dense(units=units, flatten=False,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  prefix='ffn_2_')
            if dropout:
                self.dropout_layer = nn.Dropout(rate=dropout)
            self.layer_norm = nn.LayerNorm(in_channels=units, epsilon=layer_norm_eps)

    def _get_activation(self, act):
        """Get activation block based on the name. """
        if isinstance(act, str):
            if act.lower() == 'gelu':
                return GELU()
            elif act.lower() == 'approx_gelu':
                return GELU(approximate=True)
            else:
                return gluon.nn.Activation(act)
        assert isinstance(act, gluon.Block)
        return act

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        """Position-wise encoding of the inputs.

        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (batch_size, length, C_in)

        Returns
        -------
        outputs : Symbol or NDArray
            Shape (batch_size, length, C_out)
        """
        outputs = self.ffn_1(inputs)
        if self.activation:
            outputs = self.activation(outputs)
        if self._dropout and self._ffn1_dropout:
            outputs = self.dropout_layer(outputs)
        outputs = self.ffn_2(outputs)
        if self._dropout:
            outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm(outputs)
        return outputs


class TransformerEncoderCell(HybridBlock):
    """Structure of the Transformer Encoder Cell.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
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
    use_residual : bool
    output_attention: bool
        Whether to output the attention weights
    attention_use_bias : bool, default False
        Whether to use bias when projecting the query/key/values in the attention cell.
    attention_proj_use_bias : bool, default False
        Whether to use bias when projecting the output of the attention cell.
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default None
        Prefix for name of `Block`s. (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells. Created if `None`.
    activation : str, default None
        Activation methods in PositionwiseFFN
    layer_norm_eps : float, default 1e-5
        Epsilon for layer_norm

    Inputs:
        - **inputs** : input sequence. Shape (batch_size, length, C_in)
        - **mask** : mask for inputs. Shape (batch_size, length, length)

    Outputs:
        - **outputs**: output tensor of the transformer encoder cell.
            Shape (batch_size, length, C_out)
        - **additional_outputs**: the additional output of all the transformer encoder cell.
    """

    def __init__(self, *, attention_cell='multi_head', units=128, hidden_size=512, num_heads=4,
                 scaled=True, dropout=0.0, use_residual=True, output_attention=False,
                 attention_proj_use_bias=False, attention_use_bias=False, weight_initializer=None,
                 bias_initializer='zeros', prefix=None, params=None, activation='relu',
                 layer_norm_eps=1e-5):
        super().__init__(prefix=prefix, params=params)
        self._dropout = dropout
        self._use_residual = use_residual
        self._output_attention = output_attention
        with self.name_scope():
            if dropout:
                self.dropout_layer = nn.Dropout(rate=dropout)
            self.attention_cell = _get_attention_cell(attention_cell, units=units,
                                                      num_heads=num_heads, scaled=scaled,
                                                      dropout=dropout, use_bias=attention_use_bias)
            self.proj = nn.Dense(units=units, flatten=False, use_bias=attention_proj_use_bias,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer, prefix='proj_')
            self.ffn = PositionwiseFFN(units=units, hidden_size=hidden_size, dropout=dropout,
                                       use_residual=use_residual,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer, activation=activation,
                                       layer_norm_eps=layer_norm_eps)
            self.layer_norm = nn.LayerNorm(in_channels=units, epsilon=layer_norm_eps)


    def hybrid_forward(self, F, inputs, mask=None):  # pylint: disable=arguments-differ
        """Transformer Encoder Attention Cell.

        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (batch_size, length, C_in)
        mask : Symbol or NDArray or None
            Mask for inputs. Shape (batch_size, length, length)

        Returns
        -------
        encoder_cell_outputs: list
            Outputs of the encoder cell. Contains:

            - outputs of the transformer encoder cell. Shape (batch_size, length, C_out)
            - additional_outputs of all the transformer encoder cell
        """
        outputs, attention_weights = self.attention_cell(inputs, inputs, inputs, mask)
        outputs = self.proj(outputs)
        if self._dropout:
            outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm(outputs)
        outputs = self.ffn(outputs)
        additional_outputs = []
        if self._output_attention:
            additional_outputs.append(attention_weights)
        return outputs, additional_outputs

class TransformerEncoder(HybridBlock, Seq2SeqEncoder):
    """Structure of the Transformer Encoder.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
    num_layers : int
        Number of attention layers.
    units : int
        Number of units for the output.
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    max_length : int
        Maximum length of the input sequence
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    scale_embed : bool, default True
        Whether to scale the input embeddings by the sqrt of the `units`.
    norm_inputs : bool, default True
        Whether to normalize the input embeddings with LayerNorm. If dropout is
        enabled, normalization happens after dropout is applied to inputs.
    dropout : float
        Dropout probability of the attention probabilities.
    use_residual : bool
        Whether to use residual connection.
    output_attention: bool, default False
        Whether to output the attention weights
    output_all_encodings: bool, default False
        Whether to output encodings of all encoder's cells, or only the last one
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
        - **inputs** : input sequence of shape (batch_size, length, C_in)
        - **states** : list of tensors for initial states and masks.
        - **valid_length** : valid lengths of each sequence. Usually used when part of sequence
            has been padded. Shape is (batch_size, )

    Outputs:
        - **outputs** : the output of the encoder. Shape is (batch_size, length, C_out)
        - **additional_outputs** : list of tensors.
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, mem_length) or
            (batch_size, num_heads, length, mem_length)
    """

    def __init__(self, *, attention_cell='multi_head', num_layers=2, units=512, hidden_size=2048,
                 max_length=50, num_heads=4, scaled=True, scale_embed=True, norm_inputs=True,
                 dropout=0.0, use_residual=True, output_attention=False, output_all_encodings=False,
                 weight_initializer=None, bias_initializer='zeros', prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        assert units % num_heads == 0,\
            'In TransformerEncoder, The units should be divided exactly ' \
            'by the number of heads. Received units={}, num_heads={}' \
            .format(units, num_heads)
        self._max_length = max_length
        self._units = units
        self._output_attention = output_attention
        self._output_all_encodings = output_all_encodings
        self._dropout = dropout
        self._scale_embed = scale_embed
        self._norm_inputs = norm_inputs

        with self.name_scope():
            if dropout:
                self.dropout_layer = nn.Dropout(rate=dropout)
            if self._norm_inputs:
                self.layer_norm = nn.LayerNorm(in_channels=units, epsilon=1e-5)
            self.position_weight = self.params.get_constant(
                'const', _position_encoding_init(max_length, units))
            self.transformer_cells = nn.HybridSequential()
            for i in range(num_layers):
                cell = TransformerEncoderCell(
                    units=units, hidden_size=hidden_size, num_heads=num_heads,
                    attention_cell=attention_cell, weight_initializer=weight_initializer,
                    bias_initializer=bias_initializer, dropout=dropout, use_residual=use_residual,
                    scaled=scaled, output_attention=output_attention, prefix='transformer%d_' % i)
                self.transformer_cells.add(cell)

    def __call__(self, inputs, states=None, valid_length=None): #pylint: disable=arguments-differ
        """Encode the inputs given the states and valid sequence length.

        Parameters
        ----------
        inputs : NDArray or Symbol
            Input sequence. Shape (batch_size, length, C_in)
        states : list of NDArrays or Symbols
            Initial states. The list of initial states and masks
        valid_length : NDArray or Symbol
            Valid lengths of each sequence. This is usually used when part of sequence has
            been padded. Shape (batch_size,)
        Returns
        -------
        encoder_outputs: list
            Outputs of the encoder. Contains:

            - outputs of the transformer encoder. Shape (batch_size, length, C_out)
            - additional_outputs of all the transformer encoder
        """
        return super().__call__(inputs, states, valid_length)

    def hybrid_forward(self, F, inputs, states=None, valid_length=None, position_weight=None):
        # pylint: disable=arguments-differ
        """Encode the inputs given the states and valid sequence length.

        Parameters
        ----------
        inputs : NDArray or Symbol
            Input sequence. Shape (batch_size, length, C_in)
        states : list of NDArrays or Symbols
            Initial states. The list of initial states and masks
        valid_length : NDArray or Symbol
            Valid lengths of each sequence. This is usually used when part of sequence has
            been padded. Shape (batch_size,)
        position_weight : NDArray or Symbol
            The weight of positional encoding. Shape (max_len, C_in).

        Returns
        -------
        outputs : NDArray or Symbol, or List[NDArray] or List[Symbol]
            If output_all_encodings flag is False, then the output of the last encoder.
            If output_all_encodings flag is True, then the list of all outputs of all encoders.
            In both cases, shape of the tensor(s) is/are (batch_size, length, C_out)
        additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, length) or
            (batch_size, num_heads, length, length)

        """
        steps = F.contrib.arange_like(inputs, axis=1)
        if valid_length is not None:
            ones = F.ones_like(steps)
            mask = F.broadcast_lesser(F.reshape(steps, shape=(1, -1)),
                                      F.reshape(valid_length, shape=(-1, 1)))
            mask = F.broadcast_mul(F.expand_dims(mask, axis=1),
                                   F.broadcast_mul(ones, F.reshape(ones, shape=(-1, 1))))
            if states is None:
                states = [mask]
            else:
                states.append(mask)
        else:
            mask = None

        if states is None:
            states = [steps]
        else:
            states.append(steps)

        if self._scale_embed:
            inputs = inputs * math.sqrt(self._units)
        # Positional encoding
        positional_embed = F.Embedding(steps, position_weight, self._max_length, self._units)
        inputs = F.broadcast_add(inputs, F.expand_dims(positional_embed, axis=0))

        if self._dropout:
            inputs = self.dropout_layer(inputs)

        if self._norm_inputs:
            inputs = self.layer_norm(inputs)

        all_encodings_outputs = []
        additional_outputs = []
        for cell in self.transformer_cells:
            outputs, attention_weights = cell(inputs, mask)
            inputs = outputs
            if self._output_all_encodings:
                if valid_length is not None:
                    outputs = F.SequenceMask(outputs, sequence_length=valid_length,
                                             use_sequence_length=True, axis=1)
                all_encodings_outputs.append(outputs)

            if self._output_attention:
                additional_outputs.append(attention_weights)

        if valid_length is not None and not self._output_all_encodings:
            # if self._output_all_encodings, SequenceMask is already applied above
            outputs = F.SequenceMask(outputs, sequence_length=valid_length,
                                     use_sequence_length=True, axis=1)

        if self._output_all_encodings:
            return all_encodings_outputs, additional_outputs
        return outputs, additional_outputs

###############################################################################
#                                DECODER                                      #
###############################################################################

class TransformerDecoderCell(HybridBlock):
    """Structure of the Transformer Decoder Cell.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
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
        Dropout probability.
    use_residual : bool
        Whether to use residual connection.
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
    def __init__(self, attention_cell='multi_head', units=128,
                 hidden_size=512, num_heads=4, scaled=True,
                 dropout=0.0, use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(TransformerDecoderCell, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._num_heads = num_heads
        self._dropout = dropout
        self._use_residual = use_residual
        self._output_attention = output_attention
        self._scaled = scaled
        with self.name_scope():
            if dropout:
                self.dropout_layer = nn.Dropout(rate=dropout)
            self.attention_cell_in = _get_attention_cell(attention_cell,
                                                         units=units,
                                                         num_heads=num_heads,
                                                         scaled=scaled,
                                                         dropout=dropout)
            self.attention_cell_inter = _get_attention_cell(attention_cell,
                                                            units=units,
                                                            num_heads=num_heads,
                                                            scaled=scaled,
                                                            dropout=dropout)
            self.proj_in = nn.Dense(units=units, flatten=False,
                                    use_bias=False,
                                    weight_initializer=weight_initializer,
                                    bias_initializer=bias_initializer,
                                    prefix='proj_in_')
            self.proj_inter = nn.Dense(units=units, flatten=False,
                                       use_bias=False,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer,
                                       prefix='proj_inter_')
            self.ffn = PositionwiseFFN(hidden_size=hidden_size,
                                       units=units,
                                       use_residual=use_residual,
                                       dropout=dropout,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer)

            self.layer_norm_in = nn.LayerNorm()
            self.layer_norm_inter = nn.LayerNorm()

    def hybrid_forward(self, F, inputs, mem_value, mask=None, mem_mask=None):  #pylint: disable=unused-argument
        #  pylint: disable=arguments-differ
        """Transformer Decoder Attention Cell.

        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (batch_size, length, C_in)
        mem_value : Symbol or NDArrays
            Memory value, i.e. output of the encoder. Shape (batch_size, mem_length, C_in)
        mask : Symbol or NDArray or None
            Mask for inputs. Shape (batch_size, length, length)
        mem_mask : Symbol or NDArray or None
            Mask for mem_value. Shape (batch_size, length, mem_length)

        Returns
        -------
        decoder_cell_outputs: list
            Outputs of the decoder cell. Contains:

            - outputs of the transformer decoder cell. Shape (batch_size, length, C_out)
            - additional_outputs of all the transformer decoder cell
        """
        outputs, attention_in_outputs =\
            self.attention_cell_in(inputs, inputs, inputs, mask)
        outputs = self.proj_in(outputs)
        if self._dropout:
            outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm_in(outputs)
        inputs = outputs
        outputs, attention_inter_outputs = \
            self.attention_cell_inter(inputs, mem_value, mem_value, mem_mask)
        outputs = self.proj_inter(outputs)
        if self._dropout:
            outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm_inter(outputs)
        outputs = self.ffn(outputs)
        additional_outputs = []
        if self._output_attention:
            additional_outputs.append(attention_in_outputs)
            additional_outputs.append(attention_inter_outputs)
        return outputs, additional_outputs


class _BaseTransformerDecoder(HybridBlock):
    def __init__(self, attention_cell='multi_head', num_layers=2, units=128, hidden_size=2048,
                 max_length=50, num_heads=4, scaled=True, scale_embed=True, norm_inputs=True,
                 dropout=0.0, use_residual=True, output_attention=False, weight_initializer=None,
                 bias_initializer='zeros', prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        assert units % num_heads == 0, 'In TransformerDecoder, the units should be divided ' \
                                       'exactly by the number of heads. Received units={}, ' \
                                       'num_heads={}'.format(units, num_heads)
        self._num_layers = num_layers
        self._units = units
        self._hidden_size = hidden_size
        self._num_states = num_heads
        self._max_length = max_length
        self._dropout = dropout
        self._use_residual = use_residual
        self._output_attention = output_attention
        self._scaled = scaled
        self._scale_embed = scale_embed
        self._norm_inputs = norm_inputs
        with self.name_scope():
            if dropout:
                self.dropout_layer = nn.Dropout(rate=dropout)
            if self._norm_inputs:
                self.layer_norm = nn.LayerNorm()
            encoding = _position_encoding_init(max_length, units)
            self.position_weight = self.params.get_constant('const', encoding.astype(np.float32))
            self.transformer_cells = nn.HybridSequential()
            for i in range(num_layers):
                self.transformer_cells.add(
                    TransformerDecoderCell(units=units, hidden_size=hidden_size,
                                           num_heads=num_heads, attention_cell=attention_cell,
                                           weight_initializer=weight_initializer,
                                           bias_initializer=bias_initializer, dropout=dropout,
                                           scaled=scaled, use_residual=use_residual,
                                           output_attention=output_attention,
                                           prefix='transformer%d_' % i))

    def init_state_from_encoder(self, encoder_outputs, encoder_valid_length=None):
        """Initialize the state from the encoder outputs.

        Parameters
        ----------
        encoder_outputs : list
        encoder_valid_length : NDArray or None

        Returns
        -------
        decoder_states : list
            The decoder states, includes:

            - mem_value : NDArray
            - mem_masks : NDArray or None
        """
        mem_value = encoder_outputs
        decoder_states = [mem_value]
        mem_length = mem_value.shape[1]
        if encoder_valid_length is not None:
            dtype = encoder_valid_length.dtype
            ctx = encoder_valid_length.context
            mem_masks = mx.nd.broadcast_lesser(
                mx.nd.arange(mem_length, ctx=ctx, dtype=dtype).reshape((1, -1)),
                encoder_valid_length.reshape((-1, 1)))
            decoder_states.append(mem_masks)
        else:
            decoder_states.append(None)
        return decoder_states

    def hybrid_forward(self, F, inputs, states, valid_length=None, position_weight=None):
        #pylint: disable=arguments-differ
        """Decode the decoder inputs. This function is only used for training.

        Parameters
        ----------
        inputs : NDArray, Shape (batch_size, length, C_in)
        states : list of NDArrays or None
            Initial states. The list of decoder states
        valid_length : NDArray or None
            Valid lengths of each sequence. This is usually used when part of sequence has
            been padded. Shape (batch_size,)

        Returns
        -------
        output : NDArray, Shape (batch_size, length, C_out)
        states : list
            The decoder states:
            - mem_value : NDArray
            - mem_masks : NDArray or None
        additional_outputs : list of list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, mem_length) or
            (batch_size, num_heads, length, mem_length)
        """

        length_array = F.contrib.arange_like(inputs, axis=1)
        mask = F.broadcast_lesser_equal(length_array.reshape((1, -1)),
                                        length_array.reshape((-1, 1)))
        if valid_length is not None:
            batch_mask = F.broadcast_lesser(length_array.reshape((1, -1)),
                                            valid_length.reshape((-1, 1)))
            batch_mask = F.expand_dims(batch_mask, -1)
            mask = F.broadcast_mul(batch_mask, F.expand_dims(mask, 0))
        else:
            mask = F.expand_dims(mask, axis=0)
            mask = F.broadcast_like(mask, inputs, lhs_axes=(0, ), rhs_axes=(0, ))

        mem_value, mem_mask = states
        if mem_mask is not None:
            mem_mask = F.expand_dims(mem_mask, axis=1)
            mem_mask = F.broadcast_like(mem_mask, inputs, lhs_axes=(1, ), rhs_axes=(1, ))

        if self._scale_embed:
            inputs = inputs * math.sqrt(self._units)

        # Positional Encoding
        steps = F.contrib.arange_like(inputs, axis=1)
        positional_embed = F.Embedding(steps, position_weight, self._max_length, self._units)
        inputs = F.broadcast_add(inputs, F.expand_dims(positional_embed, axis=0))

        if self._dropout:
            inputs = self.dropout_layer(inputs)

        if self._norm_inputs:
            inputs = self.layer_norm(inputs)

        additional_outputs = []
        attention_weights_l = []
        outputs = inputs
        for cell in self.transformer_cells:
            outputs, attention_weights = cell(outputs, mem_value, mask, mem_mask)
            if self._output_attention:
                attention_weights_l.append(attention_weights)
        if self._output_attention:
            additional_outputs.extend(attention_weights_l)

        if valid_length is not None:
            outputs = F.SequenceMask(outputs, sequence_length=valid_length,
                                     use_sequence_length=True, axis=1)
        return outputs, states, additional_outputs


class TransformerDecoder(_BaseTransformerDecoder, Seq2SeqDecoder):
    """Transformer Decoder.

    Multi-step ahead decoder for use during training with teacher forcing.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
    num_layers : int
        Number of attention layers.
    units : int
        Number of units for the output.
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    max_length : int
        Maximum length of the input sequence. This is used for constructing position encoding
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    scale_embed : bool, default True
        Whether to scale the input embeddings by the sqrt of the `units`.
    norm_inputs : bool, default True
        Whether to normalize the input embeddings with LayerNorm. If dropout is
        enabled, normalization happens after dropout is applied to inputs.
    dropout : float
        Dropout probability.
    use_residual : bool
        Whether to use residual connection.
    output_attention: bool
        Whether to output the attention weights
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


class TransformerOneStepDecoder(_BaseTransformerDecoder, Seq2SeqOneStepDecoder):
    """Transformer Decoder.

    One-step ahead decoder for use during inference.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
    num_layers : int
        Number of attention layers.
    units : int
        Number of units for the output.
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    max_length : int
        Maximum length of the input sequence. This is used for constructing position encoding
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    scale_embed : bool, default True
        Whether to scale the input embeddings by the sqrt of the `units`.
    norm_inputs : bool, default True
        Whether to normalize the input embeddings with LayerNorm. If dropout is
        enabled, normalization happens after dropout is applied to inputs.
    dropout : float
        Dropout probability.
    use_residual : bool
        Whether to use residual connection.
    output_attention: bool
        Whether to output the attention weights
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

    def forward(self, step_input, states):  # pylint: disable=arguments-differ
        # We implement forward, as the number of states changes between the
        # first and later calls of the one-step ahead Transformer decoder. This
        # is due to the lack of numpy shape semantics. Once we enable numpy
        # shape semantic in the GluonNLP code-base, the number of states should
        # stay constant, but the first state element will be an array of shape
        # (batch_size, 0, C_in) at the first call.
        if len(states) == 3:  # step_input from prior call is included
            last_embeds, _, _ = states
            inputs = mx.nd.concat(last_embeds, mx.nd.expand_dims(step_input, axis=1), dim=1)
            states = states[1:]
        else:
            inputs = mx.nd.expand_dims(step_input, axis=1)
        return super().forward(inputs, states)

    def hybrid_forward(self, F, inputs, states, position_weight):
        # pylint: disable=arguments-differ
        """One-step-ahead decoding of the Transformer decoder.

        Parameters
        ----------
        step_input : NDArray, Shape (batch_size, C_in)
        states : list of NDArray

        Returns
        -------
        step_output : NDArray
            The output of the decoder. Shape is (batch_size, C_out)
        new_states: list
            Includes
            - last_embeds : NDArray or None
            - mem_value : NDArray
            - mem_masks : NDArray, optional

        step_additional_outputs : list of list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, mem_length) or
            (batch_size, num_heads, length, mem_length)
        """
        outputs, states, additional_outputs = super().hybrid_forward(
            F, inputs, states, valid_length=None, position_weight=position_weight)

        # Append inputs to states: They are needed in the next one-step ahead decoding step
        new_states = [inputs] + states
        # Only return one-step ahead
        step_output = F.slice_axis(outputs, axis=1, begin=-1, end=None).reshape((0, -1))

        return step_output, new_states, additional_outputs



###############################################################################
#                                  MODEL API                                  #
###############################################################################

model_store._model_sha1.update(
    {name: checksum for checksum, name in [
        ('e25287c5a924b7025e08d626f02626d5fa3af2d1', 'transformer_en_de_512_WMT2014'),
    ]})

def get_transformer_encoder_decoder(num_layers=2,
                                    num_heads=8, scaled=True,
                                    units=512, hidden_size=2048, dropout=0.0, use_residual=True,
                                    max_src_length=50, max_tgt_length=50,
                                    weight_initializer=None, bias_initializer='zeros',
                                    prefix='transformer_', params=None):
    """Build a pair of Parallel Transformer encoder/decoder

    Parameters
    ----------
    num_layers : int
    num_heads : int
    scaled : bool
    units : int
    hidden_size : int
    dropout : float
    use_residual : bool
    max_src_length : int
    max_tgt_length : int
    weight_initializer : mx.init.Initializer or None
    bias_initializer : mx.init.Initializer or None
    prefix : str, default 'transformer_'
        Prefix for name of `Block`s.
    params : Parameter or None
        Container for weight sharing between layers.
        Created if `None`.

    Returns
    -------
    encoder : TransformerEncoder
    decoder : TransformerDecoder
    one_step_ahead_decoder : TransformerOneStepDecoder
    """
    encoder = TransformerEncoder(
        num_layers=num_layers, num_heads=num_heads, max_length=max_src_length, units=units,
        hidden_size=hidden_size, dropout=dropout, scaled=scaled, use_residual=use_residual,
        weight_initializer=weight_initializer, bias_initializer=bias_initializer,
        prefix=prefix + 'enc_', params=params)
    decoder = TransformerDecoder(
        num_layers=num_layers, num_heads=num_heads, max_length=max_tgt_length, units=units,
        hidden_size=hidden_size, dropout=dropout, scaled=scaled, use_residual=use_residual,
        weight_initializer=weight_initializer, bias_initializer=bias_initializer,
        prefix=prefix + 'dec_', params=params)
    one_step_ahead_decoder = TransformerOneStepDecoder(
        num_layers=num_layers, num_heads=num_heads, max_length=max_tgt_length, units=units,
        hidden_size=hidden_size, dropout=dropout, scaled=scaled, use_residual=use_residual,
        weight_initializer=weight_initializer, bias_initializer=bias_initializer,
        prefix=prefix + 'dec_', params=decoder.collect_params())
    return encoder, decoder, one_step_ahead_decoder


def _get_transformer_model(model_cls, model_name, dataset_name, src_vocab, tgt_vocab, encoder,
                           decoder, one_step_ahead_decoder, share_embed, embed_size, tie_weights,
                           embed_initializer, pretrained, ctx, root, **kwargs):
    src_vocab = _load_vocab(dataset_name + '_src', src_vocab, root)
    tgt_vocab = _load_vocab(dataset_name + '_tgt', tgt_vocab, root)
    kwargs['encoder'] = encoder
    kwargs['decoder'] = decoder
    kwargs['one_step_ahead_decoder'] = one_step_ahead_decoder
    kwargs['src_vocab'] = src_vocab
    kwargs['tgt_vocab'] = tgt_vocab
    kwargs['share_embed'] = share_embed
    kwargs['embed_size'] = embed_size
    kwargs['tie_weights'] = tie_weights
    kwargs['embed_initializer'] = embed_initializer
    # XXX the existing model is trained with prefix 'transformer_'
    net = model_cls(prefix='transformer_', **kwargs)
    if pretrained:
        _load_pretrained_params(net, model_name, dataset_name, root, ctx)
    return net, src_vocab, tgt_vocab

transformer_en_de_hparams = {
        'num_units': 512,
        'hidden_size': 2048,
        'dropout': 0.1,
        'epsilon': 0.1,
        'num_layers': 6,
        'num_heads': 8,
        'scaled': True,
        'share_embed': True,
        'embed_size': 512,
        'tie_weights': True,
        'embed_initializer': None
}

def transformer_en_de_512(dataset_name=None, src_vocab=None, tgt_vocab=None, pretrained=False,
                          ctx=cpu(), root=os.path.join(get_home_dir(), 'models'),
                          hparam_allow_override=False, **kwargs):
    r"""Transformer pretrained model.

    Embedding size is 400, and hidden layer size is 1150.

    Parameters
    ----------
    dataset_name : str or None, default None
    src_vocab : gluonnlp.Vocab or None, default None
    tgt_vocab : gluonnlp.Vocab or None, default None
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
        MXNET_HOME defaults to '~/.mxnet'.
    hparam_allow_override : bool, default False
        If set to True, pre-defined hyper-parameters of the model
        (e.g. the number of layers, hidden units) can be overriden.

    Returns
    -------
    gluon.Block, gluonnlp.Vocab, gluonnlp.Vocab
    """
    predefined_args = transformer_en_de_hparams.copy()
    if not hparam_allow_override:
        mutable_args = frozenset(['num_units', 'hidden_size', 'dropout', 'epsilon', 'num_layers',
                                  'num_heads', 'scaled'])
        assert all((k not in kwargs or k in mutable_args) for k in predefined_args), \
            'Cannot override predefined model settings.'
    predefined_args.update(kwargs)
    encoder, decoder, one_step_ahead_decoder = get_transformer_encoder_decoder(
        units=predefined_args['num_units'], hidden_size=predefined_args['hidden_size'],
        dropout=predefined_args['dropout'], num_layers=predefined_args['num_layers'],
        num_heads=predefined_args['num_heads'], max_src_length=530, max_tgt_length=549,
        scaled=predefined_args['scaled'])
    return _get_transformer_model(NMTModel, 'transformer_en_de_512', dataset_name, src_vocab,
                                  tgt_vocab, encoder, decoder, one_step_ahead_decoder,
                                  predefined_args['share_embed'], predefined_args['embed_size'],
                                  predefined_args['tie_weights'],
                                  predefined_args['embed_initializer'], pretrained, ctx, root)


class ParallelTransformer(Parallelizable):
    """Data parallel transformer.

    Parameters
    ----------
    model : Block
        The transformer model.
    label_smoothing: Block
        The block to perform label smoothing.
    loss_function : Block
        The loss function to optimizer.
    rescale_loss : float
        The scale to which the loss is rescaled to avoid gradient explosion.
    """
    def __init__(self, model, label_smoothing, loss_function, rescale_loss):
        self._model = model
        self._label_smoothing = label_smoothing
        self._loss = loss_function
        self._rescale_loss = rescale_loss

    def forward_backward(self, x):
        """Perform forward and backward computation for a batch of src seq and dst seq"""
        (src_seq, tgt_seq, src_valid_length, tgt_valid_length), batch_size = x
        with mx.autograd.record():
            out, _ = self._model(src_seq, tgt_seq[:, :-1],
                                 src_valid_length, tgt_valid_length - 1)
            smoothed_label = self._label_smoothing(tgt_seq[:, 1:])
            ls = self._loss(out, smoothed_label, tgt_valid_length - 1).sum()
            ls = (ls * (tgt_seq.shape[1] - 1)) / batch_size / self._rescale_loss
        ls.backward()
        return ls
