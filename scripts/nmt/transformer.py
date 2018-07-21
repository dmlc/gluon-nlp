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
"""Encoder and decoder usded in sequence-to-sequence learning."""
__all__ = ['TransformerEncoder', 'TransformerDecoder', 'get_transformer_encoder_decoder']

import math
import numpy as np
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock
try:
    from encoder_decoder import Seq2SeqEncoder, Seq2SeqDecoder, _get_attention_cell
except ImportError:
    from .encoder_decoder import Seq2SeqEncoder, Seq2SeqDecoder, _get_attention_cell


def _position_encoding_init(max_length, dim):
    """ Init the sinusoid position encoding table """
    position_enc = np.arange(max_length).reshape((-1, 1)) \
                   / (np.power(10000, (2. / dim) * np.arange(dim).reshape((1, -1))))
    # Apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    return position_enc


class PositionwiseFFN(HybridBlock):
    """Structure of the Positionwise Feed-Forward Neural Network.

    Parameters
    ----------
    units : int
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    dropout : float
    use_residual : bool
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    activation : str, default 'relu'
        Activation function
    prefix : str, default 'rnn_'
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, units=512, hidden_size=2048, dropout=0.0, use_residual=True,
                 weight_initializer=None, bias_initializer='zeros', activation='relu',
                 prefix=None, params=None):
        super(PositionwiseFFN, self).__init__(prefix=prefix, params=params)
        self._hidden_size = hidden_size
        self._units = units
        self._use_residual = use_residual
        with self.name_scope():
            self.ffn_1 = nn.Dense(units=hidden_size, flatten=False,
                                  activation=activation,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  prefix='ffn_1_')
            self.ffn_2 = nn.Dense(units=units, flatten=False,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer,
                                  prefix='ffn_2_')
            self.dropout_layer = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm()

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
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
        outputs = self.ffn_2(outputs)
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
    def __init__(self, attention_cell='multi_head', units=128,
                 hidden_size=512, num_heads=4, scaled=True,
                 dropout=0.0, use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(TransformerEncoderCell, self).__init__(prefix=prefix, params=params)
        self._units = units
        self._num_heads = num_heads
        self._dropout = dropout
        self._use_residual = use_residual
        self._output_attention = output_attention
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.attention_cell = _get_attention_cell(attention_cell,
                                                      units=units,
                                                      num_heads=num_heads,
                                                      scaled=scaled,
                                                      dropout=dropout)
            self.proj = nn.Dense(units=units, flatten=False, use_bias=False,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer,
                                 prefix='proj_')
            self.ffn = PositionwiseFFN(hidden_size=hidden_size, units=units,
                                       use_residual=use_residual, dropout=dropout,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer)
            self.layer_norm = nn.LayerNorm()

    def hybrid_forward(self, F, inputs, mask=None):  # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
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
        outputs, attention_weights =\
            self.attention_cell(inputs, inputs, inputs, mask)
        outputs = self.proj(outputs)
        outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm(outputs)
        outputs = self.ffn(outputs)
        additional_outputs = []
        if self._output_attention:
            additional_outputs.append(attention_weights)
        return outputs, additional_outputs


class TransformerDecoderCell(HybridBlock):
    """Structure of the Transformer Decoder Cell.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
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
            self.dropout_layer = nn.Dropout(dropout)
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
        outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        outputs = self.layer_norm_in(outputs)
        inputs = outputs
        outputs, attention_inter_outputs = \
            self.attention_cell_inter(inputs, mem_value, mem_value, mem_mask)
        outputs = self.proj_inter(outputs)
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


class TransformerEncoder(HybridBlock, Seq2SeqEncoder):
    """Structure of the Transformer Encoder.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
    num_layers : int
    units : int
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    max_length : int
        Maximum length of the input sequence
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
    use_residual : bool
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
    def __init__(self, attention_cell='multi_head', num_layers=2,
                 units=512, hidden_size=2048, max_length=50,
                 num_heads=4, scaled=True, dropout=0.0,
                 use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(TransformerEncoder, self).__init__(prefix=prefix, params=params)
        assert units % num_heads == 0,\
            'In TransformerEncoder, The units should be divided exactly ' \
            'by the number of heads. Received units={}, num_heads={}' \
            .format(units, num_heads)
        self._num_layers = num_layers
        self._max_length = max_length
        self._num_heads = num_heads
        self._units = units
        self._hidden_size = hidden_size
        self._output_attention = output_attention
        self._dropout = dropout
        self._use_residual = use_residual
        self._scaled = scaled
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm()
            self.position_weight = self.params.get_constant('const',
                                                            _position_encoding_init(max_length,
                                                                                    units))
            self.transformer_cells = nn.HybridSequential()
            for i in range(num_layers):
                self.transformer_cells.add(
                    TransformerEncoderCell(
                        units=units,
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        attention_cell=attention_cell,
                        weight_initializer=weight_initializer,
                        bias_initializer=bias_initializer,
                        dropout=dropout,
                        use_residual=use_residual,
                        scaled=scaled,
                        output_attention=output_attention,
                        prefix='transformer%d_' % i))

    def __call__(self, inputs, states=None, valid_length=None): #pylint: disable=arguments-differ
        """Encoder the inputs given the states and valid sequence length.

        Parameters
        ----------
        inputs : NDArray
            Input sequence. Shape (batch_size, length, C_in)
        states : list of NDArrays or None
            Initial states. The list of initial states and masks
        valid_length : NDArray or None
            Valid lengths of each sequence. This is usually used when part of sequence has
            been padded. Shape (batch_size,)

        Returns
        -------
        encoder_outputs: list
            Outputs of the encoder. Contains:

            - outputs of the transformer encoder. Shape (batch_size, length, C_out)
            - additional_outputs of all the transformer encoder
        """
        return super(TransformerEncoder, self).__call__(inputs, states, valid_length)

    def forward(self, inputs, states=None, valid_length=None, steps=None): # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        inputs : NDArray, Shape(batch_size, length, C_in)
        states : list of NDArray
        valid_length : NDArray
        steps : NDArray
            Stores value [0, 1, ..., length].
            It is used for lookup in positional encoding matrix

        Returns
        -------
        outputs : NDArray
            The output of the encoder. Shape is (batch_size, length, C_out)
        additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, length) or
            (batch_size, num_heads, length, length)

        """
        length = inputs.shape[1]
        if valid_length is not None:
            mask = mx.nd.broadcast_lesser(
                mx.nd.arange(length, ctx=valid_length.context).reshape((1, -1)),
                valid_length.reshape((-1, 1)))
            mask = mx.nd.broadcast_axes(mx.nd.expand_dims(mask, axis=1), axis=1, size=length)
            if states is None:
                states = [mask]
            else:
                states.append(mask)
        inputs = inputs * math.sqrt(inputs.shape[-1])
        steps = mx.nd.arange(length, ctx=inputs.context)
        if states is None:
            states = [steps]
        else:
            states.append(steps)
        if valid_length is not None:
            step_output, additional_outputs =\
                super(TransformerEncoder, self).forward(inputs, states, valid_length)
        else:
            step_output, additional_outputs =\
                super(TransformerEncoder, self).forward(inputs, states)
        return step_output, additional_outputs

    def hybrid_forward(self, F, inputs, states=None, valid_length=None, position_weight=None): # pylint: disable=arguments-differ
        """

        Parameters
        ----------
        inputs : NDArray or Symbol, Shape(batch_size, length, C_in)
        states : list of NDArray or Symbol
        valid_length : NDArray or Symbol
        position_weight : NDArray or Symbol

        Returns
        -------
        outputs : NDArray or Symbol
            The output of the encoder. Shape is (batch_size, length, C_out)
        additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, length) or
            (batch_size, num_heads, length, length)

        """
        if states is not None:
            steps = states[-1]
            # Positional Encoding
            inputs = F.broadcast_add(inputs, F.expand_dims(F.Embedding(steps, position_weight,
                                                                       self._max_length,
                                                                       self._units), axis=0))
        inputs = self.dropout_layer(inputs)
        inputs = self.layer_norm(inputs)
        outputs = inputs
        if valid_length is not None:
            mask = states[-2]
        else:
            mask = None
        additional_outputs = []
        for cell in self.transformer_cells:
            outputs, attention_weights = cell(inputs, mask)
            inputs = outputs
            if self._output_attention:
                additional_outputs.append(attention_weights)
        if valid_length is not None:
            outputs = F.SequenceMask(outputs, sequence_length=valid_length,
                                     use_sequence_length=True, axis=1)
        return outputs, additional_outputs


class TransformerDecoder(HybridBlock, Seq2SeqDecoder):
    """Structure of the Transformer Decoder.

    Parameters
    ----------
    attention_cell : AttentionCell or str, default 'multi_head'
        Arguments of the attention cell.
        Can be 'multi_head', 'scaled_luong', 'scaled_dot', 'dot', 'cosine', 'normed_mlp', 'mlp'
    num_layers : int
    units : int
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    max_length : int
        Maximum length of the input sequence. This is used for constructing position encoding
    num_heads : int
        Number of heads in multi-head attention
    scaled : bool
        Whether to scale the softmax input by the sqrt of the input dimension
        in multi-head attention
    dropout : float
    use_residual : bool
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
    def __init__(self, attention_cell='multi_head', num_layers=2,
                 units=128, hidden_size=2048, max_length=50,
                 num_heads=4, scaled=True, dropout=0.0,
                 use_residual=True, output_attention=False,
                 weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(TransformerDecoder, self).__init__(prefix=prefix, params=params)
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
        with self.name_scope():
            self.dropout_layer = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm()
            self.position_weight = self.params.get_constant('const',
                                                            _position_encoding_init(max_length,
                                                                                    units))
            self.transformer_cells = nn.HybridSequential()
            for i in range(num_layers):
                self.transformer_cells.add(
                    TransformerDecoderCell(
                        units=units,
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        attention_cell=attention_cell,
                        weight_initializer=weight_initializer,
                        bias_initializer=bias_initializer,
                        dropout=dropout,
                        scaled=scaled,
                        use_residual=use_residual,
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
            - mem_masks : NDArray, optional
        """
        mem_value = encoder_outputs
        decoder_states = [mem_value]
        mem_length = mem_value.shape[1]
        if encoder_valid_length is not None:
            mem_masks = mx.nd.broadcast_lesser(
                mx.nd.arange(mem_length, ctx=encoder_valid_length.context).reshape((1, -1)),
                encoder_valid_length.reshape((-1, 1)))
            decoder_states.append(mem_masks)
        self._encoder_valid_length = encoder_valid_length
        return decoder_states

    def decode_seq(self, inputs, states, valid_length=None):
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
            The decoder states, includes:

            - mem_value : NDArray
            - mem_masks : NDArray, optional
        additional_outputs : list of list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, mem_length) or
            (batch_size, num_heads, length, mem_length)
        """
        batch_size = inputs.shape[0]
        length = inputs.shape[1]
        length_array = mx.nd.arange(length, ctx=inputs.context)
        mask = mx.nd.broadcast_lesser_equal(
            length_array.reshape((1, -1)),
            length_array.reshape((-1, 1)))
        if valid_length is not None:
            batch_mask = mx.nd.broadcast_lesser(
                mx.nd.arange(length, ctx=valid_length.context).reshape((1, -1)),
                valid_length.reshape((-1, 1)))
            mask = mx.nd.broadcast_mul(mx.nd.expand_dims(batch_mask, -1),
                                       mx.nd.expand_dims(mask, 0))
        else:
            mask = mx.nd.broadcast_axes(mx.nd.expand_dims(mask, axis=0), axis=0, size=batch_size)
        states = [None] + states
        output, states, additional_outputs = self.forward(inputs, states, mask)
        states = states[1:]
        if valid_length is not None:
            output = mx.nd.SequenceMask(output,
                                        sequence_length=valid_length,
                                        use_sequence_length=True,
                                        axis=1)
        return output, states, additional_outputs

    def __call__(self, step_input, states): #pylint: disable=arguments-differ
        """One-step-ahead decoding of the Transformer decoder.

        Parameters
        ----------
        step_input : NDArray
        states : list of NDArray

        Returns
        -------
        step_output : NDArray
            The output of the decoder.
            In the train mode, Shape is (batch_size, length, C_out)
            In the test mode, Shape is (batch_size, C_out)
        new_states: list
            Includes
            - last_embeds : NDArray or None
                It is only given during testing
            - mem_value : NDArray
            - mem_masks : NDArray, optional

        step_additional_outputs : list of list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, mem_length) or
            (batch_size, num_heads, length, mem_length)
        """
        return super(TransformerDecoder, self).__call__(step_input, states)

    def forward(self, step_input, states, mask=None):  #pylint: disable=arguments-differ, missing-docstring
        input_shape = step_input.shape
        mem_mask = None
        # If it is in testing, transform input tensor to a tensor with shape NTC
        # Otherwise remove the None in states.
        if len(input_shape) == 2:
            if self._encoder_valid_length is not None:
                has_last_embeds = len(states) == 3
            else:
                has_last_embeds = len(states) == 2
            if has_last_embeds:
                last_embeds = states[0]
                step_input = mx.nd.concat(last_embeds,
                                          mx.nd.expand_dims(step_input, axis=1),
                                          dim=1)
                states = states[1:]
            else:
                step_input = mx.nd.expand_dims(step_input, axis=1)
        elif states[0] is None:
            states = states[1:]
        has_mem_mask = (len(states) == 2)
        if has_mem_mask:
            _, mem_mask = states
            augmented_mem_mask = mx.nd.expand_dims(mem_mask, axis=1)\
                .broadcast_axes(axis=1, size=step_input.shape[1])
            states[-1] = augmented_mem_mask
        if mask is None:
            length_array = mx.nd.arange(step_input.shape[1], ctx=step_input.context)
            mask = mx.nd.broadcast_lesser_equal(
                length_array.reshape((1, -1)),
                length_array.reshape((-1, 1)))
            mask = mx.nd.broadcast_axes(mx.nd.expand_dims(mask, axis=0),
                                        axis=0, size=step_input.shape[0])
        steps = mx.nd.arange(step_input.shape[1], ctx=step_input.context)
        states.append(steps)
        step_output, step_additional_outputs = \
            super(TransformerDecoder, self).forward(step_input * math.sqrt(step_input.shape[-1]),  #pylint: disable=too-many-function-args
                                                    states, mask)
        states = states[:-1]
        if has_mem_mask:
            states[-1] = mem_mask
        new_states = [step_input] + states
        # If it is in testing, only output the last one
        if len(input_shape) == 2:
            step_output = step_output[:, -1, :]
        return step_output, new_states, step_additional_outputs

    def hybrid_forward(self, F, step_input, states, mask=None, position_weight=None):  #pylint: disable=arguments-differ
        """

        Parameters
        ----------
        step_input : NDArray or Symbol, Shape (batch_size, length, C_in)
        states : list of NDArray or Symbol
        mask : NDArray or Symbol
        position_weight : NDArray or Symbol

        Returns
        -------
        step_output : NDArray or Symbol
            The output of the decoder. Shape is (batch_size, length, C_out)
        step_additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length, mem_length) or
            (batch_size, num_heads, length, mem_length)

        """
        has_mem_mask = (len(states) == 3)
        if has_mem_mask:
            mem_value, mem_mask, steps = states
        else:
            mem_value, steps = states
            mem_mask = None
        # Positional Encoding
        step_input = F.broadcast_add(step_input,
                                     F.expand_dims(F.Embedding(steps,
                                                               position_weight,
                                                               self._max_length,
                                                               self._units),
                                                   axis=0))
        step_input = self.dropout_layer(step_input)
        step_input = self.layer_norm(step_input)
        inputs = step_input
        outputs = inputs
        step_additional_outputs = []
        attention_weights_l = []
        for cell in self.transformer_cells:
            outputs, attention_weights = cell(inputs, mem_value, mask, mem_mask)
            if self._output_attention:
                attention_weights_l.append(attention_weights)
            inputs = outputs
        if self._output_attention:
            step_additional_outputs.extend(attention_weights_l)
        return outputs, step_additional_outputs


def get_transformer_encoder_decoder(num_layers=2,
                                    num_heads=8, scaled=True,
                                    units=512, hidden_size=2048, dropout=0.0, use_residual=True,
                                    max_src_length=50, max_tgt_length=50,
                                    weight_initializer=None, bias_initializer='zeros',
                                    prefix='transformer_', params=None):
    """Build a pair of Parallel GNMT encoder/decoder

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
    decoder :TransformerDecoder
    """
    encoder = TransformerEncoder(num_layers=num_layers,
                                 num_heads=num_heads,
                                 max_length=max_src_length,
                                 units=units,
                                 hidden_size=hidden_size,
                                 dropout=dropout,
                                 scaled=scaled,
                                 use_residual=use_residual,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer,
                                 prefix=prefix + 'enc_', params=params)
    decoder = TransformerDecoder(num_layers=num_layers,
                                 num_heads=num_heads,
                                 max_length=max_tgt_length,
                                 units=units,
                                 hidden_size=hidden_size,
                                 dropout=dropout,
                                 scaled=scaled,
                                 use_residual=use_residual,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer,
                                 prefix=prefix + 'dec_', params=params)
    return encoder, decoder
