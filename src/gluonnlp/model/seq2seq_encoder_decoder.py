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
__all__ = ['Seq2SeqEncoder']

from functools import partial
import mxnet as mx
from mxnet.gluon import rnn
from mxnet.gluon.block import Block
from gluonnlp.model import AttentionCell, MLPAttentionCell, DotProductAttentionCell, \
    MultiHeadAttentionCell


def _get_cell_type(cell_type):
    """Get the object type of the cell by parsing the input

    Parameters
    ----------
    cell_type : str or type

    Returns
    -------
    cell_constructor: type
        The constructor of the RNNCell
    """
    if isinstance(cell_type, str):
        if cell_type == 'lstm':
            return rnn.LSTMCell
        elif cell_type == 'gru':
            return rnn.GRUCell
        elif cell_type == 'relu_rnn':
            return partial(rnn.RNNCell, activation='relu')
        elif cell_type == 'tanh_rnn':
            return partial(rnn.RNNCell, activation='tanh')
        else:
            raise NotImplementedError
    else:
        return cell_type


def _get_attention_cell(attention_cell, units=None,
                        scaled=True, num_heads=None,
                        use_bias=False, dropout=0.0):
    """

    Parameters
    ----------
    attention_cell : AttentionCell or str
    units : int or None

    Returns
    -------
    attention_cell : AttentionCell
    """
    if isinstance(attention_cell, str):
        if attention_cell == 'scaled_luong':
            return DotProductAttentionCell(units=units, scaled=True, normalized=False,
                                           use_bias=use_bias, dropout=dropout, luong_style=True)
        elif attention_cell == 'scaled_dot':
            return DotProductAttentionCell(units=units, scaled=True, normalized=False,
                                           use_bias=use_bias, dropout=dropout, luong_style=False)
        elif attention_cell == 'dot':
            return DotProductAttentionCell(units=units, scaled=False, normalized=False,
                                           use_bias=use_bias, dropout=dropout, luong_style=False)
        elif attention_cell == 'cosine':
            return DotProductAttentionCell(units=units, scaled=False, use_bias=use_bias,
                                           dropout=dropout, normalized=True)
        elif attention_cell == 'mlp':
            return MLPAttentionCell(units=units, normalized=False)
        elif attention_cell == 'normed_mlp':
            return MLPAttentionCell(units=units, normalized=True)
        elif attention_cell == 'multi_head':
            base_cell = DotProductAttentionCell(scaled=scaled, dropout=dropout)
            return MultiHeadAttentionCell(base_cell=base_cell, query_units=units, use_bias=use_bias,
                                          key_units=units, value_units=units, num_heads=num_heads)
        else:
            raise NotImplementedError
    else:
        assert isinstance(attention_cell, AttentionCell),\
            'attention_cell must be either string or AttentionCell. Received attention_cell={}'\
                .format(attention_cell)
        return attention_cell


def _nested_sequence_last(data, valid_length):
    """

    Parameters
    ----------
    data : nested container of NDArrays/Symbols
        The input data. Each element will have shape (batch_size, ...)
    valid_length : NDArray or Symbol
        Valid length of the sequences. Shape (batch_size,)
    Returns
    -------
    data_last: nested container of NDArrays/Symbols
        The last valid element in the sequence.
    """
    assert isinstance(data, list)
    if isinstance(data[0], (mx.sym.Symbol, mx.nd.NDArray)):
        F = mx.sym if isinstance(data[0], mx.sym.Symbol) else mx.ndarray
        return F.SequenceLast(F.stack(*data, axis=0),
                              sequence_length=valid_length,
                              use_sequence_length=True)
    elif isinstance(data[0], list):
        ret = []
        for i in range(len(data[0])):
            ret.append(_nested_sequence_last([ele[i] for ele in data], valid_length))
        return ret
    else:
        raise NotImplementedError


class Seq2SeqEncoder(Block):
    r"""Base class of the encoders in sequence to sequence learning models.
    """
    def __call__(self, inputs, valid_length=None, states=None):  #pylint: disable=arguments-differ
        """Encode the input sequence.

        Parameters
        ----------
        inputs : NDArray
            The input sequence, Shape (batch_size, length, C_in).
        valid_length : NDArray or None, default None
            The valid length of the input sequence, Shape (batch_size,). This is used when the
            input sequences are padded. If set to None, all elements in the sequence are used.
        states : list of NDArrays or None, default None
            List that contains the initial states of the encoder.

        Returns
        -------
        outputs : list
            Outputs of the encoder.
        """
        return super(Seq2SeqEncoder, self).__call__(inputs, valid_length, states)

    def forward(self, inputs, valid_length=None, states=None):  #pylint: disable=arguments-differ
        raise NotImplementedError


class Seq2SeqDecoder(Block):
    r"""Base class of the decoders in sequence to sequence learning models.

    In the forward function, it generates the one-step-ahead decoding output.

    """
    def init_state_from_encoder(self, encoder_outputs, encoder_valid_length=None):
        r"""Generates the initial decoder states based on the encoder outputs.

        Parameters
        ----------
        encoder_outputs : list of NDArrays
        encoder_valid_length : NDArray or None

        Returns
        -------
        decoder_states : list
        """
        raise NotImplementedError

    def decode_seq(self, inputs, states, valid_length=None):
        r"""Given the inputs and the context computed by the encoder,
        generate the new states. This is usually used in the training phase where we set the inputs
        to be the target sequence.

        Parameters
        ----------
        inputs : NDArray
            The input embeddings. Shape (batch_size, length, C_in)
        states : list
            The initial states of the decoder.
        valid_length : NDArray or None
            valid length of the inputs. Shape (batch_size,)
        Returns
        -------
        output : NDArray
            The output of the decoder. Shape is (batch_size, length, C_out)
        states: list
            The new states of the decoder
        additional_outputs : list
            Additional outputs of the decoder, e.g, the attention weights
        """
        raise NotImplementedError

    def __call__(self, step_input, states):  #pylint: disable=arguments-differ
        r"""One-step decoding of the input

        Parameters
        ----------
        step_input : NDArray
            Shape (batch_size, C_in)
        states : list
            The previous states of the decoder
        Returns
        -------
        step_output : NDArray
            Shape (batch_size, C_out)
        states : list
        step_additional_outputs : list
            Additional outputs of the step, e.g, the attention weights
        """
        return super(Seq2SeqDecoder, self).__call__(step_input, states)

    def forward(self, step_input, states):  #pylint: disable=arguments-differ
        raise NotImplementedError
