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

"""Bidirectional LM encoder."""
__all__ = ['BiLMEncoder']

from mxnet import gluon
from mxnet.gluon import rnn
from .utils import _get_rnn_cell


class BiLMEncoder(gluon.HybridBlock):
    r"""Bidirectional LM encoder.

    We implement the encoder of the biLM proposed in the following work::

        @inproceedings{Peters:2018,
        author={Peters, Matthew E. and  Neumann, Mark and Iyyer, Mohit and Gardner, Matt and Clark,
        Christopher and Lee, Kenton and Zettlemoyer, Luke},
        title={Deep contextualized word representations},
        booktitle={Proc. of NAACL},
        year={2018}
        }

    Parameters
    ----------
    mode : str
        The type of RNN cell to use. Options are 'lstmpc', 'rnn_tanh', 'rnn_relu', 'lstm', 'gru'.
    num_layers : int
        The number of RNN cells in the encoder.
    input_size : int
        The initial input size of in the RNN cell.
    hidden_size : int
        The hidden size of the RNN cell.
    dropout : float
        The dropout rate to use for encoder output.
    skip_connection : bool
        Whether to add skip connections (add RNN cell input to output)
    proj_size : int
        The projection size of each LSTMPCellWithClip cell
    cell_clip : float
        Clip cell state between [-cellclip, cell_clip] in LSTMPCellWithClip cell
    proj_clip : float
        Clip projection between [-projclip, projclip] in LSTMPCellWithClip cell
    """
    def __init__(self, mode, num_layers, input_size, hidden_size, dropout=0.0,
                 skip_connection=True, proj_size=None, cell_clip=None, proj_clip=None, **kwargs):
        super(BiLMEncoder, self).__init__(**kwargs)

        self._mode = mode
        self._num_layers = num_layers
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._dropout = dropout
        self._skip_connection = skip_connection
        self._proj_size = proj_size
        self._cell_clip = cell_clip
        self._proj_clip = proj_clip

        with self.name_scope():
            lstm_input_size = self._input_size
            self.forward_layers = rnn.HybridSequentialRNNCell()
            with self.forward_layers.name_scope():
                for layer_index in range(self._num_layers):
                    forward_layer = _get_rnn_cell(mode=self._mode,
                                                  num_layers=1,
                                                  input_size=lstm_input_size,
                                                  hidden_size=self._hidden_size,
                                                  dropout=0
                                                  if layer_index == num_layers - 1
                                                  else self._dropout,
                                                  weight_dropout=0,
                                                  var_drop_in=0,
                                                  var_drop_state=0,
                                                  var_drop_out=0,
                                                  skip_connection=False
                                                  if layer_index == 0
                                                  else self._skip_connection,
                                                  proj_size=self._proj_size,
                                                  cell_clip=self._cell_clip,
                                                  proj_clip=self._proj_clip)

                    self.forward_layers.add(forward_layer)
                    lstm_input_size = self._proj_size

            lstm_input_size = self._input_size
            self.backward_layers = rnn.HybridSequentialRNNCell()
            with self.backward_layers.name_scope():
                for layer_index in range(self._num_layers):
                    backward_layer = _get_rnn_cell(mode=self._mode,
                                                   num_layers=1,
                                                   input_size=lstm_input_size,
                                                   hidden_size=self._hidden_size,
                                                   dropout=0
                                                   if layer_index == num_layers - 1
                                                   else self._dropout,
                                                   weight_dropout=0,
                                                   var_drop_in=0,
                                                   var_drop_state=0,
                                                   var_drop_out=0,
                                                   skip_connection=False
                                                   if layer_index == 0
                                                   else self._skip_connection,
                                                   proj_size=self._proj_size,
                                                   cell_clip=self._cell_clip,
                                                   proj_clip=self._proj_clip)
                    self.backward_layers.add(backward_layer)
                    lstm_input_size = self._proj_size

    def begin_state(self, func, **kwargs):
        return [self.forward_layers[0][0].begin_state(func=func, **kwargs)
                for _ in range(self._num_layers)], \
               [self.backward_layers[0][0].begin_state(func=func, **kwargs)
                for _ in range(self._num_layers)]

    def hybrid_forward(self, F, inputs, states=None, mask=None):
        # pylint: disable=arguments-differ
        # pylint: disable=unused-argument
        """Defines the forward computation for cache cell. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`.

        Parameters
        ----------
        inputs : NDArray
            The input data layout='TNC'.
        states : Tuple[List[List[NDArray]]]
            The states. including:
            states[0] indicates the states used in forward layer,
            Each layer has a list of two initial tensors with
            shape (batch_size, proj_size) and (batch_size, hidden_size).
            states[1] indicates the states used in backward layer,
            Each layer has a list of two initial tensors with
            shape (batch_size, proj_size) and (batch_size, hidden_size).

        Returns
        --------
        out: NDArray
            The output data with shape (num_layers, seq_len, batch_size, 2*input_size).
        [states_forward, states_backward] : List
            Including:
            states_forward: The out states from forward layer,
            which has the same structure with *states[0]*.
            states_backward: The out states from backward layer,
            which has the same structure with *states[1]*.
        """
        states_forward, states_backward = states
        if mask is not None:
            sequence_length = mask.sum(axis=1)

        outputs_forward = []
        outputs_backward = []

        for layer_index in range(self._num_layers):
            if layer_index == 0:
                layer_inputs = inputs
            else:
                layer_inputs = outputs_forward[layer_index-1]
            output, states_forward[layer_index] = F.contrib.foreach(
                self.forward_layers[layer_index],
                layer_inputs,
                states_forward[layer_index])
            outputs_forward.append(output)

            if layer_index == 0:
                layer_inputs = inputs
            else:
                layer_inputs = outputs_backward[layer_index-1]

            if mask is not None:
                layer_inputs = F.SequenceReverse(layer_inputs,
                                                 sequence_length=sequence_length,
                                                 use_sequence_length=True, axis=0)
            else:
                layer_inputs = F.SequenceReverse(layer_inputs, axis=0)
            output, states_backward[layer_index] = F.contrib.foreach(
                self.backward_layers[layer_index],
                layer_inputs,
                states_backward[layer_index])
            if mask is not None:
                backward_out = F.SequenceReverse(output,
                                                 sequence_length=sequence_length,
                                                 use_sequence_length=True, axis=0)
            else:
                backward_out = F.SequenceReverse(output, axis=0)
            outputs_backward.append(backward_out)
        out = F.concat(*[F.stack(*outputs_forward, axis=0),
                         F.stack(*outputs_backward, axis=0)], dim=-1)

        return out, [states_forward, states_backward]
