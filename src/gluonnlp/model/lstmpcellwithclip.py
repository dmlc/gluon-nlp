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
"""LSTM projection cell with cell clip and projection clip."""
__all__ = ['LSTMPCellWithClip']

from mxnet.gluon.contrib.rnn import LSTMPCell


class LSTMPCellWithClip(LSTMPCell):
    r"""Long-Short Term Memory Projected (LSTMP) network cell with cell clip and projection clip.
    Each call computes the following function:

    .. math::

    \begin{array}{ll}
    i_t = sigmoid(W_{ii} x_t + b_{ii} + W_{ri} r_{(t-1)} + b_{ri}) \\
    f_t = sigmoid(W_{if} x_t + b_{if} + W_{rf} r_{(t-1)} + b_{rf}) \\
    g_t = \tanh(W_{ig} x_t + b_{ig} + W_{rc} r_{(t-1)} + b_{rg}}) \\
    o_t = sigmoid(W_{io} x_t + b_{io} + W_{ro} r_{(t-1)} + b_{ro}) \\
    c_t = c_clip(f_t * c_{(t-1)} + i_t * g_t) \\
    h_t = o_t * \tanh(c_t) \\
    r_t = p_clip(W_{hr} h_t)
    \end{array}
    where :math:`c_clip` is the cell clip applied on the next cell;
    :math:`r_t` is the projected recurrent activation at time `t`,
    :math:`p_clip` means apply projection clip on he projected output.
    math:`h_t` is the hidden state at time `t`, :math:`c_t` is the
    cell state at time `t`, :math:`x_t` is the input at time `t`, and :math:`i_t`,
    :math:`f_t`, :math:`g_t`, :math:`o_t` are the input, forget, cell, and
    out gates, respectively.

    Parameters
    ----------
    hidden_size : int
        Number of units in cell state symbol.
    projection_size : int
        Number of units in output symbol.
    i2h_weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    h2h_weight_initializer : str or Initializer
        Initializer for the recurrent weights matrix, used for the linear
        transformation of the hidden state.
    h2r_weight_initializer : str or Initializer
        Initializer for the projection weights matrix, used for the linear
        transformation of the recurrent state.
    i2h_bias_initializer : str or Initializer, default 'lstmbias'
        Initializer for the bias vector. By default, bias for the forget
        gate is initialized to 1 while all other biases are initialized
        to zero.
    h2h_bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    cell_clip : float
        Clip cell state between [-cellclip, cell_clip] in LSTMPCellWithClip cell
    projection_clip : float
        Clip projection between [-projection_clip, projection_clip] in LSTMPCellWithClip cell
    """
    def __init__(self, hidden_size, projection_size,
                 i2h_weight_initializer=None, h2h_weight_initializer=None,
                 h2r_weight_initializer=None,
                 i2h_bias_initializer='zeros', h2h_bias_initializer='zeros',
                 input_size=0, cell_clip=None, projection_clip=None, prefix=None, params=None):
        super(LSTMPCellWithClip, self).__init__(hidden_size,
                                                projection_size,
                                                i2h_weight_initializer,
                                                h2h_weight_initializer,
                                                h2r_weight_initializer,
                                                i2h_bias_initializer,
                                                h2h_bias_initializer,
                                                input_size,
                                                prefix=prefix,
                                                params=params)

        self._cell_clip = cell_clip
        self._projection_clip = projection_clip

    # pylint: disable= arguments-differ
    def hybrid_forward(self, F, inputs, states, i2h_weight,
                       h2h_weight, h2r_weight, i2h_bias, h2h_bias):
        r"""Hybrid forward computation for Long-Short Term Memory Projected network cell
        with cell clip and projection clip.

        Parameters
        ----------
        inputs : input tensor with shape `(batch_size, input_size)`.
        states : a list of two initial recurrent state tensors, with shape
            `(batch_size, projection_size)` and `(batch_size, hidden_size)` respectively.

        Returns
        --------
        out : output tensor with shape `(batch_size, num_hidden)`.
        next_states : a list of two output recurrent state tensors. Each has
            the same shape as `states`.
        """
        prefix = 't%d_'%self._counter
        i2h = F.FullyConnected(data=inputs, weight=i2h_weight, bias=i2h_bias,
                               num_hidden=self._hidden_size*4, name=prefix+'i2h')
        h2h = F.FullyConnected(data=states[0], weight=h2h_weight, bias=h2h_bias,
                               num_hidden=self._hidden_size*4, name=prefix+'h2h')
        gates = i2h + h2h
        slice_gates = F.SliceChannel(gates, num_outputs=4, name=prefix+'slice')
        in_gate = F.Activation(slice_gates[0], act_type='sigmoid', name=prefix+'i')
        forget_gate = F.Activation(slice_gates[1], act_type='sigmoid', name=prefix+'f')
        in_transform = F.Activation(slice_gates[2], act_type='tanh', name=prefix+'c')
        out_gate = F.Activation(slice_gates[3], act_type='sigmoid', name=prefix+'o')
        next_c = F._internal._plus(forget_gate * states[1], in_gate * in_transform,
                                   name=prefix+'state')
        if self._cell_clip is not None:
            next_c = next_c.clip(-self._cell_clip, self._cell_clip)
        hidden = F._internal._mul(out_gate, F.Activation(next_c, act_type='tanh'),
                                  name=prefix+'hidden')
        next_r = F.FullyConnected(data=hidden, num_hidden=self._projection_size,
                                  weight=h2r_weight, no_bias=True, name=prefix+'out')
        if self._projection_clip is not None:
            next_r = next_r.clip(-self._projection_clip, self._projection_clip)

        return next_r, [next_r, next_c]
