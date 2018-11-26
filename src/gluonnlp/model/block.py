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

"""Building blocks and utility for models."""
__all__ = ['RNNCellLayer', 'L2Normalization', 'GELU']

import math
import warnings
from mxnet import ndarray
from mxnet.gluon import Block, HybridBlock


class RNNCellLayer(Block):
    """A block that takes an rnn cell and makes it act like rnn layer.

    Parameters
    ----------
    rnn_cell : Cell
        The cell to wrap into a layer-like block.
    layout : str, default 'TNC'
        The output layout of the layer.
    """
    def __init__(self, rnn_cell, layout='TNC', **kwargs):
        super(RNNCellLayer, self).__init__(**kwargs)
        self.cell = rnn_cell
        assert layout == 'TNC' or layout == 'NTC', \
            'Invalid layout %s; must be one of ["TNC" or "NTC"]'%layout
        self._layout = layout
        self._axis = layout.find('T')
        self._batch_axis = layout.find('N')

    def forward(self, inputs, states=None): # pylint: disable=arguments-differ
        """Defines the forward computation. Arguments can be either
        :py:class:`NDArray` or :py:class:`Symbol`."""
        batch_size = inputs.shape[self._batch_axis]
        skip_states = states is None
        if skip_states:
            states = self.cell.begin_state(batch_size, ctx=inputs.context)
        if isinstance(states, ndarray.NDArray):
            states = [states]
        for state, info in zip(states, self.cell.state_info(batch_size)):
            if state.shape != info['shape']:
                raise ValueError(
                    'Invalid recurrent state shape. Expecting %s, got %s.'%(
                        str(info['shape']), str(state.shape)))
        states = sum(zip(*((j for j in i) for i in states)), ())
        outputs, states = self.cell.unroll(
            inputs.shape[self._axis], inputs, states,
            layout=self._layout, merge_outputs=True)

        if skip_states:
            return outputs
        return outputs, states


class L2Normalization(HybridBlock):
    """Normalize the input array by dividing the L2 norm along the given axis.

    ..code

        out = data / (sqrt(sum(data**2, axis)) + eps)

    Parameters
    ----------
    axis : int, default -1
        The axis to compute the norm value.
    eps : float, default 1E-6
        The epsilon value to avoid dividing zero
    """
    def __init__(self, axis=-1, eps=1E-6, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)
        self._axis = axis
        self._eps = eps

    def hybrid_forward(self, F, x):  # pylint: disable=arguments-differ
        ret = F.broadcast_div(x, F.norm(x, axis=self._axis, keepdims=True) + self._eps)
        return ret

class GELU(HybridBlock):
    r"""Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    https://arxiv.org/abs/1606.08415

    Parameters
    ----------
    Inputs:
        - **data**: input tensor with arbitrary shape.
    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)
        self._support_erf = False
        try:
            self._support_erf = True if ndarray.erf else False
        except AttributeError:
            warnings.warn('`erf` operator support is not found. '
                          'Please consider upgrading to mxnet >= 1.4')

    def hybrid_forward(self, F, x):
        if self._support_erf:
            return x * 0.5 * (1.0 + F.erf(x / math.sqrt(2.0)))
        else:
            # approximate GELU if erf is not supported
            return 0.5 * x * (1 + F.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * (x ** 3))))

    def __repr__(self):
        s = '{name}()'
        return s.format(name=self.__class__.__name__)
