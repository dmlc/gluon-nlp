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

"""Highway layer."""
__all__ = ['Highway']

from mxnet import gluon
from mxnet.gluon import nn

class Highway(gluon.Block):
    """
    A `Highway layer <https://arxiv.org/abs/1505.00387>`_ does a gated combination of a linear
    transformation and a non-linear transformation of its input.  :math:`y = g * x + (1 - g) *
    f(A(x))`, where :math:`A` is a linear transformation, :math:`f` is an element-wise
    non-linearity, and :math:`g` is an element-wise gate, computed as :math:`sigmoid(B(x))`.

    This module will apply a fixed number of highway layers to its input, returning the final
    result.

    Parameters
    ----------
    input_dim : ``int``
        The dimensionality of :math:`x`.  We assume the input has shape ``(batch_size,
        input_dim)``.
    num_layers : ``int``, optional (default=``1``)
        The number of highway layers to apply to the input.
    activation : ``nn.activations.Activation``, optional (default=``nn.Activation('relu')``)
        The non-linearity to use in the highway layers.
    """
    def __init__(self, ninput, nlayers=1, activation='relu', **kwargs):
        super(Highway, self).__init__(**kwargs)
        self._ninputs = ninput
        self._nlayers = nlayers
        with self.name_scope():
            self._hnet = nn.HybridSequential()
            with self._hnet.name_scope():
                for i in range(self._nlayers):
                    # pylint: disable=unused-argument
                    print(i)
                    hlayer = nn.Dense(self._ninputs * 2, in_units=self._ninputs)
                    self._hnet.add(hlayer)
            self._activation = nn.Activation(activation)

    def set_bias(self):
        for layer in self._hnet:
            layer.bias.data()[self._ninputs:] = 1

    def forward(self, inputs):  # pylint: disable=arguments-differ
        current_input = inputs
        for layer in enumerate(self._hnet):
            projected_input = layer(current_input)
            linear_part = current_input
            nonlinear_part = projected_input[:, (0 * self._ninputs):(1 * self._ninputs)]
            gate = projected_input[:, (1 * self._ninputs):(2 * self._ninputs)]
            nonlinear_part = self._activation(nonlinear_part)
            gate = gate.sigmoid()
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input
