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

from __future__ import absolute_import
from __future__ import print_function

__all__ = ['Highway']

from mxnet import gluon
from mxnet.gluon import nn
from gluonnlp.initializer import HighwayBias


class Highway(gluon.HybridBlock):
    r"""Highway network.

    We implemented the highway network proposed in the following work::

        @article{srivastava2015highway,
          title={Highway networks},
          author={Srivastava, Rupesh Kumar and Greff, Klaus and Schmidhuber, J{\"u}rgen},
          journal={arXiv preprint arXiv:1505.00387},
          year={2015}
        }

    The full version of the work::

        @inproceedings{srivastava2015training,
         title={Training very deep networks},
         author={Srivastava, Rupesh K and Greff, Klaus and Schmidhuber, J{\"u}rgen},
         booktitle={Advances in neural information processing systems},
         pages={2377--2385},
         year={2015}
        }

    A Highway layer is defined as below:

    .. math::
        y = (1 - t) * x + t * f(A(x))

    which is a gated combination of a linear transform and a non-linear transform of its input,
    where :math:`x` is the input tensor, :math:`A` is a linear transformer,
    :math:`f` is an element-wise non-linear transformer,
    and :math:`t` is an element-wise transform gate, and :math:`1-t` refers to carry gate.

    Parameters
    ----------
    input_size : int
        The dimension of the input tensor.  We assume the input has shape ``(batch_size,
        input_size)``.
    num_layers : int
        The number of highway layers to apply to the input.
    activation : str, default 'relu'
        The non-linear activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    highway_bias : HighwayBias,
        default HighwayBias(nonlinear_transform_bias=0.0, transform_gate_bias=-2.0)
        The biases applied to the highway layer.
        We set the default according to the above original work.
    """

    def __init__(self,
                 input_size,
                 num_layers,
                 activation='relu',
                 highway_bias=HighwayBias(nonlinear_transform_bias=0.0, transform_gate_bias=-2.0),
                 **kwargs):
        super(Highway, self).__init__(**kwargs)
        self._input_size = input_size
        self._num_layers = num_layers

        with self.name_scope():
            self.hnet = nn.HybridSequential()
            with self.hnet.name_scope():
                for _ in range(self._num_layers):
                    self.hnet.add(nn.Dense(units=self._input_size * 2,
                                           in_units=self._input_size,
                                           bias_initializer=highway_bias,
                                           use_bias=True,
                                           flatten=False))
            self._activation = nn.Activation(activation)

    def hybrid_forward(self, F, inputs, **kwargs):
        # pylint: disable=unused-argument
        r"""
        Forward computation for highway layer

        Parameters
        ----------
        inputs: NDArray
            The input tensor is of shape `(..., input_size)`.

        Returns
        ----------
        outputs: NDArray
            The output tensor is of the same shape with input tensor `(..., input_size)`.
        """
        current_input = inputs
        for layer in self.hnet:
            projected_input = layer(current_input)
            linear_transform = current_input
            nonlinear_transform, transform_gate = projected_input.split(num_outputs=2, axis=-1)
            nonlinear_transform = self._activation(nonlinear_transform)
            transform_gate = transform_gate.sigmoid()
            current_input = (1 - transform_gate) * linear_transform + \
                            transform_gate * nonlinear_transform
        return current_input
