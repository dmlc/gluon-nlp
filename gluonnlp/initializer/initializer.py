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

# pylint: disable=
"""Highway layer initializer."""
from __future__ import absolute_import, print_function

__all__ = ['HighwayBias']

import mxnet
from mxnet.initializer import Initializer


@mxnet.initializer.register
class HighwayBias(Initializer):
    r"""Initialize all biases of an Highway layer by setting the biases
    of nonlinear transformer and the transform gate differently.
    The dimension of the biases are identical and equals to the :math:`arr.shape[0]/2`,
    where :math:`arr` is the bias tensor.

    The definition of the biases follows the work::

        @inproceedings{srivastava2015training,
         title={Training very deep networks},
         author={Srivastava, Rupesh K and Greff, Klaus and Schmidhuber, J{\"u}rgen},
         booktitle={Advances in neural information processing systems},
         pages={2377--2385},
         year={2015}
        }

    Parameters
    ----------
    nonlinear_transform_bias: float, default 0.0
        bias for the non linear transformer.
        We set the default according to the above original work.
    transform_gate_bias: float, default -2.0
        bias for the transform gate.
        We set the default according to the above original work.
    """
    def __init__(self, nonlinear_transform_bias=0.0, transform_gate_bias=-2.0, **kwargs):
        super(HighwayBias, self).__init__(**kwargs)
        self.nonlinear_transform_bias = nonlinear_transform_bias
        self.transform_gate_bias = transform_gate_bias

    def _init_weight(self, name, arr):
        # pylint: disable=unused-argument
        """Abstract method to Initialize weight."""
        arr[:int(arr.shape[0] / 2)] = self.nonlinear_transform_bias
        arr[int(arr.shape[0] / 2):] = self.transform_gate_bias
