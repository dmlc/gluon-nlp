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

import mxnet as mx
from numpy.testing import assert_almost_equal

from gluonnlp.model import Highway


def test_highway_forward_simple_input():
    highway = Highway(input_size=2, num_layers=2)
    print(highway)
    highway.initialize(init='Xavier')
    highway.set_bias()
    highway.hnet[0].weight.data()[:] = 1
    highway.hnet[0].bias.data()[:] = 0
    highway.hnet[1].weight.data()[:] = 2
    highway.hnet[1].bias.data()[:] = -2
    input = mx.nd.array([[-2, 1], [3, -2]])
    output = highway(input)
    print(output)
    assert output.shape == (2, 2)
    assert_almost_equal(output.asnumpy(),
                        mx.nd.array([[-1.4177, 0.7088], [1.4764, 1.2234]]).asnumpy(),
                        decimal=4)


def test_highway_forward():
    highway = Highway(input_size=2, num_layers=2)
    print(highway)
    highway.initialize()
    highway.set_bias()
    input = mx.nd.ones((2, 3, 2))
    output = highway(input)
    print(output)
    assert output.shape == (2, 3, 2), output.shape
