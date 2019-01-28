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

import pytest
from mxnet.gluon import nn
from gluonnlp import initializer


def test_truncnorm_string_alias_works():
    try:
        layer = nn.Dense(prefix="test_layer", in_units=1, units=1, weight_initializer='truncnorm')
        layer.initialize()
    except RuntimeError:
        pytest.fail('Layer couldn\'t be initialized')


def test_truncnorm_all_values_inside_boundaries():
    mean = 0
    std = 0.01
    layer = nn.Dense(prefix="test_layer", in_units=1, units=1000)
    layer.initialize(init=initializer.TruncNorm(mean, std))
    assert ((layer.weight.data() > 2 * std).sum() +
            (layer.weight.data() < -2 * std).sum()).sum().asscalar() == 0


def test_truncnorm_generates_values_with_defined_mean_and_std():
    from scipy import stats

    mean = 10
    std = 5
    layer = nn.Dense(prefix="test_layer", in_units=1, units=100000)
    layer.initialize(init=initializer.TruncNorm(mean, std))
    samples = layer.weight.data().reshape((-1, )).asnumpy()

    p_value = stats.kstest(samples, 'truncnorm', args=(-2, 2, mean, std)).pvalue
    assert p_value > 0.0001

