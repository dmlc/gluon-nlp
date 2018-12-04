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

"""Test Sampler."""
from __future__ import print_function

import mxnet as mx
import numpy as np
from numpy.testing import assert_allclose
from ..language_model.sampler import LogUniformSampler


def test_log_uniform_sampler():
    ntokens = 793472
    num_sampled = 8192
    sampler = LogUniformSampler(ntokens, num_sampled)
    true_cls = mx.nd.array([5, 10, 20])
    sample, cnt_sample, cnt_true = sampler(true_cls)
    assert np.unique(sample.asnumpy()).size == num_sampled
    assert cnt_true.size == true_cls.size
    assert cnt_sample.size == cnt_sample.size
