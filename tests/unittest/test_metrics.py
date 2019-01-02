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
import numpy as np
from gluonnlp.metric import MaskedAccuracy

def test_acc():
    pred = mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])
    label = mx.nd.array([0, 1, 1])
    mask = mx.nd.array([1, 1, 0])
    metric = MaskedAccuracy()
    metric.update([label], [pred], [mask])
    _, acc = metric.get()
    matched = (np.argmax(pred.asnumpy(), axis=1) == label.asnumpy()) * mask.asnumpy()
    valid_count = mask.asnumpy().sum()
    expected_acc = 1.0 * matched.sum() / valid_count
    assert acc == expected_acc

    metric = MaskedAccuracy()
    metric.update([label], [pred])
    _, acc = metric.get()
    matched = (np.argmax(pred.asnumpy(), axis=1) == label.asnumpy())
    valid_count = len(label)
    expected_acc = 1.0 * matched.sum() / valid_count
    assert acc == expected_acc
