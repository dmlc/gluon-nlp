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

import gluonnlp as nlp


def test_spearmanr():
    pr = nlp.metric.SpearmanRankCorrelation()

    predicts = [mx.nd.array([1, 2, 3, 4, 5])]
    labels = [mx.nd.array([5, 6, 7, 8, 7])]
    pr.update(labels, predicts)
    assert pr.get() == ('spearmanr', 0.82078268166812329)
    pr.reset()

    predicts = [mx.nd.array([1, 2, 3])]
    labels = [mx.nd.array([5, 6, 7])]
    pr.update(labels, predicts)
    predicts = [mx.nd.array([4, 5])]
    labels = [mx.nd.array([8, 7])]
    pr.update(labels, predicts)
    assert pr.get() == ('spearmanr', 0.82078268166812329)
    pr.reset()
