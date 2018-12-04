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

from __future__ import print_function

import sys

import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
import pytest

@pytest.mark.parametrize('f', [nlp.model.NCEDense, nlp.model.SparseNCEDense])
def test_nce_loss(f):
    ctx = mx.cpu()
    batch_size = 2
    num_sampled = 3
    vocab_size = 10
    num_hidden = 5
    model = f(vocab_size, num_sampled, num_hidden)
    loss = gluon.loss.SigmoidBCELoss()
    model.hybridize()
    model.initialize(mx.init.Xavier(), ctx=ctx)
    trainer = mx.gluon.Trainer(model.collect_params(), 'sgd')
    x = mx.nd.ones((batch_size, num_hidden))
    y = mx.nd.ones((batch_size,))
    sampled_cls = mx.nd.ones((num_sampled,))
    sampled_cls_cnt = mx.nd.ones((num_sampled,))
    true_cls_cnt = mx.nd.ones((batch_size,))
    samples = (sampled_cls, sampled_cls_cnt, true_cls_cnt)
    with mx.autograd.record():
        pred, new_y = model(x, samples, y)
        assert pred.shape == (batch_size, 1+num_sampled)
        assert new_y.shape == (batch_size, 1+num_sampled)
        l = loss(pred, new_y)
    l.backward()
    mx.nd.waitall()

@pytest.mark.parametrize('f', [nlp.model.ISDense, nlp.model.SparseISDense])
def test_is_softmax_loss(f):
    ctx = mx.cpu()
    batch_size = 2
    num_sampled = 3
    vocab_size = 10
    num_hidden = 5
    model = f(vocab_size, num_sampled, num_hidden)
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    model.hybridize()
    model.initialize(mx.init.Xavier(), ctx=ctx)
    trainer = mx.gluon.Trainer(model.collect_params(), 'sgd')
    x = mx.nd.ones((batch_size, num_hidden))
    y = mx.nd.ones((batch_size,))
    sampled_cls = mx.nd.ones((num_sampled,))
    sampled_cls_cnt = mx.nd.ones((num_sampled,))
    true_cls_cnt = mx.nd.ones((batch_size,))
    samples = (sampled_cls, sampled_cls_cnt, true_cls_cnt)
    with mx.autograd.record():
        pred, new_y = model(x, samples, y)
        assert pred.shape == (batch_size, 1+num_sampled)
        assert new_y.shape == (batch_size,)
        l = loss(pred, new_y)
    l.backward()
    mx.nd.waitall()
