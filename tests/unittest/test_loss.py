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

import gluonnlp as nlp
import mxnet as mx
from mxnet import gluon
import numpy as np

def testActivationRegularizationLoss():
    ar = nlp.loss.ActivationRegularizationLoss(2)
    print(ar)
    ar(*[mx.nd.arange(1000).reshape(10, 10, 10),
         mx.nd.arange(1000).reshape(10, 10, 10)])

def testTemporalActivationRegularizationLoss():
    tar = nlp.loss.TemporalActivationRegularizationLoss(1)
    print(tar)
    tar(*[mx.nd.arange(1000).reshape(10, 10, 10),
          mx.nd.arange(1000).reshape(10, 10, 10)])

def testMaskedSoftmaxCrossEntropyLoss():
    loss_fn = nlp.loss.MaskedSoftmaxCELoss()
    pred = mx.nd.array([[[0,0,10],[10,0,0]]]) #N,T,C 1,2,3
    label = mx.nd.array([[2,2]])
    valid_length = mx.nd.array([1,])
    loss = loss_fn(pred, label, valid_length)
    assert loss < 0.1, "1st timestep prediction is correct, but loss was high"
    valid_length = mx.nd.array([2,])
    loss = loss_fn(pred, label, valid_length)
    assert loss > 1, "2nd timestep prediction was wrong, but loss did not go up"

def testLabelSmoothing():
    # Testing that the label gets smoothed at the right location
    sparse_labels = [0,1,2]
    for epsilon, units in zip([0.1, 0.3, 0.5], [5, 10, 20]):
        smoother = nlp.loss.LabelSmoothing(epsilon=epsilon, units=units)
        smoothed_labels = smoother(mx.nd.array(sparse_labels))
        for i, label in enumerate(sparse_labels):
            for k in range(units):
                if k == label:
                    mx.test_utils.assert_almost_equal(
                        smoothed_labels[i,k].asnumpy(),
                        np.array([1 - epsilon/units * (units-1)])
                    )
                else:
                    mx.test_utils.assert_almost_equal(
                        smoothed_labels[i,k].asnumpy(),
                        np.array([epsilon/units])
                    )
