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
from gluonnlp import optimizer

# BERT ADAM
class PyBERTAdam(mx.optimizer.Optimizer):
    """python reference implemenation of BERT style adam"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 wd=0, **kwargs):
        super(PyBERTAdam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.wd = wd

    def create_state(self, index, weight):
        """Create additional optimizer state: mean, variance

        Parameters
        ----------
        weight : NDArray
        The weight data
        """
        return (mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype))  # variance

    def update(self, index, weight, grad, state):
        """Update the parameters.

        Parameters
        ----------
        index : int
        An unique integer key used to index the parameters
        weight : NDArray
        weight ndarray
        grad : NDArray
        grad ndarray
        state : NDArray or other objects returned by init_state
        The auxiliary state used in optimization.
        """
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)
        mean, variance = state
        grad = grad * self.rescale_grad
        # clip gradients
        if self.clip_gradient is not None:
            mx.nd.clip(grad, -self.clip_gradient, self.clip_gradient, out=grad)
        # update mean
        mean[:] = self.beta1 * mean + (1. - self.beta1) * grad
        # update variance
        variance[:] = self.beta2 * variance + (1 - self.beta2) * grad.square()
        # include weight decay
        update = mean / (mx.nd.sqrt(variance) + self.epsilon) + wd * weight
        # update weight
        weight -= lr * update


def test_bert_adam():
    opt1 = PyBERTAdam
    opt2 = optimizer.BERTAdam
    shape = (3, 4, 5)
    cg_options = [{}, {'clip_gradient': 0.4}, {'clip_gradient': 0.5}]
    rg_options = [{}, {'rescale_grad': 0.14}, {'rescale_grad': 0.8}]
    wd_options = [{}, {'wd': 0.03}, {'wd': 0.05}, {'wd': 0.07}]
    for dtype in [np.float16, np.float32, np.float64]:
        for cg_option in cg_options:
            for rg_option in rg_options:
                for wd_option in wd_options:
                    kwarg = {}
                    kwarg.update(cg_option)
                    kwarg.update(rg_option)
                    kwarg.update(wd_option)
                    mx.test_utils.compare_optimizer(opt1(**kwarg), opt2(**kwarg), shape, dtype,
                                                    rtol=1e-4, atol=2e-5)
