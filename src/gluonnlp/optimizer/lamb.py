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

"""LAMB optimizer"""

from mxnet.optimizer import Optimizer, register
from mxnet.ndarray import zeros, NDArray
from mxnet.ndarray import square, power, sqrt, maximum, minimum, clip

__all__ = ['LAMB']


@register
class LAMB(Optimizer):
    """The LAMB optimizer: 
    It has been proposed in `Reducing BERT Pre-Training Time from 3 Days to 76 Minutes`.
    https://arxiv.org/abs/1904.00962

    Updates are applied by::
        grad = clip(grad * rescale_grad, clip_gradient)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - power(beta1, t))
        v_hat = m / (1 - power(beta2, t))
        r1 = w.norm()
        g = m_hat / (sqrt(v_hat + epsilon)) + wd * w
        r2 = g.norm()
        r = if r1 == 0. or r2 == 0. else minimum(
            maximum(r1 / r2, self.lower_bound), self.upper_bound)
        lr = r * lr
        w = w - lr * g

    Parameters
    ----------
    beta1 : float, optional, default is 0.9
        Exponential decay rate for the first moment estimates.
    beta2 : float, optional, default is 0.999
        Exponential decay rate for the second moment estimates.
    epsilon : float, optional, default is 1e-6
        Small value to avoid division by 0.
    lower_bound : float, optional, default is 1e-3
        Lower limit of lamb_trust_ratio
    upper_bound : float, optional, default is 10.0
        Upper limit of lamb_trust_ratio
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-6,
                 lower_bound=1e-3, upper_bound=10.0, lazy_update=False, **kwargs):
        super(LAMB, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.lazy_update = lazy_update

    def create_state(self, index, weight):
        stype = weight.stype if self.lazy_update else 'default'
        return (zeros(weight.shape, weight.context, dtype=weight.dtype,
                      stype=stype),  # mean
                zeros(weight.shape, weight.context, dtype=weight.dtype,
                      stype=stype))  # variance

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        t = self._index_update_count[index]

        # preprocess grad
        grad *= self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        mean, var = state
        mean[:] = self.beta1 * mean + (1. - self.beta1) * grad
        var[:] = self.beta2 * var + (1. - self.beta2) * square(grad)

        # execution bias correction
        mean_hat = mean / (1. - power(self.beta1, t))
        var_hat = var / (1. - power(self.beta2, t))

        r1 = weight.norm()
        g = mean_hat / sqrt(var_hat + self.epsilon) + wd * weight
        r2 = g.norm()

        # calculate lamb_trust_ratio
        r = 1. if r1 == 0. or r2 == 0. else minimum(
            maximum(r1 / r2, self.lower_bound), self.upper_bound)
        lr *= r

        # update weight
        weight[:] -= lr * g
