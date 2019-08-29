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

"""RAdam optimizer"""
import math
from mxnet.optimizer import Optimizer, register
from mxnet.ndarray import zeros, NDArray
from mxnet.ndarray import square, power, sqrt, maximum, minimum, clip

__all__ = ['RAdam']

@register
class RAdam(Optimizer):
    """The RAdam optimizer proposed in
    On The Variance Of The Adaptive Learning Rate And Beyond https://arxiv.org/abs/1908.03265
    Originl Code : https://github.com/LiyuanLucasLiu/RAdam
    Parameters
    ----------
    beta1 : float, optional, default is 0.9
        Exponential decay rate for the first moment estimates.
    beta2 : float, optional, default is 0.999
        Exponential decay rate for the second moment estimates.
    epsilon : float, optional, default is 1e-6
        Small value to avoid division by 0.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-6, **kwargs):
        super(RAdam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.buffer = [[None, None, None] for ind in range(10)]

    def create_state(self, index, weight):
        stype = weight.stype
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
        t = self._index_update_count[index] ## update count

        # preprocess grad
        grad *= self.rescale_grad
        if self.clip_gradient is not None:
            grad = clip(grad, -self.clip_gradient, self.clip_gradient)

        mean, var = state
        mean[:] = self.beta1 * mean + (1. - self.beta1) * grad
        var[:] = self.beta2 * var + (1. - self.beta2) * square(grad)
        mean_hat = mean / (1. - power(self.beta1, t))
        buffered = self.buffer[int(t % 10)]

        if t == buffered[0]:
            N_sma, step_size = buffered[1], buffered[2]
        else:
            buffered[0] = t
            beta2_t = power(self.beta2, t)
            N_sma_max = 2 / (1 - self. beta2) - 1
            N_sma = N_sma_max - 2 * t * beta2_t / (1 - beta2_t)
            buffered[1] = N_sma
            # more conservative since it's an approximated value      
            if N_sma >= 5:
                step_size = lr * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - self.beta1 ** t)
            else:
                step_size = lr / (1 - self.beta1 ** t)
            buffered[2] = step_size
        
        if wd != 0:
            weight[:] -= lr * wd * weight[:]

        # update weight
        # more conservative since it's an approximated value
        if N_sma >= 5:
            denom = sqrt(var) + self.epsilon   
            weight[:] -= step_size * mean_hat / denom
        else:
            weight[:] -= step_size * mean_hat
