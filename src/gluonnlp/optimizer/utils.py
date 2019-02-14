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

"""Weight updating functions."""
import mxnet as mx
import numpy as np
from mxnet.gluon import Trainer
from mxnet.optimizer import Optimizer, register
from mxnet.ndarray import zeros, NDArray

class FP16Trainer(object):

    def __init__(self, trainer):
        # TODO add args for scaler
        if trainer._kvstore_params['update_on_kvstore'] is not False:
            raise NotImplementedError('Only gluon.Trainer created with update_on_kvstore=False is supported.')
        self.fp32_trainer = trainer
        self._scaler = DynamicLossScaler()#init_scale=1)

    def backward(self, loss):
        with mx.autograd.record():
            if isinstance(loss, (tuple, list)):
                ls = [l * self._scaler.loss_scale for l in loss]
            else:
                ls = loss * self._scaler.loss_scale
        mx.autograd.backward(ls)

    def step(self, batch_size, ignore_stale_grad=False, global_norm=None):
        self.fp32_trainer.allreduce_grads()
        step_size = batch_size * self._scaler.loss_scale
        if global_norm:
            from ..utils import clip_grad_global_norm, grad_global_norm
            norm, ratio = grad_global_norm(self.fp32_trainer._params, global_norm * self._scaler.loss_scale)
            step_size = ratio * step_size
        self.fp32_trainer.update(step_size,
                                 ignore_stale_grad=ignore_stale_grad)

        if global_norm:
            overflow = not np.isfinite(norm.asscalar())
        else:
            overflow = self._scaler.has_overflow(self.fp32_trainer._params)
        self._scaler.update_scale(overflow)
        if overflow:
            #raise Exception()
            #print('skip step. zero grad')
            for param in self.fp32_trainer._params:
                param.zero_grad()
        # copy FP32 params back into FP16 model
        #offset = 0
        #for p in self.params:
        #    if not p.requires_grad:
        #        continue
        #    numel = p.data.numel()
        #    p.data.copy_(self.fp32_params.data[offset:offset+numel].view_as(p.data))
        #    offset += numel

class DynamicLossScaler(object):
    def __init__(self, init_scale=2.**15, scale_factor=2., scale_window=2000, tolerance=0.05):
        self.loss_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.tolerance = tolerance
        self._iter = 0
        self._last_overflow_iter = -1
        self._last_rescale_iter = -1
        self._overflows_since_rescale = 0

    def update_scale(self, overflow):
        iter_since_rescale = self._iter - self._last_rescale_iter
        if overflow:
            self._last_overflow_iter = self._iter
            self._overflows_since_rescale += 1
            pct_overflow = self._overflows_since_rescale / float(iter_since_rescale)
            if pct_overflow >= self.tolerance:
                self.loss_scale /= self.scale_factor
                self._last_rescale_iter = self._iter
                self._overflows_since_rescale = 0
            print('overflow', self.loss_scale)
        elif (self._iter - self._last_overflow_iter) % self.scale_window == 0:
            self.loss_scale *= self.scale_factor
            self._last_rescale_iter = self._iter
            #print('no overflow', self.loss_scale)
        self._iter += 1

    def has_overflow(self, params):
        # detect inf and nan
        for param in params:
            if param.grad_req != 'null':
                grad = param.list_grad()[0]
                if mx.nd.contrib.isnan(grad).sum():
                    return True
                if mx.nd.contrib.isinf(grad).sum():
                    return True
        return False
