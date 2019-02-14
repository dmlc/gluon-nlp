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

"""Trainer for mixed precision training."""
import mxnet as mx
import numpy as np
import logging
from mxnet import nd
from mxnet.gluon import Trainer
from mxnet.optimizer import Optimizer, register
from mxnet.ndarray import zeros, NDArray

def grad_global_norm(parameters, max_norm):
    """Rescales gradients of parameters so that the sum of their 2-norm is smaller than `max_norm`.

    If gradients exist for more than one context for a parameter, user needs to explicitly call
    ``trainer.allreduce_grads`` so that the gradients are summed first before calculating
    the 2-norm.

    .. note::

        This function is only for use when `update_on_kvstore` is set to False in trainer.

    Example::

        trainer = Trainer(net.collect_params(), update_on_kvstore=False, ...)
        for x, y in mx.gluon.utils.split_and_load(X, [mx.gpu(0), mx.gpu(1)]):
            with mx.autograd.record():
                y = net(x)
                loss = loss_fn(y, label)
            loss.backward()
        trainer.allreduce_grads()
        norm, ratio = nlp.utils.grad_global_norm(net.collect_params().values(), max_norm)
        trainer.update(batch_size * ratio)
        ...

    Parameters
    ----------
    parameters : list of Parameters

    Returns
    -------
    NDArray
      Total norm.
    NDArray
      Ratio for rescaling gradients based on max_norm s.t. grad = grad / ratio.
      If total norm is NaN, ratio will be NaN, too.
    """
    # collect gradient arrays
    arrays = []
    idx = 0
    for p in parameters:
        if p.grad_req != 'null':
            p_grads = p.list_grad()
            arrays.append(p_grads[idx % len(p_grads)])
            idx += 1
    assert len(arrays) > 0, 'No parameter found available for gradient norm.'

    # compute gradient norms
    def _norm(array):
        if array.stype == 'default':
            # TODO(haibin) remove temporary cast to fp32.
            x = array.reshape((-1,)).astype('float32', copy=False)
            return nd.dot(x, x)
        return array.norm().square()
    norm_arrays = [_norm(arr) for arr in arrays]

    # group norm arrays by ctx
    def group_by_ctx(arr_list):
        groups = {}
        for arr in arr_list:
            ctx = arr.context
            if ctx in groups:
                groups[ctx].append(arr)
            else:
                groups[ctx] = [arr]
        return groups
    norm_groups = group_by_ctx(norm_arrays)

    # reduce
    ctx, dtype = arrays[0].context, 'float32'
    total_norm = mx.nd.zeros((1,), ctx=ctx, dtype=dtype)
    for group in norm_groups:
        total_norm += nd.add_n(*norm_groups[group]).as_in_context(ctx)
    total_norm = nd.sqrt(total_norm)
    scale = total_norm / max_norm
    # is_finite = 0 if NaN or Inf, 1 otherwise.
    is_finite = nd.contrib.isfinite(scale)
    # if scale is finite, nd.minimum selects the minimum between scale and 1. That is,
    # 1 is returned if total_norm does not exceed max_norm.
    # if scale = NaN or Inf, the result of nd.minimum is undefined. Therefore, we use
    # choices.take to return NaN or Inf.
    scale_or_one = nd.minimum(nd.ones((1,), dtype=dtype, ctx=ctx), scale)
    choices = nd.concat(scale, scale_or_one, dim=0)
    chosen_scale = choices.take(is_finite)
    return total_norm, chosen_scale


class FP16Trainer(object):
    """ Trainer for mixed precision training. """
    # TODO(haibin): inherit from gluon.Trainer
    def __init__(self, trainer, fp16=True):
        if trainer._kvstore_params['update_on_kvstore'] is not False:
            raise NotImplementedError('Only gluon.Trainer created with update_on_kvstore=False is supported.')
        self.fp32_trainer = trainer
        self._scaler = DynamicLossScaler() if fp16 else StaticLossScaler()

    def backward(self, loss):
        with mx.autograd.record():
            if isinstance(loss, (tuple, list)):
                ls = [l * self._scaler.loss_scale for l in loss]
            else:
                ls = loss * self._scaler.loss_scale
        mx.autograd.backward(ls)

    def step(self, batch_size, max_norm=None):
        self.fp32_trainer.allreduce_grads()
        step_size = batch_size * self._scaler.loss_scale
        if max_norm:
            norm, ratio = grad_global_norm(self.fp32_trainer._params,
                                           max_norm * self._scaler.loss_scale)
            step_size = ratio * step_size
            self.fp32_trainer.update(step_size)
            overflow = not np.isfinite(norm.asscalar())
        else:
            self.fp32_trainer.update(step_size)
            overflow = self._scaler.has_overflow(self.fp32_trainer._params)
        # update scale based on overflow information
        self._scaler.update_scale(overflow)
        if overflow:
            for param in self.fp32_trainer._params:
                param.zero_grad()

class LossScaler(object):
    def has_overflow(self, params):
        """ detect inf and nan """
        is_not_finite = 0
        for param in params:
            if param.grad_req != 'null':
                grad = param.list_grad()[0]
                is_not_finite += mx.nd.contrib.isnan(grad).sum()
                is_not_finite += mx.nd.contrib.isinf(grad).sum()
        if is_not_finite == 0:
            return False
        else:
            return True

class StaticLossScaler(LossScaler):
    def __init__(self, init_scale=1):
        self.loss_scale = init_scale

    def update_scale(self, overflow):
        pass

class DynamicLossScaler(object):
    def __init__(self, init_scale=2.**15, scale_factor=2., scale_window=2000,
                 tolerance=0.05, verbose=False):
        self.loss_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.tolerance = tolerance
        self._iter = 0
        self._last_overflow_iter = -1
        self._last_rescale_iter = -1
        self._overflows_since_rescale = 0
        self._verbose = verbose

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
            if self._verbose:
                logging.info('overflow detected. set loss_scale = %s'%(self.loss_scale))
        elif (self._iter - self._last_overflow_iter) % self.scale_window == 0:
            self.loss_scale *= self.scale_factor
            self._last_rescale_iter = self._iter
        self._iter += 1
