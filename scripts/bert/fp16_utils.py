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
import warnings
import collections
import numpy as np
import mxnet as mx
from mxnet import nd

def grad_global_norm(parameters, max_norm):
    """Calculate the 2-norm of gradients of parameters, and how much they should be scaled down
    such that their 2-norm does not exceed `max_norm`.

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
        norm, ratio = grad_global_norm(net.collect_params().values(), max_norm)
        trainer.update(batch_size * ratio)
        ...

    Parameters
    ----------
    parameters : list of Parameters

    Returns
    -------
    NDArray
      Total norm. Shape is (1,)
    NDArray
      Ratio for rescaling gradients based on max_norm s.t. grad = grad / ratio.
      If total norm is NaN, ratio will be NaN, too. Shape is (1,)
    NDArray
      Whether the total norm is finite. Shape is (1,)
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
        # TODO(haibin) norm operator does not support fp16 safe reduction.
        # Issue is tracked at: https://github.com/apache/incubator-mxnet/issues/14126
        x = array.reshape((-1,)).astype('float32', copy=False)
        return nd.dot(x, x)

    norm_arrays = [_norm(arr) for arr in arrays]

    # group norm arrays by ctx
    def group_by_ctx(arr_list):
        groups = collections.defaultdict(list)
        for arr in arr_list:
            ctx = arr.context
            groups[ctx].append(arr)
        return groups
    norm_groups = group_by_ctx(norm_arrays)

    # reduce
    ctx, dtype = arrays[0].context, 'float32'
    norms = [nd.add_n(*g).as_in_context(ctx) for g in norm_groups.values()]
    total_norm = nd.add_n(*norms).sqrt()
    scale = total_norm / max_norm
    # is_finite = 0 if NaN or Inf, 1 otherwise.
    is_finite = nd.contrib.isfinite(scale)
    # if scale is finite, nd.maximum selects the max between scale and 1. That is,
    # 1 is returned if total_norm does not exceed max_norm.
    # if scale = NaN or Inf, the result of nd.minimum is undefined. Therefore, we use
    # choices.take to return NaN or Inf.
    scale_or_one = nd.maximum(nd.ones((1,), dtype=dtype, ctx=ctx), scale)
    choices = nd.concat(scale, scale_or_one, dim=0)
    chosen_scale = choices.take(is_finite)
    return total_norm, chosen_scale, is_finite


class FP16Trainer(object):
    """ Trainer for mixed precision training.

    Parameters
    ----------
    trainer: gluon.Trainer
      the original gluon Trainer object for fp32 training.
    dynamic_loss_scale: bool. Default is True
      whether to use dynamic loss scaling. This is recommended for optimizing model
      parameters using FP16.
    loss_scaler_params : dict
        Key-word arguments to be passed to loss scaler constructor. For example,
        `{"init_scale" : 2.**15, "scale_window" : 2000, "tolerance" : 0.05}`
        for `DynamicLossScaler`.
        See each `LossScaler` for a list of supported arguments'
    """
    def __init__(self, trainer, dynamic_loss_scale=True, loss_scaler_params=None):
        if trainer._kvstore_params['update_on_kvstore'] is not False and trainer._kvstore:
            err = 'Only gluon.Trainer created with update_on_kvstore=False is supported.'
            raise NotImplementedError(err)
        self.fp32_trainer = trainer
        loss_scaler_params = loss_scaler_params if loss_scaler_params else {}
        self._scaler = DynamicLossScaler(**loss_scaler_params) if dynamic_loss_scale \
                       else StaticLossScaler(**loss_scaler_params)
        # if the optimizer supports NaN check, we can always defer the NaN check to the optimizer
        # TODO(haibin) this should be added via registry
        self._support_nan_check = trainer._optimizer.__class__.__name__ == 'BERTAdam'

    def backward(self, loss):
        """backward propagation with loss"""
        with mx.autograd.record():
            if isinstance(loss, (tuple, list)):
                ls = [l * self._scaler.loss_scale for l in loss]
            else:
                ls = loss * self._scaler.loss_scale
        mx.autograd.backward(ls)

    def step(self, batch_size, max_norm=None):
        """Makes one step of parameter update. Should be called after
        `fp16_optimizer.backward()`, and outside of `record()` scope.

        Parameters
        ----------
        batch_size : int
            Batch size of data processed. Gradient will be normalized by `1/batch_size`.
            Set this to 1 if you normalized loss manually with `loss = mean(loss)`.
        max_norm : NDArray, optional, default is None
            max value for global 2-norm of gradients.
        """
        self.fp32_trainer.allreduce_grads()
        step_size = batch_size * self._scaler.loss_scale
        if max_norm:
            norm, ratio, is_finite = grad_global_norm(self.fp32_trainer._params,
                                                      max_norm * self._scaler.loss_scale)
            step_size = ratio * step_size
            if self._support_nan_check:
                self.fp32_trainer.update(step_size)
                overflow = is_finite.asscalar() < 1
            else:
                overflow = not np.isfinite(norm.asscalar())
                if not overflow:
                    self.fp32_trainer.update(step_size)
        else:
            # TODO(haibin) optimize the performance when max_norm is not present
            # sequentially adding isnan/isinf results may be slow
            if self._support_nan_check:
                self.fp32_trainer.update(step_size)
                overflow = self._scaler.has_overflow(self.fp32_trainer._params)
            else:
                overflow = self._scaler.has_overflow(self.fp32_trainer._params)
                if not overflow:
                    self.fp32_trainer.update(step_size)
        # update scale based on overflow information
        self._scaler.update_scale(overflow)

class LossScaler(object):
    """Abstract loss scaler"""
    def has_overflow(self, params):
        """ detect inf and nan """
        is_not_finite = 0
        for param in params:
            if param.grad_req != 'null':
                grad = param.list_grad()[0]
                is_not_finite += mx.nd.contrib.isnan(grad).sum()
                is_not_finite += mx.nd.contrib.isinf(grad).sum()
        # NDArray is implicitly converted to bool
        if is_not_finite == 0:
            return False
        else:
            return True

    def update_scale(self, overflow):
        raise NotImplementedError()

class StaticLossScaler(LossScaler):
    """Static loss scaler"""
    def __init__(self, init_scale=1):
        self.loss_scale = init_scale

    def update_scale(self, overflow):
        """update loss scale"""
        pass

class DynamicLossScaler(LossScaler):
    """Class that manages dynamic loss scaling.

    There are two problems regarding gradient scale when fp16 is used for training.
    One is overflow: the fp16 gradient is too large that it causes NaN.
    To combat such an issue, we need to scale down the gradient when such an event
    is detected. The other is underflow: the gradient is too small such that the
    precision suffers. This is hard to detect though. What dynamic loss scaler does
    it that, it starts the scale at a relatively large value (e.g. 2**15).
    Everytime when a NaN is detected in the gradient, the scale is reduced (by default)
    by 2x. On the other hand, if a NaN is not detected for a long time
    (e.g. 2000 steps), then the scale is increased (by default) by 2x."""
    def __init__(self, init_scale=2.**15, scale_factor=2., scale_window=2000,
                 tolerance=0.01):
        self.loss_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.tolerance = tolerance
        self._num_steps = 0
        self._last_overflow_iter = -1
        self._last_rescale_iter = -1
        self._overflows_since_rescale = 0

    def update_scale(self, overflow):
        """dynamically update loss scale"""
        iter_since_rescale = self._num_steps - self._last_rescale_iter
        if overflow:
            self._last_overflow_iter = self._num_steps
            self._overflows_since_rescale += 1
            percentage = self._overflows_since_rescale / float(iter_since_rescale)
            # we tolerate a certrain amount of NaNs before actually scaling it down
            if percentage >= self.tolerance:
                self.loss_scale /= self.scale_factor
                self._last_rescale_iter = self._num_steps
                self._overflows_since_rescale = 0
                if self.loss_scale < 1:
                    warnings.warn('DynamicLossScaler: overflow detected. set loss_scale = %s'%
                                  self.loss_scale)
        elif (self._num_steps - self._last_overflow_iter) % self.scale_window == 0:
            self.loss_scale *= self.scale_factor
            self._last_rescale_iter = self._num_steps
        self._num_steps += 1
