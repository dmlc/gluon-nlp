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
import mxnet as mx
import gluonnlp as nlp

class FP16Trainer:
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
        `{"init_scale" : 2.**10, "scale_window" : 2000, "tolerance" : 0.05}`
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
            _, ratio, is_finite = nlp.utils.grad_global_norm(self.fp32_trainer._params,
                                                             max_norm * self._scaler.loss_scale)
            step_size = ratio * step_size
            if self._support_nan_check:
                self.fp32_trainer.update(step_size)
                overflow = is_finite.asscalar() < 1
            else:
                overflow = is_finite.asscalar() < 1
                if not overflow:
                    step_size = step_size.asscalar()
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

class LossScaler:
    """Abstract loss scaler"""
    def has_overflow(self, params):
        """ detect inf and nan """
        is_not_finite = 0
        for param in params:
            if param.grad_req != 'null':
                grad = param.list_grad()[0]
                is_not_finite += mx.nd.contrib.isnan(grad).sum().astype('float32', copy=False)
                is_not_finite += mx.nd.contrib.isinf(grad).sum().astype('float32', copy=False)
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
    def __init__(self, init_scale=2.**10, scale_factor=2., scale_window=2000,
                 tolerance=0.):
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
