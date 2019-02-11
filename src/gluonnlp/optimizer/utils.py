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

__all__ = ['FP16Trainer']

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

    #'''
    #def _clip_grad_norm(self, max_norm):
    #    """Clips gradient norm and updates dynamic loss scaler."""
    #    #self._unscale_grads()
    #    #grad_norm = self.wrapped_optimizer.clip_grad_norm(max_norm)

    #    # detect overflow and adjust loss scale
    #    overflow = DynamicLossScaler.has_overflow(grad_norm)
    #    self.scaler.update_scale(overflow)

    #    if overflow:
    #        if self.scaler.loss_scale <= self.args.min_loss_scale:
    #            # Use FloatingPointError as an uncommon error that parent
    #            # functions can safely catch to stop training.
    #            raise FloatingPointError((
    #                'Minimum loss scale reached ({}). Your loss is probably exploding. '
    #                'Try lowering the learning rate, using gradient clipping or '
    #                'increasing the batch size.'
    #            ).format(self.args.min_loss_scale))
    #        raise OverflowError('setting loss scale to: ' + str(self.scaler.loss_scale))
    #    return grad_norm
    #'''

    def step(self, batch_size, ignore_stale_grad=False, global_norm=None):
        self.fp32_trainer.allreduce_grads()
        if global_norm:
            from ..utils import clip_grad_global_norm
            norm = clip_grad_global_norm(self.fp32_trainer._params, global_norm * self._scaler.loss_scale)
            overflow = not np.isfinite(norm)
        else:
            overflow = self._scaler.has_overflow(self.fp32_trainer._params)

        self._scaler.update_scale(overflow)
        if not overflow:
            #print('perform update')
            self.fp32_trainer.update(batch_size * self._scaler.loss_scale,
                                     ignore_stale_grad=ignore_stale_grad)
        else:
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

#class DynamicLossScaler(object):
#    """The DynamicLossScaler.
#    """
#    def __init__(self, init_scale=2**32, scale_factor=2., scale_window=1000):
#    #def __init__(self, init_scale=2**15, scale_factor=2., scale_window=2000, tolerance=0.05):
#        self._init_scale = init_scale
#        self._scale_factor = scale_factor
#        self._scale_window = scale_window
#
#    '''
#    def __init__(self, args, params, fp32_optimizer, fp32_params):
#        super().__init__(args, params)
#        self.fp32_optimizer = fp32_optimizer
#        self.fp32_params = fp32_params
#
#        if getattr(args, 'fp16_scale_window', None) is None:
#            if len(args.update_freq) > 1:
#                raise ValueError(
#                    '--fp16-scale-window must be given explicitly when using a '
#                    'custom --update-freq schedule'
#                )
#            scale_window = 2**14 / args.distributed_world_size / args.update_freq[0]
#        else:
#            scale_window = args.fp16_scale_window
#
#        self.scaler = DynamicLossScaler(
#            init_scale=args.fp16_init_scale,
#            scale_window=scale_window,
#            tolerance=args.fp16_scale_tolerance,
#        )
#
#    parser.add_argument('--fp16-init-scale', default=2**7, type=int,
#                        help='default FP16 loss scale')
#    parser.add_argument('--fp16-scale-window', type=int,
#                        help='number of updates before increasing loss scale')
#    parser.add_argument('--fp16-scale-tolerance', default=0.0, type=float,
#                        help='pct of updates that can overflow before decreasing the loss scale')
#    group.add_argument('--min-loss-scale', default=1e-4, type=float, metavar='D',
#                       help='minimum loss scale (for FP16 training)')
#
#
#    def clip_grad_norm(self, max_norm):
#        """Clips gradient norm and updates dynamic loss scaler."""
#        self._sync_fp16_grads_to_fp32()
#        grad_norm = utils.clip_grad_norm_(self.fp32_params.grad.data, max_norm)
#
#        # detect overflow and adjust loss scale
#        overflow = DynamicLossScaler.has_overflow(grad_norm)
#        self.scaler.update_scale(overflow)
#        if overflow:
#            if self.scaler.loss_scale <= self.args.min_loss_scale:
#                # Use FloatingPointError as an uncommon error that parent
#                # functions can safely catch to stop training.
#                raise FloatingPointError((
#                    'Minimum loss scale reached ({}). Your loss is probably exploding. '
#                    'Try lowering the learning rate, using gradient clipping or '
#                    'increasing the batch size.'
#                ).format(self.args.min_loss_scale))
#            raise OverflowError('setting loss scale to: ' + str(self.scaler.loss_scale))
#        return grad_norm
#    '''
#
#    # `params` is a list / generator of torch.Variable
#    def has_overflow(self, params):
#        '''
#        total_norm = total_norm.asscalar()
#        if not np.isfinite(total_norm):
#            warnings.warn(
#                UserWarning('nan or inf is detected. '
#                            'Clipping results will be undefined.'), stacklevel=2)
#        '''
#        for p in params:
#            if p.grad is not None and DynamicLossScaler._has_inf_or_nan(p.grad.data):
#                return True
#
#        return False
#
#    # `x` is a torch.Tensor
#    def _has_inf_or_nan(x):
#        try:
#            # if x is half, the .float() incurs an additional deep copy, but it's necessary if 
#            # Pytorch's .sum() creates a one-element tensor of the same type as x 
#            # (which is true for some recent version of pytorch).
#            cpu_sum = float(x.float().sum())
#            # More efficient version that can be used if .sum() returns a Python scalar
#            # cpu_sum = float(x.sum())
#        except RuntimeError as instance:
#            # We want to check if inst is actually an overflow exception.
#            # RuntimeError could come from a different error.
#            # If so, we still want the exception to propagate.
#            if "value cannot be converted" not in instance.args[0]:
#                raise
#            return True
#        else:
#            if cpu_sum == float('inf') or cpu_sum == -float('inf') or cpu_sum != cpu_sum:
#                return True
#            return False
#
#    # `overflow` is boolean indicating whether the gradient overflowed
#    def update_scale(self, overflow):
#        if overflow:
#            # self.cur_scale /= self.scale_factor
#            self.cur_scale = max(self.cur_scale/self.scale_factor, 1)
#            self.last_overflow_iter = self.cur_iter
#        else:
#            if (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
#                self.cur_scale *= self.scale_factor
#        self.cur_iter += 1
#
#    @property
#    def loss_scale(self):
#        return self.cur_scale
#
#    def scale_gradient(self, module, grad_in, grad_out):
#        return tuple(self.loss_scale * g for g in grad_in)
#
#    def backward(self, loss, retain_graph=False):
#        scaled_loss = loss * self.loss_scale
#        scaled_loss.backward(retain_graph=retain_graph)

##############################################################        
# Example usage below here -- assuming it's in a separate file
##############################################################

"""
TO-DO separate out into an example.
if __name__ == "__main__":
    import torch
    from torch.autograd import Variable
    from dynamic_loss_scaler import DynamicLossScaler
    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10
    # Create random Tensors to hold inputs and outputs, and wrap them in Variables.
    x = Variable(torch.randn(N, D_in), requires_grad=False)
    y = Variable(torch.randn(N, D_out), requires_grad=False)
    w1 = Variable(torch.randn(D_in, H), requires_grad=True)
    w2 = Variable(torch.randn(H, D_out), requires_grad=True)
    parameters = [w1, w2]
    learning_rate = 1e-6
    optimizer = torch.optim.SGD(parameters, lr=learning_rate)
    loss_scaler = DynamicLossScaler()
    for t in range(500):
        y_pred = x.mm(w1).clamp(min=0).mm(w2)
        loss = (y_pred - y).pow(2).sum() * loss_scaler.loss_scale
        print('Iter {} loss scale: {}'.format(t, loss_scaler.loss_scale))
        print('Iter {} scaled loss: {}'.format(t, loss.data[0]))
        print('Iter {} unscaled loss: {}'.format(t, loss.data[0] / loss_scaler.loss_scale))
        # Run backprop
        optimizer.zero_grad()
        loss.backward()
        
        # Check for overflow
        has_overflow = DynamicLossScaler.has_overflow(parameters)
        
        # If no overflow, unscale grad and update as usual
        if not has_overflow:
            for param in parameters:
                param.grad.data.mul_(1. / loss_scaler.loss_scale)
            optimizer.step()
        # Otherwise, don't do anything -- ie, skip iteration
        else:
            print('OVERFLOW!')
        # Update loss scale for next iteration
        loss_scaler.update_scale(has_overflow)

"""
