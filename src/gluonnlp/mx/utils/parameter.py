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
"""Utility functions for trainer and parameters."""
__all__ = ['grad_global_norm',
           'clip_grad_global_norm', 'deduplicate_param_dict',
           'count_parameters', 'AverageSGDTracker']


import warnings

import numpy as np
import mxnet as mx
from collections import defaultdict
from mxnet.gluon import Parameter
from mxnet.util import use_np
from typing import Iterable, Optional, Tuple
from collections import OrderedDict
from mxnet.gluon.utils import shape_is_known


class AverageSGDTracker(object):
    def __init__(self, params=None):
        """Maintain a set of shadow variables "v" that is calculated by

            v[:] = (1 - 1/t) v + 1/t \theta

        The t is the number of training steps.

        It is also known as "Polyak-Rupert averaging" applied to SGD and was rediscovered in
        "Towards Optimal One Pass Large Scale Learning withAveraged Stochastic Gradient Descent"
         Wei Xu (2011).

        The idea is to average the parameters obtained by stochastic gradient descent.


        Parameters
        ----------
        params : ParameterDict
            The parameters that we are going to track.
        """
        self._track_params = None
        self._average_params = None
        self._initialized = False
        self._n_steps = 0
        if params is not None:
            self.apply(params)

    @property
    def n_steps(self):
        return self._n_steps

    @property
    def average_params(self):
        return self._average_params

    @property
    def initialized(self):
        return self._initialized

    def apply(self, params):
        """ Tell the moving average tracker which parameters we are going to track.

        Parameters
        ----------
        params : ParameterDict
            The parameters that we are going to track and calculate the moving average.
        """
        assert self._track_params is None, 'The MovingAverageTracker is already initialized and'\
                                           ' is not allowed to be initialized again. '
        self._track_params = deduplicate_param_dict(params)
        self._n_steps = 0

    def step(self):
        assert self._track_params is not None, 'You will need to use `.apply(params)`' \
                                               ' to initialize the MovingAverageTracker.'
        for k, v in self._track_params.items():
            assert shape_is_known(v.shape),\
                'All shapes of the tracked parameters must be given.' \
                ' The shape of {} is {}, and it has not been fully initialized.' \
                ' You should call step after the first forward of the model.'.format(k, v.shape)
        ctx = next(iter(self._track_params.values())).list_ctx()[0]
        if self._average_params is None:
            self._average_params = OrderedDict([(k, v.data(ctx).copy())
                                                for k, v in self._track_params.items()])
        self._n_steps += 1
        decay = 1.0 / self._n_steps
        for name, average_param in self._average_params.items():
            average_param += decay * (self._track_params[name].data(ctx) - average_param)

    def copy_back(self, params=None):
        """ Copy the average parameters back to the given parameters

        Parameters
        ----------
        params : ParameterDict
            The parameters that we will copy tha average params to.
            If it is not given, the tracked parameters will be updated

        """
        if params is None:
            params = self._track_params
        for k, v in self._average_params.items():
            params[k].set_data(v)


def grad_global_norm(parameters: Iterable[Parameter]) -> float:
    """Calculate the 2-norm of gradients of parameters, and how much they should be scaled down
    such that their 2-norm does not exceed `max_norm`, if `max_norm` if provided.
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
        norm = grad_global_norm(net.collect_params().values())
        ...

    Parameters
    ----------
    parameters
        The list of Parameters

    Returns
    -------
    total_norm
        Total norm. It's a numpy scalar.
    """
    # Distribute gradients among contexts,
    # For example, assume there are 8 weights and four GPUs, we can ask each GPU to
    # compute the squared sum of two weights and then add the results together
    idx = 0
    arrays = defaultdict(list)
    sum_norms = []
    num_ctx = None
    param_uuid_set = set()
    for p in parameters:
        if p._uuid in param_uuid_set:
            continue
        param_uuid_set.add(p._uuid)
        if p.grad_req != 'null':
            p_grads = p.list_grad()
            if num_ctx is None:
                num_ctx = len(p_grads)
            else:
                assert num_ctx == len(p_grads)
            arrays[idx % num_ctx].append(p_grads[idx % num_ctx])
            idx += 1
    assert len(arrays) > 0, 'No parameter found available for gradient norm.'

    # TODO(sxjscience)
    #  Investigate the float16 case.
    #  The inner computation accumulative type of norm should be float32.
    ctx = arrays[0][0].context
    for idx, arr_l in enumerate(arrays.values()):
        sum_norm = mx.np.linalg.norm(mx.np.concatenate([mx.np.ravel(ele) for ele in arr_l]))
        sum_norms.append(sum_norm.as_in_ctx(ctx))

    # Reduce over ctx
    if num_ctx == 1:
        total_norm = sum_norms[0]
    else:
        total_norm = mx.np.linalg.norm(mx.np.concatenate(sum_norms, axis=None))
    total_norm = float(total_norm)
    return total_norm


def clip_grad_global_norm(parameters: Iterable[Parameter],
                          max_norm: float,
                          check_isfinite: bool = True) -> Tuple[float, float, bool]:
    """Rescales gradients of parameters so that the sum of their 2-norm is smaller than `max_norm`.
    If gradients exist for more than one context for a parameter, user needs to explicitly call
    ``trainer.allreduce_grads`` so that the gradients are summed first before calculating
    the 2-norm.

    .. note::

        This function is only for use when `update_on_kvstore` is set to False in trainer.
        In cases where training happens on multiple contexts, this method should be used in
        conjunction with ``trainer.allreduce_grads()`` and ``trainer.update()``.
        (**not** ``trainer.step()``)

    Example::
    
        trainer = Trainer(net.collect_params(), update_on_kvstore=False, ...)
        for x, y in mx.gluon.utils.split_and_load(X, [mx.gpu(0), mx.gpu(1)]):
            with mx.autograd.record():
                y = net(x)
                loss = loss_fn(y, label)
            loss.backward()
        trainer.allreduce_grads()
        nlp.utils.clip_grad_global_norm(net.collect_params().values(), max_norm)
        trainer.update(batch_size)
        ...

    Parameters
    ----------
    parameters
        The list of parameters to calculate the norm
    max_norm
        If the gradient norm is larger than max_norm, it will be clipped to have max_norm
    check_isfinite
         If True, check whether the total_norm is finite (not nan or inf).
    Returns
    -------
    total_norm
        The total norm
    ratio
        The expected clipping ratio: grad = grad / ratio
        It will be calculated as max(total_norm / max_norm, 1)
    is_finite
        Whether the total norm is finite
    """
    total_norm = grad_global_norm(parameters)
    is_finite = bool(np.isfinite(total_norm))
    ratio = np.maximum(1, total_norm / max_norm)
    if check_isfinite and not is_finite:
        warnings.warn(
            UserWarning('nan or inf is detected. Clipping results will be undefined.'
                        ' Thus, skip clipping'),
            stacklevel=2)
        return total_norm, ratio, is_finite
    scale = 1 / ratio
    param_uuid_set = set()
    for p in parameters:
        if p._uuid in param_uuid_set:
            continue
        param_uuid_set.add(p._uuid)
        if p.grad_req != 'null':
            for arr in p.list_grad():
                arr *= scale
    return total_norm, ratio, is_finite


@use_np
def move_to_ctx(arr, ctx):
    """Move a nested structure of array to the given context

    Parameters
    ----------
    arr
        The input array
    ctx
        The MXNet context

    Returns
    -------
    new_arr
        The array that has been moved to context
    """
    if isinstance(arr, tuple):
        return tuple(move_to_ctx(ele, ctx) for ele in arr)
    elif isinstance(arr, list):
        return [move_to_ctx(ele, ctx) for ele in arr]
    else:
        return None if arr is None else arr.as_in_ctx(ctx)


def deduplicate_param_dict(param_dict):
    """Get a parameter dict that has been deduplicated

    Parameters
    ----------
    param_dict
        The parameter dict returned by `model.collect_params()`

    Returns
    -------
    dedup_param_dict
    """
    dedup_param_dict = dict()
    param_uuid_set = set()
    for k in sorted(param_dict.keys()):
        v = param_dict[k]
        if v._uuid in param_uuid_set:
            continue
        dedup_param_dict[k] = v
        param_uuid_set.add(v._uuid)
    return dedup_param_dict


# TODO(sxjscience) Consider to move it into the official MXNet gluon package
#  Also currently we have not printed the grad_req flag in Parameters, i.e.,
#  print(net.collect_params()) will not print the grad_req flag.
def count_parameters(params) -> Tuple[int, int]:
    """

    Parameters
    ----------
    params
        The input parameter dict

    Returns
    -------
    num_params
        The number of parameters that requires gradient
    num_fixed_params
        The number of parameters that does not require gradient
    """
    num_params = 0
    num_fixed_params = 0
    param_uuid_set = set()
    for k, v in params.items():
        if v._uuid in param_uuid_set:
            continue
        param_uuid_set.add(v._uuid)
        if v.grad_req != 'null':
            if v._data is None:
                warnings.warn('"{}" is not initialized! The total parameter count '
                              'will not be correct.'.format(k))
            else:
                num_params += np.prod(v.shape)
        else:
            if v._data is None:
                warnings.warn('"{}" is not initialized! The total fixed parameter count '
                              'will not be correct.'.format(k))
            else:
                num_fixed_params += np.prod(v.shape)
    return num_params, num_fixed_params

