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
"""Utility functions for parameters."""

__all__ = ['clip_grad_global_norm']

import warnings

import numpy as np
from mxnet import nd

def clip_grad_global_norm(parameters, max_norm, check_isfinite=True):
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
        nlp.utils.clip_grad_global_norm(net.collect_params().values(), max_norm)
        trainer.update(batch_size)
        ...

    Parameters
    ----------
    parameters : list of Parameters
    max_norm : float
    check_isfinite : bool, default True
         If True, check that the total_norm is finite (not nan or inf). This
         requires a blocking .asscalar() call.

    Returns
    -------
    NDArray or float
      Total norm. Return type is NDArray of shape (1,) if check_isfinite is
      False. Otherwise a float is returned.

    """
    def _norm(array):
        if array.stype == 'default':
            x = array.reshape((-1))
            return nd.dot(x, x)
        return array.norm().square()

    arrays = []
    i = 0
    for p in parameters:
        if p.grad_req != 'null':
            grad_list = p.list_grad()
            arrays.append(grad_list[i % len(grad_list)])
            i += 1
    assert len(arrays) > 0, 'No parameter found available for gradient norm clipping.'
    ctx, dtype = arrays[0].context, arrays[0].dtype
    total_norm = nd.add_n(*[_norm(arr).as_in_context(ctx) for arr in arrays])
    total_norm = nd.sqrt(total_norm)
    if check_isfinite:
        total_norm = total_norm.asscalar()
        if not np.isfinite(total_norm):
            warnings.warn(
                UserWarning('nan or inf is detected. '
                            'Clipping results will be undefined.'), stacklevel=2)
    scale = max_norm / (total_norm + 1e-8)
    if check_isfinite:
        scale = nd.array([scale], dtype=dtype, ctx=ctx)
    scale = nd.min(nd.concat(scale, nd.ones((1,), dtype=dtype, ctx=ctx), dim=0))
    for p in parameters:
        if p.grad_req != 'null':
            for arr in p.list_grad():
                arr *= scale.as_in_context(arr.context)
    return total_norm
