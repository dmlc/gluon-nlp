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

__all__ = ['grad_global_norm', 'clip_grad_global_norm', 'save_parameters',
           'save_states', 'load_parameters', 'load_states']

import warnings

from collections import defaultdict
import mxnet as mx
from mxnet import nd
from .. import _constants as C
from .files import _TempFilePath, _transfer_file_s3

def grad_global_norm(parameters, max_norm=None):
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
    parameters : list of Parameters
    max_norm: NDArray, optional
        The maximum L2 norm threshold. If provided, `ratio` and `is_finite` will be returned.

    Returns
    -------
    NDArray
      Total norm. Shape is (1,)
    NDArray
      Ratio for rescaling gradients based on max_norm s.t. grad = grad / ratio.
      If total norm is NaN, ratio will be NaN, too.
      Returned if `max_norm` is provided. Shape is (1,)
    NDArray
      Whether the total norm is finite, returned if `max_norm` is provided. Shape is (1,)
    """
    # distribute gradients among contexts
    idx = 0
    arrays = defaultdict(list)
    sum_norms = []
    for p in parameters:
        if p.grad_req != 'null':
            p_grads = p.list_grad()
            arrays[idx % len(p_grads)].append(p_grads[idx % len(p_grads)])
            idx += 1
    assert len(arrays) > 0, 'No parameter found available for gradient norm.'

    ctx, dtype = arrays[0][0].context, 'float32'
    for idx, arr in enumerate(arrays.values()):
        sum_norm = mx.nd.multi_sum_sq(*arr, num_arrays=len(arr))
        sum_norm = nd.add_n(*sum_norm)
        sum_norms.append(sum_norm.as_in_context(ctx))

    # reduce
    total_norm = nd.add_n(*sum_norms).sqrt()
    if max_norm is None:
        return total_norm
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

def clip_grad_global_norm(parameters, max_norm, check_isfinite=True):
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
    total_norm, ratio, is_finite = grad_global_norm(parameters, max_norm)
    scale = 1 / ratio
    if check_isfinite:
        if is_finite != 1:
            warnings.warn(
                UserWarning('nan or inf is detected. '
                            'Clipping results will be undefined.'), stacklevel=2)
    for p in parameters:
        if p.grad_req != 'null':
            for arr in p.list_grad():
                arr *= scale.as_in_context(arr.context)
    return total_norm

def _s3_compatible_save_load(is_save, save_load_method, filename, *args, **kwargs):
    """Dispatch function for save load with s3."""
    if C.S3_PREFIX in filename:
        # create temp dir
        with _TempFilePath() as temp_path:
            if is_save:
                # save model
                save_load_method(temp_path, *args, **kwargs)
                _transfer_file_s3(temp_path, filename, upload=is_save)
            else:
                # load model
                _transfer_file_s3(temp_path, filename, upload=is_save)
                save_load_method(temp_path, *args, **kwargs)
    else:
        save_load_method(filename, *args, **kwargs)

def load_parameters(model, filename, ctx=None, allow_missing=False,
                    ignore_extra=False, cast_dtype=None):
    """Load parameters from file previously saved by `save_parameters`.

    Both local file system path and S3 URI are supported.
    For example, 's3://mybucket/folder/net.params', './folder/net.params'.

    Parameters
    ----------
    filename : str
        Path to parameter file.
    ctx : Context or list of Context, default cpu()
        Context(s) to initialize loaded parameters on.
    allow_missing : bool, default False
        Whether to silently skip loading parameters not represents in the file.
    ignore_extra : bool, default False
        Whether to silently ignore parameters from the file that are not
        present in this Block.
    cast_dtype : bool, default False
        Cast the data type of the NDArray loaded from the checkpoint to the dtype
        provided by the Parameter if any.
    """
    if cast_dtype is not None:
        _s3_compatible_save_load(False, model.load_parameters, filename, ctx=ctx,
                                 allow_missing=allow_missing, ignore_extra=ignore_extra,
                                 cast_dtype=cast_dtype)
    else:
        _s3_compatible_save_load(False, model.load_parameters, filename, ctx=ctx,
                                 allow_missing=allow_missing, ignore_extra=ignore_extra)

def save_parameters(model, filename):
    """Save parameters to file.

    Saved parameters can only be loaded with `Block.load_parameters`. Note that this
    method only saves parameters, not model structure.

    Both local file system path and S3 URI are supported.
    For example, 's3://mybucket/folder/net.params', './folder/net.params'.

    Parameters
    ----------
    model : mx.gluon.Block
        The model to save.
    uri : str
        Path to file.
    """
    _s3_compatible_save_load(True, model.save_parameters, filename)


def load_states(trainer, fname):
    """Loads trainer states (e.g. optimizer, momentum) from a file.

    Both local file system path and S3 URI are supported.
    For example, 's3://mybucket/folder/net.states', './folder/net.states'.

    Parameters
    ----------
    trainer : mxnet.gluon.Trainer
        The trainer whose states will be loaded.
    fname : str
        Path to input states file.

    Note
    ----
    `optimizer.param_dict`, which contains Parameter information (such as
    `lr_mult` and `wd_mult`) will not be loaded from the file, but rather set
    based on current Trainer's parameters.
    """
    _s3_compatible_save_load(False, trainer.load_states, fname)

def save_states(trainer, fname):
    """Saves trainer states (e.g. optimizer, momentum) to a file.

    Both local file system path and S3 URI are supported.
    For example, 's3://mybucket/folder/net.states', './folder/net.states'.

    Parameters
    ----------
    trainer : mxnet.gluon.Trainer
        The trainer whose states will be saved.
    fname : str
        Path to output states file.

    Note
    ----
    `optimizer.param_dict`, which contains Parameter information (such as
    `lr_mult` and `wd_mult`) will not be saved.
    """
    _s3_compatible_save_load(True, trainer.save_states, fname)
