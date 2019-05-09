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

__all__ = ['clip_grad_global_norm', 'save_parameters', 'save_states']

import warnings
import os
import tempfile
import logging

import numpy as np
from mxnet import nd
from .. import _constants as C

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
    if C.S3_PREFIX in filename:
        # create temp dir
        temp_dir = os.path.join(tempfile.gettempdir(), str(hash(os.times())))
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        temp_path = os.path.join(temp_dir, str(hash(os.times())))
        # save model
        model.save_parameters(temp_path)
        _upload_file(temp_path, filename)
        os.remove(temp_path)
    else:
        model.save_parameters(filename)

def _upload_file(filename, s3_filename):
    """Upload file to S3."""
    try:
        import boto3
    except ImportError:
        raise ImportError('boto3 is required to support s3 URI. Please install'
                          'boto3 via `pip install boto3`')
    # parse s3 uri
    prefix_len = len(C.S3_PREFIX)
    bucket_idx = s3_filename[prefix_len:].index('/') + prefix_len
    bucket_name = s3_filename[prefix_len:bucket_idx]

    # filename after the bucket, excluding '/'
    s3_target_file = s3_filename[bucket_idx + 1:]

    log_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.INFO)
    # upload to s3
    s3 = boto3.client('s3')
    s3.upload_file(filename, bucket_name, s3_target_file)
    logging.getLogger().setLevel(log_level)

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
    if C.S3_PREFIX in fname:
        # create temp dir
        temp_dir = os.path.join(tempfile.gettempdir(), str(hash(os.times())))
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        temp_path = os.path.join(temp_dir, str(hash(os.times())))
        trainer.save_states(temp_path)
        _upload_file(temp_path, fname)
        os.remove(temp_path)
    else:
        trainer.save_states(fname)
