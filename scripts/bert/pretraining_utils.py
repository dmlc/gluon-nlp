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
# pylint:disable=redefined-outer-name,logging-format-interpolation
import os
import logging
import gluonnlp as nlp
import mxnet as mx

__all__ = ['get_model']

def get_model(ctx, model, pretrained, dataset_name, dtype, ckpt_dir=None, start_step=None):
    """Get model for pre-training."""
    # model
    model, vocabulary = nlp.model.get_model(model,
                                            dataset_name=dataset_name,
                                            pretrained=pretrained, ctx=ctx)

    if not pretrained:
        model.initialize(init=mx.init.Normal(0.02), ctx=ctx)

    if ckpt_dir and start_step:
        param_path = os.path.join(ckpt_dir, '%07d.params'%start_step)
        model.load_parameters(param_path, ctx=ctx)
        logging.info('Loading step %d checkpoints from %s.', start_step, param_path)

    model.cast(dtype)
    model.hybridize(static_alloc=True)

    # losses
    nsp_loss = mx.gluon.loss.SoftmaxCELoss()
    mlm_loss = mx.gluon.loss.SoftmaxCELoss()
    nsp_loss.hybridize(static_alloc=True)
    mlm_loss.hybridize(static_alloc=True)

    return model, nsp_loss, mlm_loss, vocabulary
