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

# coding: utf-8
# pylint: disable=wildcard-import, unused-variable
""" Gluon Languange Model Estimator """

import mxnet as mx
from mxnet.gluon.contrib.estimator import BatchProcessor
from mxnet.gluon.utils import split_and_load

__all__ = ['LanguageModelBatchProcessor']

class LanguageModelBatchProcessor(BatchProcessor):
    def __init__(self):
        pass

    def fit_batch(self, estimator, train_batch, batch_axis=0):
        data = train_batch[:-1]
        target = train_batch[1:]
        batch_size = train_batch.shape[batch_axis]
        data = split_and_load(data, estimator.context, batch_axis=batch_axis, even_split=True)
        target = split_and_load(target, estimator.context, batch_axis=batch_axis, even_split=True)
        
        Ls = []
        outputs = []
        data_size = 0
        if estimator.hiddens is None:
            estimator.hiddens = [estimator.net.begin_state(batch_size // len(estimator.context),
                                                           func=mx.nd.zeros,
                                                           ctx=ctx) for ctx in estimator.context]
        else:
            estimator.hiddens = estimator.detach(estimator.hiddens)
        with mx.autograd.record():
            for i, (X, y, h) in enumerate(zip(data, target, estimator.hiddens)):
                output, h, encoder_hs, dropped_encoder_hs = estimator.net(X, h)
                l = estimator.loss(output, y, encoder_hs, dropped_encoder_hs)
                Ls.append(l / (len(estimator.context) * X.size))
                estimator.hiddens[i] = h
                outputs.append(output)

        for L in Ls:
            L.backward()

        return data, target, outputs, Ls

    def evaluate_batch(self, estimator, val_batch, batch_axis=0):
        batch_size = val_batch.shape[batch_axis]
        val_batch = [split_and_load(x, ctx_list=estimator.context, batch_axis=batch_axis) for x in val_batch]
        data, target = val_batch
        Ls = []
        outputs = []
        if estimator.eval_hiddens is None:
            estimator.eval_hiddens = \
            [estimator.eval_net.begin_state(batch_size //
                                            len(estimator.context), func=mx.nd.zeros, ctx=ctx) for ctx \
             in estimator.context]
        else:
            estimator.eval_hiddens = estimator.detach(estimator.eval_hiddens)
        for i, (X, y, h) in enumerate(zip(data, target, estimator.eval_hiddens)):
            output, h = estimator.eval_net(X, h)
            L = estimator.evaluation_loss(output.reshape(-3, -1), y.reshape(-1,))
            self.eval_hiddens[i] = h
            Ls.append(L)
            outputs.append(output)

        return data, target, outputs, Ls
