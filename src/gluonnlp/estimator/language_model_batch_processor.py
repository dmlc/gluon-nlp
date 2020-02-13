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
from ..utils import Parallel
from ..model.train.language_model import ParallelBigRNN

__all__ = ['LanguageModelBatchProcessor', 'ParallelLanguageModelBatchProcessor']

class LanguageModelBatchProcessor(BatchProcessor):
    def __init__(self):
        pass

    def fit_batch(self, estimator, train_batch, batch_axis=0):
        data = train_batch[:-1]
        target = train_batch[1:]
        batch_size = train_batch.shape[batch_axis]
        data = split_and_load(data, estimator.context, batch_axis=batch_axis, even_split=True)
        target = split_and_load(target, estimator.context, batch_axis=batch_axis, even_split=True)
        if estimator.hiddens is None:
            estimator.hiddens = [estimator.net.begin_state(batch_size // len(estimator.context),
                                                           func=mx.nd.zeros,
                                                           ctx=ctx) for ctx in estimator.context]
        else:
            estimator.hiddens = estimator.detach(estimator.hiddens)
        
        Ls = []
        outputs = []
        data_size = 0
        with mx.autograd.record():
            for i, (X, y, h) in enumerate(zip(data, target, estimator.hiddens)):
                output, h, encoder_hs, dropped_encoder_hs = estimator.net(X, h)
                l = estimator.loss(output, y, encoder_hs, dropped_encoder_hs)
                Ls.append(l / (len(estimator.context) * X.size))
                estimator.hiddens[i] = h
                outputs.append(output)

        for L in Ls:
            L.backward()

        Ls = [l * (len(estimator.context) * X.size) for l in Ls]
        return data, target, outputs, Ls

    def evaluate_batch(self, estimator, val_batch, batch_axis=0):
        batch_axis = 1 #temporary work around, removed after estimator is fixed
        data = val_batch[:-1]
        target = val_batch[1:]
        batch_size = val_batch.shape[batch_axis]
        data = split_and_load(data, estimator.context, batch_axis=batch_axis, even_split=True)
        target = split_and_load(target, estimator.context, batch_axis=batch_axis, even_split=True)

        Ls = []
        outputs = []
        if estimator.val_hiddens is None:
            estimator.val_hiddens = \
            [estimator.val_net.begin_state(batch_size //
                                            len(estimator.context), func=mx.nd.zeros, ctx=ctx) for ctx \
             in estimator.context]
        else:
            estimator.val_hiddens = estimator.detach(estimator.val_hiddens)
        for i, (X, y, h) in enumerate(zip(data, target, estimator.val_hiddens)):
            output, h = estimator.val_net(X, h)
            L = estimator.val_loss(output.reshape(-3, -1), y.reshape(-1,))
            estimator.val_hiddens[i] = h
            Ls.append(L)
            outputs.append(output)

        return data, target, outputs, Ls

class ParallelLanguageModelBatchProcessor(BatchProcessor):
    def __init__(self, loss, vocab, batch_size, val_batch_size):
        self.loss = loss
        self.parallel_model = None
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.vocab = vocab

    def _get_parallel_model(self, estimator):
        if self.parallel_model is None:
            self.parallel_model = ParallelBigRNN(estimator.net, self.loss, self.batch_size)
            self.parallel_model = Parallel(len(estimator.context), self.parallel_model)

    def fit_batch(self, estimator, train_batch, batch_axis=0):
        self._get_parallel_model(estimator)
        data, target, mask, sample = train_batch
        if estimator.hiddens is None:
            estimator.hiddens = [estimator.net.begin_state(batch_size=self.batch_size,
                                                           func=mx.nd.zeros,
                                                           ctx=ctx) for ctx in estimator.context]
        else:
            estimator.hiddens = estimator.detach(estimator.hiddens)
        Ls = []
        for _, batch in enumerate(zip(data, target, mask, sample, estimator.hiddens)):
            self.parallel_model.put(batch)

        for _ in range(len(data)):
            hidden, ls = self.parallel_model.get()
            index = estimator.context.index(hidden[0].context)
            estimator.hiddens[index] = hidden
            Ls.append(ls)

        Ls = [l / estimator.bptt for l in Ls]
        Ls = [mx.nd.sum(l) for l in Ls]
        return data, target, None, Ls

    def evaluate_batch(self, estimator, val_batch, batch_axis=0):
        data, target = val_batch
        ctx = estimator.context[0]
        data = data.as_in_context(ctx)
        target = target.as_in_context(ctx)
        if estimator.val_hiddens is None:
            estimator.val_hiddens = estimator.val_net.begin_state(batch_size=self.val_batch_size,
                                                               func=mx.nd.zeros,
                                                               ctx=ctx)
        else:
            estimator.val_hiddens = estimator.detach(estimator.val_hiddens)

        mask = data != self.vocab[self.vocab.padding_token]
        mask = mask.reshape(-1)
        output, estimator.val_hiddens = estimator.val_net(data, estimator.val_hiddens)
        output = output.reshape((-3, -1))
        L = estimator.val_loss(output, target.reshape(-1, ) * mask.reshape(-1))

        return data, [target, mask], output, L
