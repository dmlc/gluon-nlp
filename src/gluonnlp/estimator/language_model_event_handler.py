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
""" Gluon Language Model Event Handler """

import copy
import warnings

import mxnet as mx
from mxnet.gluon.contrib.estimator import TrainBegin, TrainEnd, EpochBegin
from mxnet.gluon.contrib.estimator import EpochEnd, BatchBegin, BatchEnd
from mxnet.gluon.contrib.estimator import GradientUpdateHandler
from mxnet.gluon.utils import clip_global_norm

__all__ = ['HiddenStateHandler', 'AvgParamHandler', 'LearningRateHandler',
           'RNNGradientUpdateHandler']

class HiddenStateHandler(EpochBegin):
    def __init__(self):
        pass

    def epoch_begin(self, estimator, *args, **kwargs):
        estimator.hiddens = None
        estimator.eval_hiddens = None
    
class AvgParamHandler(BatchEnd, EpochEnd):
    def __init__(self):
        self.ntasgd = False
        self.epoch_id = 0
        self.batch_id = 0
        self.avg_trigger = 0
        # self.ntasgd is always False during the first epoch
        self.batches_per_epoch = 0
        self.t = 0
        self.n = 5
        self.valid_losses = []

    def batch_end(self, estimator, *args, **kwargs):
        parameters = estimator.net.collect_params()
        if self.ntasgd:
            if estimator.avg_param is None:
                estimator.avg_param = {k.split(estimator.net._prefix)[1]: v.data(estimator.context[0]).copy()
                                       for k, v in parameters.items()}
            else:
                gamma = 1. / max(1, self.epoch_id * (self.batches_per_epoch // estimator.bptt) +
                                 self.batch_index - avg_trigger + 2)
                for key, val in estimator.avg_param.items():
                    val[:] += gamma * (parameters['{}{}'.format(estimator.net.__prefix, key)]
                                       .data(estimator.context[0]) - val)
        self.batch_id += 1

    def epoch_end(self, estimator, *args, **kwargs):
        parameters = estimator.net.collect_params()
        self.batches_per_epoch = self.batch_id
        if self.ntasgd == False and self.avg_trigger == 0:
            if self.t > self.n and estimator.val_metrics > min(self.valid_losses[-self.n:]):
                if estimator.avg_param is None:
                    estimator.avg_param = {k.split(estimator.net._prefix)[1]: v.data(estimator.context[0]).copy()
                                           for k, v in parameters.items()}
                else:
                    for key, val in parameters.items():
                        estimator.avg_param[key.split(estimator.net._prefix)[1]] \
                            = val.data(estimator.context[0]).copy()
                self.avg_trigger = (self.epoch_id + 1) * (self.batches_per_epoch // estimator.bptt)
                print('Switching to NTASGD and avg_trigger is : %d' % self.avg_trigger)
                self.ntasgd = True
            self.valid_losses.append(estimator.val_metrics)
            self.t += 1
        self.batch_id = 0
        self.epoch_id += 1

class LearningRateHandler(BatchBegin, BatchEnd, EpochEnd):
    def __init__(self, lr_update_interval=30, lr_update_factor=0.1):
        self.lr_batch_start = 0
        self.best_val = float('Inf')
        self.update_lr_epoch = 0
        self.lr_update_interval = lr_update_interval
        self.lr_update_factor = lr_update_factor

    def batch_begin(self, estimator, *args, **kwargs):
        batch = kwargs['batch']
        self.lr_batch_start = estimator.trainer.learning_rate
        seq_len = batch.shape[0] - 1
        estimator.trainer.set_learning_rate(self.lr_batch_start * seq_len / estimator.bptt)

    def batch_end(self, estimator, *args, **kwargs):
        estimator.trainer.set_learning_rate(self.lr_batch_start)

    def epoch_end(self, estimator, *args, **kwargs):
        if estimator.val_metrics < self.best_val:
            self.update_lr_epoch = 0
            self.best_val = estimator.val_metrics
        else:
            self.update_lr_epoch += 1
            if self.update_lr_epoch % self.lr_update_interval == 0 and self.update_lr_epoch != 0:
                lr_scale = estimator.trainer.learning_rate * self.lr_update_factor
                estimator.trainer.set_learning_rate(lr_scale)
                self.update_lr_epoch = 0

class RNNGradientUpdateHandler(GradientUpdateHandler):
    def __init__(self, clip=None, **kwargs):
        super().__init__(**kwargs)
        self.clip = clip

    def batch_end(self, estimator, *args, **kwargs):
        parameters = estimator.net.collect_params()
        grads = [p.grad(ctx) for p in parameters.values() for ctx in estimator.context]
        if self.clip is not None:
            clip_global_norm(grads, self.clip)

        estimator.trainer.step(1)
