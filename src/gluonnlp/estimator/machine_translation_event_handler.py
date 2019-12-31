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
# pylint: disable=eval-used, redefined-outer-name
""" Gluon Machine Translation Event Handler """

import copy
import warnings

import mxnet as mx
from mxnet.gluon.contrib.estimator import TrainBegin, TrainEnd, EpochBegin
from mxnet.gluon.contrib.estimator import EpochEnd, BatchBegin, BatchEnc
from mxnet.gluon.contrib.estimator import GradientUpdateHandler
from mxnet.gluon.contrib.estimator import MetricHandler

__all__ = ['AvgParamUpdateHandler']

class AvgParamUpdateHandler(BatchEnd, EpochEnd):
    def __init__(self, avg_start, grad_interval=1):
        self.batch_id = 0
        self.grad_interval = grad_interval
        self.step_num = 0
        self.avg_start = avg_start

    def _update_avg_param(self, estimator):
        if estimator.avg_param is None:
            # estimator.net is parallel model estimator.net._model is the model
            # to be investigated on
            estimator.avg_param = {k:v.data(estimator.context[0]).copy() for k, v in
                                   estimator.net._model.collect_params().items()}
        if self.step_num > self.avg_start:
            params = estimator.net._model.collect_params()
            alpha = 1. / max(1, self.step_num - self.avg_start)
            for key, val in estimator.avg_param.items():
                estimator.avg_param[:] += alpha *
                (params[key].data(estimator.context[0]) -
                 val)

    def batch_end(self, estimator, *args, **kwargs):
        if self.batch_id % self.grad_interval == 0:
            self.step_num += 1
        if self.batch_id % self.grad_interval == self.grad_interval - 1:
            _update_avg_param(estimator)

    def epoch_end(self, estimator, *args, **kwargs):
        _update_avg_param(estimator)
