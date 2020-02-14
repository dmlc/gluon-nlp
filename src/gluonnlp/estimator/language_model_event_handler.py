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

import time

import mxnet as mx
from mxnet.gluon.contrib.estimator import EpochBegin, EpochEnd
from mxnet.gluon.contrib.estimator import BatchBegin, BatchEnd
from mxnet.gluon.contrib.estimator import GradientUpdateHandler, LoggingHandler
from mxnet.gluon.contrib.estimator import MetricHandler
from mxnet.gluon.utils import clip_global_norm
from mxnet.metric import Loss as MetricLoss
from ..metric.length_normalized_loss import LengthNormalizedLoss

__all__ = ['HiddenStateHandler', 'AvgParamHandler', 'LearningRateHandler',
           'RNNGradientUpdateHandler', 'MetricResetHandler',
           'WordLanguageModelCheckpointHandler', 'ParallelLoggingHandler',
           'LargeRNNGradientUpdateHandler']

class HiddenStateHandler(EpochBegin):
    '''Hidden state reset event handler

    Reset hidden states for language model at each epoch
    '''
    def __init__(self):
        pass

    def epoch_begin(self, estimator, *args, **kwargs):
        estimator.hiddens = None
        estimator.val_hiddens = None

class AvgParamHandler(BatchEnd, EpochEnd):
    '''NTASGD average parameter event handler

    Average model parameters used in word language model estimator

    Parameters
    ----------
    data_length: int
        Length of training data, i.e., len(train_data). It is used to normalize the weight
        average coefficient.
    '''
    def __init__(self, data_length):
        self.epoch_id = 0
        self.batch_id = 0
        self.avg_trigger = 0
        self.t = 0
        self.n = 5
        self.valid_losses = []
        self.data_length = data_length

    def batch_end(self, estimator, *args, **kwargs):
        parameters = estimator.net.collect_params()
        if estimator.ntasgd:
            if estimator.avg_param is None:
                estimator.avg_param =
                {k.split(estimator.net._prefix)[1]:
                 v.data(estimator.context[0]).copy()
                 for k, v in parameters.items()}
            else:
                gamma = 1. / max(1, self.epoch_id * (self.data_length // estimator.bptt) +
                                 self.batch_id - self.avg_trigger + 2)
                for key, val in estimator.avg_param.items():
                    val[:] += gamma * (parameters['{}{}'.format(estimator.net._prefix, key)]
                                       .data(estimator.context[0]) - val)
        self.batch_id += 1

    def epoch_end(self, estimator, *args, **kwargs):
        if not isinstance(estimator.val_metrics, list):
            val_metrics = [estimator.val_metrics]
        else:
            val_metrics = estimator.val_metrics
        parameters = estimator.net.collect_params()
        if self.avg_trigger == 0:
            if self.t > self.n and val_metrics[0].get()[1] > min(self.valid_losses[-self.n:]):
                if estimator.avg_param is None:
                    estimator.avg_param =
                    {k.split(estimator.net._prefix)[1]:
                     v.data(estimator.context[0]).copy()
                     for k, v in
                     parameters.items()}
                else:
                    for key, val in parameters.items():
                        estimator.avg_param[key.split(estimator.net._prefix)[1]] \
                            = val.data(estimator.context[0]).copy()
                self.avg_trigger = (self.epoch_id + 1) * (self.data_length // estimator.bptt)
                print('Switching to NTASGD and avg_trigger is : %d' % self.avg_trigger)
                estimator.ntasgd = True
            self.valid_losses.append(val_metrics[0].get()[1])
            self.t += 1
        self.batch_id = 0
        self.epoch_id += 1

class LearningRateHandler(BatchBegin, BatchEnd, EpochEnd):
    '''NTASGD learning rate event handler

    Dynamically adjust the learning rate during word language model training
    TODO: Investigate whether the learing rate event handler can be replaced with
    learning rate scheduler

    Parameters
    ----------
    lr_update_interval : int
        Epoch interval of updating the learning rate during training the word
        language model
    lr_update_factor : float
        learning rate decay factor used when updating the learning rate
    '''
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
        if not isinstance(estimator.val_metrics, list):
            val_metrics = [estimator.val_metrics]
        else:
            val_metrics = estimator.val_metrics

        if val_metrics[0].get()[1] < self.best_val:
            self.update_lr_epoch = 0
            self.best_val = val_metrics[0].get()[1]
        else:
            self.update_lr_epoch += 1
            if self.update_lr_epoch % self.lr_update_interval == 0 and self.update_lr_epoch != 0:
                lr_scale = estimator.trainer.learning_rate * self.lr_update_factor
                estimator.trainer.set_learning_rate(lr_scale)
                self.update_lr_epoch = 0

class RNNGradientUpdateHandler(GradientUpdateHandler):
    '''NTASGD gradient clipping update event handler

    clipping gradient during word language model training
    Parameters
    ----------
    clip : clip
        Gradient clipping threshold. Gradient norm exceeds this value should be scaled
        down within the valid range. 
    '''
    def __init__(self, clip=None, **kwargs):
        super().__init__(**kwargs)
        self.clip = clip

    def batch_end(self, estimator, *args, **kwargs):
        loss = kwargs['loss']
        loss_size = sum([l.size for l in loss])
        parameters = estimator.net.collect_params()
        grads = [p.grad(ctx) for p in parameters.values() for ctx in estimator.context]
        if self.clip is not None:
            clip_global_norm(grads, self.clip)

        estimator.trainer.step(1)

class LargeRNNGradientUpdateHandler(GradientUpdateHandler):
    '''Parallel Large RNN gradient clipping update event handler

    Rescale gradients of embedding parameters and clipping gradients of encoder parameters
    during training parallel large RNN

    Parameters
    ----------
    batch_size : int
        batch size per gpu used during training parallel large RNN
    clip : float
        gradient clipping threshold. Gradients of encoder parameters exceed this value
        should be scaled down within the valid range.
    '''
    def __init__(self, batch_size, clip=None, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.clip = clip

    def batch_end(self, estimator, *args, **kwargs):
        encoder_params = estimator.net.encoder.collect_params().values()
        embedding_params = list(estimator.net.embedding.collect_params().values())

        for ctx in estimator.context:
            x = embedding_params[0].grad(ctx)
            x[:] *= self.batch_size
            encoder_grad = [p.grad(ctx) for p in encoder_params]
            clip_global_norm(encoder_grad, self.clip)

        estimator.trainer.step(len(estimator.context))

class MetricResetHandler(BatchBegin, MetricHandler):
    '''Event handler for reseting local metrics

    Reset local metrics for each few iterations and add support of LengthNormalizedMetrics
    to compute both local and global metrics.
    TODO: Move this event handler to be reusable by other estimators, e.g.,
    MachineTranslationEstimator

    Parameters
    ----------
    Metrics : mxnet.metric
        Metrics to be reset during training
    log_interval : int or None
        If log_interval is of int type, it represents the interval of reseting local
        metrics. Otherwise, metrics do not need to be reset. 
    '''
    def __init__(self, metrics, log_interval=None):
        super().__init__(metrics=metrics)
        self.batch_id = 0
        self.log_interval = log_interval

    def epoch_begin(self, estimator, *args, **kwargs):
        self.batch_id = 0
        for metric in self.metrics:
            metric.reset()

    def batch_begin(self, estimator, *args, **kwargs):
        if self.log_interval is not None:
            if self.batch_id % self.log_interval == 0:
                for metric in self.metrics:
                    metric.reset_local()
        self.batch_id += 1

    def batch_end(self, estimator, *args, **kwargs):
        pred = kwargs['pred']
        label = kwargs['label']
        loss = kwargs['loss']
        for metric in self.metrics:
            if isinstance(metric, MetricLoss):
                metric.update(0, loss)
            elif isinstance(metric, LengthNormalizedLoss):
                metric.update(label, loss)
            else:
                metric.update(label, pred)

class WordLanguageModelCheckpointHandler(EpochEnd):
    '''Checkpoint Event handler of word language model

    Save the model checkpoint of word language model
    
    Parameters
    ----------
    save : string
        The model checkpoint save path prefix
    '''
    def __init__(self, save):
        self.save = save
        self.best_val = float('Inf')

    def epoch_end(self, estimator, *args, **kwargs):
        if not isinstance(estimator.val_metrics, list):
            val_metrics = [estimator.val_metrics]
        else:
            val_metrics = estimator.val_metrics

        if estimator.ntasgd:
            mx.nd.save('{}.val.params'.format(self.save), estimator.avg_param)
        else:
            estimator.net.save_parameters('{}.val.params'.format(self.save))

        if val_metrics[0].get()[1] < self.best_val:
            self.best_val = val_metrics[0].get()[1]
            if estimator.ntasgd:
                mx.nd.save(self.save, estimator.avg_param)
            else:
                estimator.net.save_parameters(self.save)


class ParallelLoggingHandler(LoggingHandler):
    '''Logging handler of Parallel language model training

    Generating logging information of parallel large RNN training. This event handler
    is designed specifically to handle the batches taken from multiple gpus.
    '''
    def __init__(self, *args, **kwargs):
        super(ParallelLoggingHandler, self).__init__(*args, **kwargs)

    def batch_end(self, estimator, *args, **kwargs):
        if isinstance(self.log_interval, int):
            batch_time = time.time() - self.batch_start
            msg = '[Epoch %d][Batch %d]' % (self.current_epoch, self.batch_index)
            cur_batches = kwargs['batch'][0]
            for batch in cur_batches:
                self.processed_samples += batch.shape[0]
            msg += '[Samples %s]' % (self.processed_samples)
            self.log_interval_time += batch_time
            if self.batch_index % self.log_interval == self.log_interval - 1:
                msg += 'time/interval %.3fs ' % self.log_interval_time
                self.log_interval_time = 0
                for metric in self.metrics:
                    name, val = metric.get()
                    msg += '%s: %.4f, ' % (name, val)
                estimator.logger.info(msg.rstrip(', '))
        self.batch_index += 1

