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
import math

import numpy as np
import mxnet as mx
from mxnet.gluon.contrib.estimator import TrainBegin, TrainEnd, EpochBegin
from mxnet.gluon.contrib.estimator import EpochEnd, BatchBegin, BatchEnd
from mxnet.gluon.contrib.estimator import GradientUpdateHandler
from mxnet.gluon.contrib.estimator import MetricHandler
from mxnet import gluon
from mxnet.metric import Loss as MetricLoss
from .length_normalized_loss import LengthNormalizedLoss

__all__ = ['MTTransformerParamUpdateHandler', 'TransformerLearningRateHandler',
           'MTTransformerMetricHandler', 'TransformerGradientAccumulationHandler',
           'ComputeBleuHandler', 'ValBleuHandler', 'MTGNMTGradientUpdateHandler']

class MTTransformerParamUpdateHandler(EpochBegin, BatchEnd, EpochEnd):
    def __init__(self, avg_start, grad_interval=1):
        self.batch_id = 0
        self.grad_interval = grad_interval
        self.step_num = 0
        self.avg_start = avg_start

    def _update_avg_param(self, estimator):
        if estimator.avg_param is None:
            estimator.avg_param = {k:v.data(estimator.context[0]).copy() for k, v in
                                   estimator.net.collect_params().items()}
        if self.step_num > self.avg_start:
            params = estimator.net.collect_params()
            alpha = 1. / max(1, self.step_num - self.avg_start)
            for key, val in estimator.avg_param.items():
                estimator.avg_param[:] += alpha * \
                                          (params[key].data(estimator.context[0]) -
                                           val)

    def epoch_begin(self, estimator, *args, **kwargs):
        self.batch_id = 0
                
    def batch_end(self, estimator, *args, **kwargs):
        if self.batch_id % self.grad_interval == 0:
            self.step_num += 1
        if self.batch_id % self.grad_interval == self.grad_interval - 1:
            self._update_avg_param(estimator)
        self.batch_id += 1

    def epoch_end(self, estimator, *args, **kwargs):
        self._update_avg_param(estimator)


class TransformerLearningRateHandler(EpochBegin, BatchBegin):
    def __init__(self, lr,
                 num_units=512,
                 warmup_steps=4000,
                 grad_interval=1):
        self.lr = lr
        self.num_units = num_units
        self.warmup_steps = warmup_steps
        self.grad_interval = grad_interval
        self.step_num = 0

    def epoch_begin(self, estimator, *args, **kwargs):
        self.batch_id = 0

    def batch_begin(self, estimator, *args, **kwargs):
        if self.batch_id % self.grad_interval == 0:
            self.step_num += 1
            new_lr = self.lr /  math.sqrt(self.num_units) * \
                     min(1. / math.sqrt(self.step_num), self.step_num *
                         self.warmup_steps ** (-1.5))
            estimator.trainer.set_learning_rate(new_lr)
        self.batch_id += 1

class MTGNMTGradientUpdateHandler(GradientUpdateHandler):
    def __init__(self, clip):
        super(MTGNMTGradientUpdateHandler, self).__init__()
        self.clip = clip

    def batch_end(self, estimator, *args, **kwargs):
        grads = [p.grad(ctx) for p in estimator.net.collect_params().values()]
        gnorm = gluon.utils.clip_global_norm(grads, self.clip)
        estimator.trainer.step(1)

class TransformerGradientAccumulationHandler(GradientUpdateHandler,
                                             TrainBegin,
                                             EpochBegin,
                                             EpochEnd):
    def __init__(self, grad_interval=1,
                 batch_size=1024,
                 rescale_loss=100):
        super(TransformerGradientAccumulationHandler, self).__init__()
        self.grad_interval = grad_interval
        self.batch_size = batch_size
        self.rescale_loss = rescale_loss

    def _update_gradient(self, estimator):
        estimator.trainer.step(float(self.loss_denom) /
                               self.batch_size /self.rescale_loss)
        params = estimator.net.collect_params()
        params.zero_grad()
        self.loss_denom = 0

    def train_begin(self, estimator, *args, **kwargs):
        params = estimator.net.collect_params()
        params.setattr('grad_req', 'add')
        params.zero_grad()

    def epoch_begin(self, estimator, *args, **kwargs):
        self.batch_id = 0
        self.loss_denom = 0

    def batch_end(self, estimator, *args, **kwargs):
        self.loss_denom += estimator.tgt_valid_length
        if self.batch_id % self.grad_interval == self.grad_interval - 1:
            self._update_gradient(estimator)
        self.batch_id += 1

    def epoch_end(self, estimator, *args, **kwargs):
        if self.loss_denom > 0:
            self._update_gradient(estimator)

class MTTransformerMetricHandler(MetricHandler, BatchBegin):
    def __init__(self, grad_interval, *args, **kwargs):
        super(MTTransformerMetricHandler, self).__init__(*args, **kwargs)
        self.grad_interval = grad_interval

    def epoch_begin(self, estimator, *args, **kwargs):
        self.batch_id = 0
        for metric in self.metrics:
            metric.reset()

    def batch_begin(self, estimator, *args, **kwargs):
        if self.batch_id % self.grad_interval == 0:
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

# A temporary workaround for computing the bleu function. After bleu is in the metric
# api, this event handler could be removed.
class ComputeBleuHandler(BatchEnd, EpochEnd):
    def __init__(self,
                 tgt_vocab,
                 tgt_sentence,
                 translator,
                 compute_bleu_fn,
                 tokenized,
                 tokenizer,
                 split_compound_word,
                 bpe,
                 bleu,
                 detokenizer,
                 _bpe_to_words):
        self.tgt_vocab = tgt_vocab
        self.tgt_sentence = tgt_sentence
        self.translator = translator
        self.compute_bleu_fn = compute_bleu_fn
        self.tokenized = tokenized
        self.tokenizer = tokenizer
        self.split_compound_word = split_compound_word
        self.bpe = bpe
        self.bleu = bleu
        self.detokenizer = detokenizer
        self._bpe_to_words = _bpe_to_words

        self.all_inst_ids = []
        self.translation_out = []

    def batch_end(self, estimator, *args, **kwargs):
        batch = kwargs['batch']
        label = kwargs['label']
        src_seq, tgt_seq, src_valid_length, tgt_valid_length, inst_ids = batch
        self.all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())
        samples, _, sample_valid_length = self.translator.translate(
            src_seq=src_seq, src_valid_length=src_valid_length)
        max_score_sample = samples[:, 0, :].asnumpy()
        sample_valid_length = sample_valid_length[:, 0].asnumpy()
        for i in range(max_score_sample.shape[0]):
            self.translation_out.append(
                [self.tgt_vocab.idx_to_token[ele] for ele in
                 max_score_sample[i][1:(sample_valid_length[i] - 1)]])
        
    def epoch_end(self, estimator, *args, **kwargs):
        real_translation_out = [None for _ in range(len(all_inst_ids))]
        for ind, sentence in zip(self.all_inst_ids, self.translation_out):
            if self.bleu == 'tweaked':
                real_translation_out[ind] = sentence
            elif self.bleu == '13a' or self.bleu == 'intl':
                real_translation_out[ind] = self.detokenizer(self._bpe_to_words(sentence))
            else:
                raise NotImplementedError
        estimator.bleu_score, _, _, _, _ = self.compute_bleu_fn([self.tgt_sentence],
                                                           real_translation_out,
                                                           tokenized=self.tokenized,
                                                           tokenizer=self.tokenizer,
                                                           split_compound_word=self.split_compound_word,
                                                           bpe=self.bpe)


# temporary validation bleu metric hack, it can be removed once bleu metric api is available
class ValBleuHandler(EpochEnd):
    def __init__(self, val_data,
                 val_tgt_vocab,
                 val_tgt_sentences,
                 translator,
                 tokenized,
                 tokenizer,
                 split_compound_word,
                 bpe,
                 compute_bleu_fn,
                 bleu,
                 detokenizer,
                 _bpe_to_words):
        self.val_data = val_data
        self.val_tgt_vocab = val_tgt_vocab
        self.val_tgt_sentences = val_tgt_sentences
        self.translator = translator
        self.tokenized = tokenized
        self.tokenizer = tokenizer
        self.split_compound_word = split_compound_word
        self.bpe = bpe
        self.compute_bleu_fn = compute_bleu_fn
        self.bleu = bleu
        self.detokenizer = detokenizer
        self._bpe_to_words = _bpe_to_words

    def epoch_end(self, estimator, *args, **kwargs):
        translation_out = []
        all_inst_ids = []
        for  _, (src_seq, tgt_seq, src_valid_length, tgt_valid_length, inst_ids) \
             in enumerate(self.val_data):
            src_seq = src_seq.as_in_context(estimator.context[0])
            tgt_seq = tgt_seq.as_in_context(estimator.context[0])
            src_valid_length = src_valid_length.as_in_context(estimator.context[0])
            tgt_valid_length = tgt_valid_length.as_in_context(estimator.context[0])
            all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())
            samples, _, sample_valid_length = self.translator.translate(
                src_seq=src_seq, src_valid_length=src_valid_length)
            max_score_sample = samples[:, 0, :].asnumpy()
            sample_valid_length = sample_valid_length[:, 0].asnumpy()
            for i in range(max_score_sample.shape[0]):
                translation_out.append(
                    [self.val_tgt_vocab.idx_to_token[ele] for ele in
                     max_score_sample[i][1:(sample_valid_length[i] - 1)]])
        real_translation_out = [None for _ in range(len(all_inst_ids))]
        for ind, sentence in zip(all_inst_ids, translation_out):
            if self.bleu == 'tweaked':
                real_translation_out[ind] = sentence
            elif self.bleu == '13a' or self.beu == 'intl':
                real_translation_out[ind] = self.detokenizer(self._bpe_to_words(sentence))
            else:
                raise NotImplementedError
        estimator.bleu, _, _, _, _ = self.compute_bleu_fn([self.val_tgt_sentences],
                                                          real_translation_out,
                                                          tokenized=self.tokenized,
                                                          tokenizer=self.tokenizer,
                                                          split_compound_word=self.split_compound_word,
                                                          bpe=self.bpe)
