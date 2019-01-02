"""
Pre-training Bidirectional Encoder Representations from Transformers
=========================================================================================
This example shows how to pre-train a BERT model with Gluon NLP Toolkit.
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
"""

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

import sys
import os
import argparse
import random
import logging
import numpy as np
import mxnet as mx
import time
from mxnet import gluon
from mxnet.gluon.data import ArrayDataset, DataLoader
from gluonnlp.utils import clip_grad_global_norm
from gluonnlp.model import bert_12_768_12, bert_pretraining_12_768_12
from gluonnlp.data import SimpleDatasetStream, SplitSampler, NumpyDataset
from tokenizer import FullTokenizer
from dataset import MRPCDataset, ClassificationTransform

parser = argparse.ArgumentParser(description='BERT pretraining example.')
parser.add_argument('--num_steps', type=int, default=1000, help='Number of optimization steps')
parser.add_argument('--dtype', type=str, default='float32', help='data dtype')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU.')
parser.add_argument('--batch_size_eval', type=int, default=8, help='Batch size per GPU for evaluation.')
parser.add_argument('--dataset_name', type=str, default='book_corpus_wiki_en_uncased',
                    help='The dataset from which the vocabulary is created. '
                         'Options include book_corpus_wiki_en_uncased, book_corpus_wiki_en_cased. '
                         'Default is book_corpus_wiki_en_uncased')
parser.add_argument('--load_ckpt', type=str, default=None, help='Load model from a checkpoint.')
parser.add_argument('--pretrained', action='store_true',
                    help='Load the pretrained model released by Google.')
parser.add_argument('--data', type=str, default=None,
                    help='Path to training data.')
parser.add_argument('--data_eval', type=str, default=None,
                    help='Path to evaluation data.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--warmup_ratio', type=float, default=0.1,
                    help='ratio of warmup steps used in NOAM\'s stepsize schedule')
parser.add_argument('--log_interval', type=int, default=10, help='report interval')
parser.add_argument('--max_len', type=int, default=512,
                    help='Maximum length of the sentence pairs. Default is 512')
#parser.add_argument('--gpu', action='store_true', help='Whether to use GPU')
parser.add_argument('--gpu', type=int, required=True,help='Whether to use GPU')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--do-training', action='store_true',
                    help='Whether to do training on the training set.')
parser.add_argument('--do-eval', action='store_true',
                    help='Whether to do evaluation on the eval set.')
args = parser.parse_args()
# logging
logging.getLogger().setLevel(logging.INFO)
logging.info(args)

def get_model(ctx):
    # model
    pretrained = args.pretrained
    dataset = args.dataset_name
    # TODO API
    model, vocabulary = bert_pretraining_12_768_12(dataset_name=dataset,
                                                   pretrained=pretrained, ctx=ctx)
    # load from checkpoint
    if pretrained and args.load_ckpt:
        raise UserWarning('Both pretrained and load_ckpt are set. Do you intend to load from '
                          'the checkpoint instead of the pretrained model from Google?')
    if args.load_ckpt:
        raise NotImplementedError()

    if not pretrained:
        model.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    model.cast(args.dtype)
    model.hybridize(static_alloc=True)
    logging.debug(model)

    # losses
    nsp_loss = gluon.loss.SoftmaxCELoss()
    mlm_loss = gluon.loss.SoftmaxCELoss()
    nsp_loss.hybridize(static_alloc=True)
    mlm_loss.hybridize(static_alloc=True)

    return model, nsp_loss, mlm_loss, vocabulary

def get_dataset(data):
    t0 = time.time()
    data_train = NumpyDataset(data)
    t1 = time.time()
    logging.debug('Loading {} took {:.3f}s'.format(data, t1-t0))
    return data_train

def get_dataloader(data, batch_size, evaluate):
    if evaluate:
        dataloader = DataLoader(data, batch_size=batch_size,
                                shuffle=False, last_batch='keep')
    else:
        dataloader = DataLoader(data, batch_size=batch_size,
                                shuffle=True, last_batch='rollover')
    return dataloader

def as_in_ctx(arrs, ctx):
    if isinstance(arrs, list):
        return [arr.as_in_context(ctx) for arr in arrs]
    return arrs.as_in_context(ctx)

@mx.metric.register
@mx.metric.alias('masked_acc')
# TODO update meth
class MaskedAccuracy(mx.metric.EvalMetric):
    """Computes accuracy classification score.
    The accuracy score is defined as

    .. math::
        \\text{accuracy}(y, \\hat{y}) = \\frac{1}{n} \\sum_{i=0}^{n-1}
        \\text{1}(\\hat{y_i} == y_i)

    Parameters
    ----------
    axis : int, default=1
        The axis that represents classes
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.

    Examples
    --------
    >>> predicts = [mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.nd.array([0, 1, 1])]
    >>> acc = mx.metric.Accuracy()
    >>> acc.update(preds = predicts, labels = labels)
    >>> print acc.get()
    ('accuracy', 0.6666666666666666)
    """
    def __init__(self, axis=1, name='masked-accuracy',
                 output_names=None, label_names=None):
        super(MaskedAccuracy, self).__init__(
            name, axis=axis,
            output_names=output_names, label_names=label_names,
            has_global_stats=True)
        self.axis = axis

    # TODO update doc
    def update(self, labels, preds, masks=None):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data with class indices as values, one per sample.
        preds : list of `NDArray`
            Prediction values for samples. Each prediction value can either be the class index,
            or a vector of likelihoods for all classes.
        """
        labels, preds = mx.metric.check_label_shapes(labels, preds, True)

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.astype('int32', copy=False).reshape((-1,))
            label = label.astype('int32', copy=False).reshape((-1,))
            # flatten before checking shapes to avoid shape miss match
            mx.metric.check_label_shapes(label, pred_label)

            if masks is not None:
                masks = masks.astype('int32', copy=False).reshape((-1,))
                mx.metric.check_label_shapes(label, masks)
                num_correct = ((pred_label == label) * masks).sum().asscalar()
                num_inst =  masks.sum().asscalar()
            else:
                num_correct = (pred_label == label).sum().asscalar()
                num_inst =  len(label)
            self.sum_metric += num_correct
            self.global_sum_metric += num_correct
            self.num_inst += num_inst
            self.global_num_inst += num_inst

def evaluate(data_eval, model, nsp_loss, mlm_loss, vocab_size, ctx):
    """Evaluation function."""
    mlm_metric = MaskedAccuracy()
    nsp_metric = MaskedAccuracy()
    mlm_metric.reset()
    nsp_metric.reset()

    eval_begin_time = time.time()
    begin_time = time.time()
    step_num = 0
    local_mlm_loss = local_nsp_loss = 0
    total_mlm_loss = total_nsp_loss = 0
    for _, data in enumerate(data_eval):
        step_num += 1
        data = as_in_ctx(data, ctx[0])
        with mx.autograd.pause():
            input_id, masked_id, masked_position, masked_weight, next_sentence_label, segment_id, valid_length = data
            # avoid divide by zero error
            num_masks = masked_weight.sum() + 1e-8
            valid_length = valid_length.astype('float32', copy=False)
            classified, decoded = model(input_id, segment_id, valid_length=valid_length, positions=masked_position)
            masked_id = masked_id.reshape(-1)
            ls1 = mlm_loss(decoded, masked_id, masked_weight.reshape((-1, 1))).sum() / num_masks
            ls2 = nsp_loss(classified, next_sentence_label).mean()
            ls = ls1 + ls2
        local_mlm_loss += ls1
        local_nsp_loss += ls2
        decoded = decoded.reshape((-1, vocab_size))
        nsp_metric.update(next_sentence_label, classified)
        mlm_metric.update(masked_id, decoded, masked_weight)
        if (step_num + 1) % (args.log_interval) == 0:
            end_time = time.time()
            duration = end_time - begin_time
            throughput = args.log_interval * args.batch_size_eval * args.max_len / 1000.0 / duration
            total_mlm_loss += local_mlm_loss
            total_nsp_loss += local_nsp_loss
            local_mlm_loss /= args.log_interval
            local_nsp_loss /= args.log_interval
            logging.info('[step {}]\tmlm_loss={:.8f}\tmlm_acc={:.8f}\tnsp_loss={:.8f}\tnsp_acc={:.5f}\tthroughput={:.1f}K tks/s\t'
                         .format(step_num, local_mlm_loss.asscalar(), mlm_metric.get()[1] * 100, local_nsp_loss.asscalar(),
                                 nsp_metric.get()[1] * 100, throughput))
            begin_time = end_time
            local_mlm_loss = 0
            local_nsp_loss = 0
            mlm_metric.reset_local()
            nsp_metric.reset_local()

    mx.nd.waitall()
    eval_end_time = time.time()
    total_mlm_loss /= step_num
    total_nsp_loss /= step_num
    logging.info('mlm_loss={:.3f}\tmlm_acc={:.1f}\tnsp_loss={:.3f}\tnsp_acc={:.1f}\t'
                 .format(total_mlm_loss.asscalar(), mlm_metric.get_global()[1] * 100,
                         total_nsp_loss.asscalar(), nsp_metric.get_global()[1] * 100))
    logging.info('Eval cost={:.1f}s'.format(eval_end_time - eval_begin_time))

def train(data_train, model, nsp_loss, mlm_loss, vocab_size, ctx):
    """Training function."""
    mlm_metric = MaskedAccuracy()
    nsp_metric = MaskedAccuracy()
    mlm_metric.reset()
    nsp_metric.reset()

    lr = args.lr
    trainer = gluon.Trainer(model.collect_params(), 'bertadam',
                            {'learning_rate': lr, 'epsilon': 1e-6, 'wd': 0.01})
    num_train_steps = args.num_steps
    warmup_ratio = args.warmup_ratio
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    params = [p for p in model.collect_params().values() if p.grad_req != 'null']

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

    mx.nd.waitall()

    train_begin_time = time.time()
    begin_time = time.time()
    local_mlm_loss = 0
    local_nsp_loss = 0
    step_num = 0
    while step_num < num_train_steps:
        for _, data in enumerate(data_train):
            if step_num >= num_train_steps:
                break
            step_num += 1
            # update learning rate
            if step_num <= num_warmup_steps:
                new_lr = lr * step_num / num_warmup_steps
            else:
                offset = (step_num - num_warmup_steps) * lr / (num_train_steps - num_warmup_steps)
                new_lr = lr - offset
            trainer.set_learning_rate(new_lr)

            data = as_in_ctx(data, ctx[0])
            with mx.autograd.record():
                input_id, masked_id, masked_position, masked_weight, next_sentence_label, segment_id, valid_length = data
                num_masks = masked_weight.sum()
                valid_length = valid_length.astype('float32', copy=False)
                classified, decoded = model(input_id, segment_id, valid_length=valid_length, positions=masked_position)
                masked_id = masked_id.reshape(-1)
                ls1 = mlm_loss(decoded, masked_id, masked_weight.reshape((-1, 1))).sum() / num_masks
                ls2 = nsp_loss(classified, next_sentence_label).mean()
                ls = ls1 + ls2
            mx.autograd.backward(ls)
            local_mlm_loss += ls1
            local_nsp_loss += ls2
            trainer.allreduce_grads()
            clip_grad_global_norm(params, 1)
            trainer.update(1)
            decoded = decoded.reshape((-1, vocab_size))
            nsp_metric.update(next_sentence_label, classified)
            mlm_metric.update(masked_id, decoded, masked_weight)
            if (step_num + 1) % (args.log_interval) == 0:
                end_time = time.time()
                duration = end_time - begin_time
                throughput = args.log_interval * batch_size * args.max_len / 1000.0 / duration
                local_mlm_loss /= args.log_interval
                local_nsp_loss /= args.log_interval
                logging.info('[step {}]\tmlm_loss={:.5f}\tmlm_acc={:.5f}\tnsp_loss={:.5f}\tnsp_acc={:.5f}\tthroughput={:.1f}K tks/s\tlr={:.7f}'
                             .format(step_num, local_mlm_loss.asscalar(), mlm_metric.get()[1] * 100, local_nsp_loss.asscalar(),
                                     nsp_metric.get()[1] * 100, throughput, trainer.learning_rate))
                begin_time = end_time
                local_mlm_loss = 0
                local_nsp_loss = 0
                mlm_metric.reset_local()
                nsp_metric.reset_local()
    mx.nd.waitall()
    train_end_time = time.time()
    logging.info('Train cost={:.1f}s'.format(train_end_time - train_begin_time))

if __name__ == '__main__':
    # random seed
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    mx.random.seed(seed)

    # TODO update me
    ctx = [mx.gpu(args.gpu)]# else mx.cpu()]
    model, nsp_loss, mlm_loss, vocabulary = get_model(ctx)

    batch_size = args.batch_size * len(ctx)

    do_lower_case = 'uncased' in args.dataset_name
    tokenizer = FullTokenizer(vocabulary, do_lower_case=do_lower_case)
    if args.do_training:
        assert args.data, '--data must be provided for training'
        dataset_train = get_dataset(args.data)
        data_train = get_dataloader(dataset_train, args.batch_size, False)
        train(data_train, model, nsp_loss, mlm_loss, len(tokenizer.vocab), ctx)
    if args.do_eval:
        assert args.data_eval, '--data_eval must be provided for evaluation'
        dataset_eval = get_dataset(args.data_eval)
        data_eval = get_dataloader(dataset_eval, args.batch_size_eval, True)
        evaluate(data_eval, model, nsp_loss, mlm_loss, len(tokenizer.vocab), ctx)
