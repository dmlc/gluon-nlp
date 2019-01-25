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
import gluonnlp as nlp
from gluonnlp.utils import clip_grad_global_norm, Parallelizable, Parallel
from gluonnlp.metric import MaskedAccuracy
from gluonnlp.model import bert_12_768_12
from gluonnlp.data.batchify import Tuple, Stack, Pad
from gluonnlp.data import SimpleDatasetStream, FixedBucketSampler, NumpyDataset
from tokenizer import FullTokenizer
from dataset import MRPCDataset, ClassificationTransform

parser = argparse.ArgumentParser(description='BERT pretraining example.')
parser.add_argument('--num_steps', type=int, default=20, help='Number of optimization steps')
parser.add_argument('--num_buckets', type=int, default=1,
                    help='Number of buckets for variable length sequence sampling')
parser.add_argument('--dtype', type=str, default='float32', help='data dtype')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU.')
parser.add_argument('--accumulate', type=int, default=1, help='Number of batches for gradient accumulation.')
parser.add_argument('--batch_size_eval', type=int, default=8, help='Batch size per GPU for evaluation.')
parser.add_argument('--dataset_name', type=str, default='book_corpus_wiki_en_uncased',
                    help='The dataset from which the vocabulary is created. '
                         'Options include book_corpus_wiki_en_uncased, book_corpus_wiki_en_cased. '
                         'Default is book_corpus_wiki_en_uncased')
parser.add_argument('--pretrained', action='store_true',
                    help='Load the pretrained model released by Google.')
parser.add_argument('--data', type=str, default=None,
                    help='Path to training data.')
parser.add_argument('--ckpt_dir', type=str, default=None,
                    help='Path to checkpoint directory')
parser.add_argument('--start_step', type=int, default=0, help='Start optimization step from the checkpoint.')
parser.add_argument('--data_eval', type=str, default=None,
                    help='Path to evaluation data.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--warmup_ratio', type=float, default=0.1,
                    help='ratio of warmup steps used in NOAM\'s stepsize schedule')
parser.add_argument('--log_interval', type=int, default=10, help='Report interval')
parser.add_argument('--ckpt_interval', type=int, default=10, help='Checkpoint interval')
parser.add_argument('--gpus', type=str, default='0', help='List of GPUs to use. e.g. 1,3')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--verbose', action='store_true', help='verbose logging')
parser.add_argument('--profile', action='store_true', help='profile the program')
parser.add_argument('--do-training', action='store_true',
                    help='Whether to do training on the training set.')
parser.add_argument('--do-eval', action='store_true',
                    help='Whether to do evaluation on the eval set.')
args = parser.parse_args()

# logging
level = logging.DEBUG if args.verbose else logging.INFO
logging.getLogger().setLevel(level)
logging.info(args)

def get_model(ctx):
    # model
    pretrained = args.pretrained
    dataset = args.dataset_name
    model, vocabulary = bert_12_768_12(dataset_name=dataset,
                                       pretrained=pretrained, ctx=ctx)
    if not pretrained:
        model.initialize(init=mx.init.Normal(0.02), ctx=ctx)

    if args.ckpt_dir and args.start_step:
        param_path = os.path.join(args.ckpt_dir, '%07d.params'%args.start_step)
        model.load_parameters(param_path, ctx=ctx)
        logging.info('Loading step %d checkpoints from %s.'%(args.start_step, param_path))

    model.cast(args.dtype)
    model.hybridize(static_alloc=True)

    # losses
    nsp_loss = gluon.loss.SoftmaxCELoss()
    mlm_loss = gluon.loss.SoftmaxCELoss()
    nsp_loss.hybridize(static_alloc=True)
    mlm_loss.hybridize(static_alloc=True)

    return model, nsp_loss, mlm_loss, vocabulary

def get_dataset(data, batch_size, is_train):
    data = data
    # numpy
    stream = SimpleDatasetStream(NumpyDataset, data)

    def get_dataloader(dataset):
        t0 = time.time()
        lengths = dataset.get_field('valid_lengths')
        logging.debug('Num samples = %d'%len(lengths))
        sampler = FixedBucketSampler(lengths,
                                     batch_size=batch_size,
                                     num_buckets=args.num_buckets,
                                     ratio=0,
                                     shuffle=is_train)
        batchify_fn = Tuple(Pad(), Pad(), Pad(), Pad(), Stack(), Pad(), Stack())
        dataloader = DataLoader(dataset=dataset,
                                batch_sampler=sampler,
                                batchify_fn=batchify_fn,
                                num_workers=0)
        t1 = time.time()
        logging.debug('Dataloader creation cost = %.2f s'%(t1-t0))
        logging.debug('Batch Sampler:\n%s', sampler.stats())
        return dataloader

    stream = stream.transform(get_dataloader)
    return stream

def split_and_load(arrs, ctx):
    assert isinstance(arrs, (list, tuple))
    if len(ctx) == 1:
        return [[arr.as_in_context(ctx[0]) for arr in arrs]]
    else:
        # split and load
        loaded_arrs = [gluon.utils.split_and_load(arr, ctx, even_split=False) for arr in arrs]
        return zip(*loaded_arrs)

class ParallelBERT(Parallelizable):
    """Data parallel BERT model.

    Parameters
    ----------
    model : Block
        The BERT model.
    """
    def __init__(self, model, mlm_loss, nsp_loss, vocab_size):
        self._model = model
        self._mlm_loss = mlm_loss
        self._nsp_loss = nsp_loss
        self._vocab_size = vocab_size

    def forward_backward(self, x):
        input_id, masked_id, masked_position, masked_weight, next_sentence_label, segment_id, valid_length = x
        num_masks = masked_weight.sum() + 1e-8
        valid_length = valid_length.reshape(-1)
        valid_length = valid_length.astype('float32', copy=False)
        masked_id = masked_id.reshape(-1)
        with mx.autograd.record():
            _, _, classified, decoded = self._model(input_id, segment_id, valid_length, masked_position)
            decoded = decoded.reshape((-1, self._vocab_size))
            ls1 = self._mlm_loss(decoded, masked_id, masked_weight.reshape((-1, 1))).sum() / num_masks
            ls2 = self._nsp_loss(classified, next_sentence_label).mean()
            ls = ls1 + ls2
        ls.backward()
        return ls, next_sentence_label, classified, masked_id, decoded, masked_weight, ls1, ls2, valid_length

def forward(data, model, mlm_loss, nsp_loss, vocab_size):
    input_id, masked_id, masked_position, masked_weight, next_sentence_label, segment_id, valid_length = data
    num_masks = masked_weight.sum() + 1e-8
    valid_length = valid_length.reshape(-1)
    valid_length = valid_length.astype('float32', copy=False)
    _, _, classified, decoded = model(input_id, segment_id, valid_length, masked_position)
    masked_id = masked_id.reshape(-1)
    decoded = decoded.reshape((-1, vocab_size))
    ls1 = mlm_loss(decoded, masked_id, masked_weight.reshape((-1, 1))).sum() / num_masks
    ls2 = nsp_loss(classified, next_sentence_label).mean()
    ls = ls1 + ls2
    return ls, next_sentence_label, classified, masked_id, decoded, masked_weight, ls1, ls2, valid_length

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
    local_num_tks = 0
    for _, dataloader in enumerate(data_eval):
        for _, data in enumerate(dataloader):
            step_num += 1

            data_list = split_and_load(data, ctx)
            loss_list = []
            ns_label_list, ns_pred_list = [], []
            mask_label_list, mask_pred_list, mask_weight_list = [], [], []
            for data in data_list:
                out = forward(data, model, mlm_loss, nsp_loss, vocab_size)
                ls, next_sentence_label, classified, masked_id, decoded, masked_weight, ls1, ls2, valid_length = out
                loss_list.append(ls)
                ns_label_list.append(next_sentence_label)
                ns_pred_list.append(classified)
                mask_label_list.append(masked_id)
                mask_pred_list.append(decoded)
                mask_weight_list.append(masked_weight)

                local_mlm_loss += ls1.as_in_context(mx.cpu())
                local_nsp_loss += ls2.as_in_context(mx.cpu())
                local_num_tks += valid_length.sum().as_in_context(mx.cpu())
            nsp_metric.update(ns_label_list, ns_pred_list)
            mlm_metric.update(mask_label_list, mask_pred_list, mask_weight_list)

            # logging
            if (step_num + 1) % (args.log_interval) == 0:
                total_mlm_loss += local_mlm_loss
                total_nsp_loss += local_nsp_loss
                log(begin_time, local_num_tks, local_mlm_loss, local_nsp_loss, step_num, mlm_metric, nsp_metric, None)
                begin_time = time.time()
                local_mlm_loss = local_nsp_loss = local_num_tks = 0
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

def log(begin_time, local_num_tks, local_mlm_loss, local_nsp_loss, step_num, mlm_metric, nsp_metric, trainer):
    end_time = time.time()
    duration = end_time - begin_time
    throughput = local_num_tks / 1000.0 / duration
    local_mlm_loss = local_mlm_loss / args.log_interval
    local_nsp_loss = local_nsp_loss / args.log_interval
    lr = trainer.learning_rate if trainer else 0
    logging.info('[step {}]\tmlm_loss={:.5f}\tmlm_acc={:.5f}\tnsp_loss={:.5f}\tnsp_acc={:.3f}\tthroughput={:.1f}K tks/s\tlr={:.7f} time={:.2f}'
                 .format(step_num, local_mlm_loss.asscalar(), mlm_metric.get()[1] * 100, local_nsp_loss.asscalar(),
                         nsp_metric.get()[1] * 100, throughput.asscalar(), lr, duration))


def train(data_train, model, nsp_loss, mlm_loss, vocab_size, ctx):
    """Training function."""
    mlm_metric = MaskedAccuracy()
    nsp_metric = MaskedAccuracy()
    mlm_metric.reset()
    nsp_metric.reset()

    lr = args.lr
    trainer = gluon.Trainer(model.collect_params(), 'bertadam',
                            {'learning_rate': lr, 'epsilon': 1e-6, 'wd': 0.01}, update_on_kvstore=False)

    if args.ckpt_dir and args.start_step:
        trainer.load_states(os.path.join(args.ckpt_dir, '%07d.states'%args.start_step))

    accumulate = args.accumulate
    num_train_steps = args.num_steps
    warmup_ratio = args.warmup_ratio
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    params = [p for p in model.collect_params().values() if p.grad_req != 'null']

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    for p in params:
        p.grad_req = 'add'

    train_begin_time = time.time()
    begin_time = time.time()
    local_mlm_loss = 0
    local_nsp_loss = 0
    local_num_tks = 0
    batch_num = 0
    step_num = args.start_step

    parallel_model = ParallelBERT(model, mlm_loss, nsp_loss, vocab_size)
    parallel = Parallel(len(ctx), parallel_model)

    while step_num < num_train_steps:
        for _, dataloader in enumerate(data_train):
            iterator = enumerate(dataloader)
            _, data_batch = next(iterator)
            while True:
            #for _, data_batch in enumerate(dataloader):
                if batch_num % accumulate == 0:
                    step_num += 1
                    # zero grad
                    model.collect_params().zero_grad()
                    # update learning rate
                    if step_num <= num_warmup_steps:
                        new_lr = lr * step_num / num_warmup_steps
                    else:
                        offset = (step_num - num_warmup_steps) * lr / (num_train_steps - num_warmup_steps)
                        new_lr = lr - offset
                    trainer.set_learning_rate(new_lr)
                    if args.profile:
                        if step_num == 2:
                            mx.nd.waitall()
                            mx.profiler.set_config(profile_memory=False, profile_symbolic=True,
                                                   profile_imperative=True, filename='profile.json',
                                                   aggregate_stats=True)
                            mx.profiler.set_state('run')
                        if step_num == 3:
                            mx.nd.waitall()
                            mx.profiler.set_state('stop')
                            logging.info(mx.profiler.dumps())
                            mx.profiler.dump()
                            exit()
                if data_batch[0].shape[0] < len(ctx):
                    continue
                data_list = split_and_load(data_batch, ctx)
                loss_list = []
                ns_label_list, ns_pred_list = [], []
                mask_label_list, mask_pred_list, mask_weight_list = [], [], []

                # parallel forward / backward
                for data in data_list:
                    parallel.put(data)
                for _ in range(len(ctx)):
                    ls, next_sentence_label, classified, masked_id, decoded, masked_weight, ls1, ls2, valid_length = parallel.get()
                    loss_list.append(ls)
                    ns_label_list.append(next_sentence_label)
                    ns_pred_list.append(classified)
                    mask_label_list.append(masked_id)
                    mask_pred_list.append(decoded)
                    mask_weight_list.append(masked_weight)
                    local_mlm_loss += ls1.as_in_context(mx.cpu())
                    local_nsp_loss += ls2.as_in_context(mx.cpu())
                    local_num_tks += valid_length.sum().as_in_context(mx.cpu())

                # update
                if (batch_num + 1) % accumulate == 0:
                    trainer.allreduce_grads()
                    clip_grad_global_norm(params, 1, check_isfinite=False)
                    trainer.update(1)
                nsp_metric.update(ns_label_list, ns_pred_list)
                mlm_metric.update(mask_label_list, mask_pred_list, mask_weight_list)
                # logging
                if (step_num + 1) % (args.log_interval) == 0 and (batch_num + 1) % accumulate == 0:
                    print(local_num_tks.asscalar())
                    log(begin_time, local_num_tks, local_mlm_loss / accumulate,
                        local_nsp_loss / accumulate, step_num, mlm_metric, nsp_metric, trainer)
                    begin_time = time.time()
                    local_mlm_loss = local_nsp_loss = local_num_tks = 0
                    mlm_metric.reset_local()
                    nsp_metric.reset_local()

                # saving checkpoints
                if args.ckpt_dir and (step_num + 1) % (args.ckpt_interval) == 0 and (batch_num + 1) % accumulate == 0:
                    param_path = os.path.join(args.ckpt_dir, '%07d.params'%step_num)
                    trainer_path = os.path.join(args.ckpt_dir, '%07d.states'%step_num)
                    logging.info('[step %d] Saving checkpoints to %s, %s.'%(step_num, param_path, trainer_path))
                    model.save_parameters(param_path)
                    trainer.save_states(trainer_path)
                batch_num += 1
    mx.nd.waitall()
    train_end_time = time.time()
    logging.info('Train cost={:.1f}s'.format(train_end_time - train_begin_time))

if __name__ == '__main__':
    # random seed
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    mx.random.seed(seed)

    ctx = [mx.cpu()] if args.gpus is None or args.gpus == '' else \
          [mx.gpu(int(x)) for x in args.gpus.split(',')]

    model, nsp_loss, mlm_loss, vocabulary = get_model(ctx)

    batch_size = args.batch_size * len(ctx)
    batch_size_eval = args.batch_size_eval * len(ctx)

    do_lower_case = 'uncased' in args.dataset_name
    tokenizer = FullTokenizer(vocabulary, do_lower_case=do_lower_case)

    if args.ckpt_dir:
        ckpt_dir = os.path.expanduser(args.ckpt_dir)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

    if args.do_training:
        assert args.data, '--data must be provided for training'
        data_train = get_dataset(args.data, batch_size, True)
        train(data_train, model, nsp_loss, mlm_loss, len(tokenizer.vocab), ctx)
    if args.do_eval:
        assert args.data_eval, '--data_eval must be provided for evaluation'
        data_eval = get_dataset(args.data_eval, batch_size_eval, False)
        evaluate(data_eval, model, nsp_loss, mlm_loss, len(tokenizer.vocab), ctx)
