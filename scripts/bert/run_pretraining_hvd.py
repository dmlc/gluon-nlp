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

import os
import argparse
import random
import logging
import time
import numpy as np

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data import DataLoader

import gluonnlp as nlp
from gluonnlp.utils import Parallelizable, Parallel
from gluonnlp.metric import MaskedAccuracy
from gluonnlp.data import SimpleDatasetStream, FixedBucketSampler, NumpyDataset, BERTTokenizer

from utils import profile
from fp16_utils import FP16Trainer
from pretraining_utils import get_model, get_pretrain_dataset, get_dummy_dataloader
from pretraining_utils import save_params, log

parser = argparse.ArgumentParser(description='BERT pretraining example.')
parser.add_argument('--num_steps', type=int, default=20, help='Number of optimization steps')
parser.add_argument('--num_buckets', type=int, default=1,
                    help='Number of buckets for variable length sequence sampling')
parser.add_argument('--dtype', type=str, default='float32', help='data dtype')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU.')
parser.add_argument('--accumulate', type=int, default=1,
                    help='Number of batches for gradient accumulation.')
parser.add_argument('--batch_size_eval', type=int, default=8,
                    help='Batch size per GPU for evaluation.')
parser.add_argument('--dataset_name', type=str, default='book_corpus_wiki_en_uncased',
                    help='The dataset from which the vocabulary is created. '
                         'Options include book_corpus_wiki_en_uncased, book_corpus_wiki_en_cased. '
                         'Default is book_corpus_wiki_en_uncased')
parser.add_argument('--pretrained', action='store_true',
                    help='Load the pretrained model released by Google.')
parser.add_argument('--model', type=str, default='bert_12_768_12',
                    help='Model to run pre-training on. Options are bert_12_768_12, bert_24_1024_16')
parser.add_argument('--data', type=str, default=None, help='Path to training data.')
parser.add_argument('--data_eval', type=str, default=None, help='Path to evaluation data.')
parser.add_argument('--ckpt_dir', type=str, required=True,
                    help='Path to checkpoint directory')
parser.add_argument('--start_step', type=int, default=0,
                    help='Start optimization step from the checkpoint.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--warmup_ratio', type=float, default=0.1,
                    help='ratio of warmup steps used in NOAM\'s stepsize schedule')
parser.add_argument('--log_interval', type=int, default=10, help='Report interval')
parser.add_argument('--ckpt_interval', type=int, default=250000, help='Checkpoint interval')
parser.add_argument('--dummy_data_len', type=int, default=None,
                    help='If provided, a data batch of target sequence length is repeatedly used')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--verbose', action='store_true', help='verbose logging')
parser.add_argument('--profile', type=str, default=None,
                    help='output profiling result to the target file')
parser.add_argument('--use_avg_len', action='store_true',
                    help='Use average length information for the bucket sampler. '
                         'The batch size is now approximately the number of tokens in the batch')
args = parser.parse_args()

# logging
level = logging.DEBUG if args.verbose else logging.INFO
logging.getLogger().setLevel(level)
logging.info(args)
os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'

try:
    import horovod.mxnet as hvd
except ImportError:
    logging.info('horovod must be installed.')
    exit()
hvd.init()
store = None
num_workers = hvd.size()
rank = hvd.rank()
local_rank = hvd.local_rank()
if not args.use_avg_len and hvd.size() > 1:
    logging.info('Specifying --use-avg-len and setting --batch_size with the '
                 'target number of tokens would help improve training throughput.')
logging.debug('local_rank = %d, rank = %d, num_workers = %d',
              local_rank, rank, num_workers)
logging.info(nlp)

def split_and_load(arrs, ctx):
    """split and load arrays to a list of contexts"""
    assert isinstance(arrs, (list, tuple))
    if len(ctx) == 1:
        return [[arr.as_in_context(ctx[0]) for arr in arrs]]
    else:
        # split and load
        loaded_arrs = [gluon.utils.split_and_load(arr, ctx, even_split=False) for arr in arrs]
        return zip(*loaded_arrs)

def forward(data, model, mlm_loss, nsp_loss, vocab_size):
    """forward computation for evaluation"""
    (input_id, masked_id, masked_position, masked_weight, \
     next_sentence_label, segment_id, valid_length) = data
    num_masks = masked_weight.sum() + 1e-8
    valid_length = valid_length.reshape(-1)
    masked_id = masked_id.reshape(-1)
    valid_length_typed = valid_length.astype(args.dtype, copy=False)
    _, _, classified, decoded = model(input_id, segment_id, valid_length_typed,
                                      masked_position)
    decoded = decoded.reshape((-1, vocab_size))
    ls1 = mlm_loss(decoded.astype('float32', copy=False),
                   masked_id, masked_weight.reshape((-1, 1)))
    ls2 = nsp_loss(classified.astype('float32', copy=False), next_sentence_label)
    ls1 = ls1.sum() / num_masks
    ls2 = ls2.mean()
    ls = ls1 + ls2
    return ls, next_sentence_label, classified, masked_id, decoded, \
           masked_weight, ls1, ls2, valid_length.astype('float32', copy=False)

class ParallelBERT(Parallelizable):
    """Data parallel BERT model.

    Parameters
    ----------
    model : Block
        The BERT model.
    """
    def __init__(self, model, mlm_loss, nsp_loss, vocab_size, rescale_factor, trainer=None):
        self._model = model
        self._mlm_loss = mlm_loss
        self._nsp_loss = nsp_loss
        self._vocab_size = vocab_size
        self._rescale_factor = rescale_factor
        self._trainer = trainer

    def forward_backward(self, x):
        """forward backward implementation"""
        with mx.autograd.record():
            (ls, next_sentence_label, classified, masked_id, decoded, \
             masked_weight, ls1, ls2, valid_length) = forward(x, self._model, self._mlm_loss,
                                                              self._nsp_loss, self._vocab_size)
            ls = ls / self._rescale_factor
        if args.dtype == 'float16':
            self._trainer.backward(ls)
        else:
            ls.backward()
        return ls, next_sentence_label, classified, masked_id, decoded, \
               masked_weight, ls1, ls2, valid_length

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
                (ls, next_sentence_label, classified, masked_id,
                 decoded, masked_weight, ls1, ls2, valid_length) = out
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
                log(begin_time, local_num_tks, local_mlm_loss, local_nsp_loss,
                    step_num, mlm_metric, nsp_metric, None, args.log_interval)
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

def train(data_train, model, nsp_loss, mlm_loss, vocab_size, ctx):
    """Training function."""

    hvd.broadcast_parameters(model.collect_params(), root_rank=0)
    logging.debug('Broadcasting parameters to workers')

    mlm_metric = MaskedAccuracy()
    nsp_metric = MaskedAccuracy()
    mlm_metric.reset()
    nsp_metric.reset()

    logging.debug('Creating distributed trainer...')
    lr = args.lr
    optim_params = {'learning_rate': lr, 'epsilon': 1e-6, 'wd': 0.01}
    if args.dtype == 'float16':
        optim_params['multi_precision'] = True

    dynamic_loss_scale = args.dtype == 'float16'
    if dynamic_loss_scale:
        loss_scale_param = {'scale_window': 2000 / num_workers, 'tolerance': 0.01}
    else:
        loss_scale_param = None
    trainer = hvd.DistributedTrainer(model.collect_params(), 'bertadam', optim_params)
    fp16_trainer = FP16Trainer(trainer, dynamic_loss_scale=dynamic_loss_scale,
                               loss_scaler_params=loss_scale_param)

    if args.ckpt_dir and args.start_step:
        trainer.load_states(os.path.join(args.ckpt_dir, '%07d.states'%args.start_step))

    accumulate = args.accumulate
    num_train_steps = args.num_steps
    warmup_ratio = args.warmup_ratio
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    params = [p for p in model.collect_params().values() if p.grad_req != 'null']
    param_dict = model.collect_params()

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0
    if accumulate > 1:
        for p in params:
            p.grad_req = 'add'

    train_begin_time = time.time()
    begin_time = time.time()
    local_mlm_loss = 0
    local_nsp_loss = 0
    local_num_tks = 0
    batch_num = 0
    step_num = args.start_step

    parallel_model = ParallelBERT(model, mlm_loss, nsp_loss, vocab_size,
                                  accumulate, trainer=fp16_trainer)
    num_ctxes = len(ctx)
    parallel = Parallel(num_ctxes, parallel_model)

    logging.debug('Training started')
    while step_num < num_train_steps:
        for _, dataloader in enumerate(data_train):
            if step_num >= num_train_steps:
                break

            # create dummy data loader if needed
            if args.dummy_data_len:
                target_shape = (args.batch_size*num_ctxes, args.dummy_data_len)
                dataloader = get_dummy_dataloader(dataloader, target_shape)

            for _, data_batch in enumerate(dataloader):
                if step_num >= num_train_steps:
                    break
                if batch_num % accumulate == 0:
                    step_num += 1
                    # if accumulate > 1, grad_req is set to 'add', and zero_grad is required
                    if accumulate > 1:
                        param_dict.zero_grad()
                    # update learning rate
                    if step_num <= num_warmup_steps:
                        new_lr = lr * step_num / num_warmup_steps
                    else:
                        offset = lr * step_num / num_train_steps
                        new_lr = lr - offset
                    trainer.set_learning_rate(new_lr)
                    if args.profile:
                        profile(step_num, 10, 14, profile_name=args.profile, skip=local_rank != 0)
                if args.use_avg_len:
                    data_list = [[seq.as_in_context(context) for seq in shard]
                                 for context, shard in zip(ctx, data_batch)]
                else:
                    if data_batch[0].shape[0] < len(ctx):
                        continue
                    data_list = split_and_load(data_batch, ctx)

                ns_label_list, ns_pred_list = [], []
                mask_label_list, mask_pred_list, mask_weight_list = [], [], []

                # parallel forward / backward
                for data in data_list:
                    parallel.put(data)
                for _ in range(len(ctx)):
                    (_, next_sentence_label, classified, masked_id,
                     decoded, masked_weight, ls1, ls2, valid_length) = parallel.get()
                    ns_label_list.append(next_sentence_label)
                    ns_pred_list.append(classified)
                    mask_label_list.append(masked_id)
                    mask_pred_list.append(decoded)
                    mask_weight_list.append(masked_weight)
                    local_mlm_loss += ls1.as_in_context(mx.cpu()) / num_ctxes
                    local_nsp_loss += ls2.as_in_context(mx.cpu()) / num_ctxes
                    local_num_tks += valid_length.sum().as_in_context(mx.cpu())

                # update
                if (batch_num + 1) % accumulate == 0:
                    fp16_trainer.step(1, max_norm=1)
                nsp_metric.update(ns_label_list, ns_pred_list)
                mlm_metric.update(mask_label_list, mask_pred_list, mask_weight_list)
                # logging
                if (step_num + 1) % (args.log_interval) == 0 and (batch_num + 1) % accumulate == 0:
                    log(begin_time, local_num_tks, local_mlm_loss / accumulate,
                        local_nsp_loss / accumulate, step_num, mlm_metric, nsp_metric, trainer, args.log_interval)
                    begin_time = time.time()
                    local_mlm_loss = local_nsp_loss = local_num_tks = 0
                    mlm_metric.reset_local()
                    nsp_metric.reset_local()

                # saving checkpoints
                if args.ckpt_dir and (step_num + 1) % (args.ckpt_interval) == 0 \
                   and (batch_num + 1) % accumulate == 0 and local_rank == 0:
                    save_params(step_num, model, trainer, args.ckpt_dir)
                batch_num += 1
    if local_rank == 0:
        save_params(step_num, model, trainer, args.ckpt_dir)
    mx.nd.waitall()
    train_end_time = time.time()
    logging.info('Train cost={:.1f}s'.format(train_end_time - train_begin_time))

if __name__ == '__main__':
    # random seed
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    mx.random.seed(seed)

    ctx = [mx.gpu(local_rank)]

    model, nsp_loss, mlm_loss, vocabulary = get_model(ctx, args.model, args.pretrained,
                                                      args.dataset_name, args.dtype,
                                                      ckpt_dir=args.ckpt_dir,
                                                      start_step=args.start_step)
    lower = 'uncased' in args.dataset_name
    tokenizer = BERTTokenizer(vocabulary, lower=lower)

    if args.ckpt_dir:
        ckpt_dir = os.path.expanduser(args.ckpt_dir)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

    if args.data:
        data_train = get_pretrain_dataset(args.data, args.batch_size, len(ctx), True,
                                          args.use_avg_len, args.num_buckets,
                                          num_parts=1 if args.dummy_data_len else num_workers,
                                          part_idx=0 if args.dummy_data_len else rank,
                                          prefetch=not args.dummy_data_len)
        train(data_train, model, nsp_loss, mlm_loss, len(tokenizer.vocab), ctx)
    if args.data_eval:
        data_eval = get_dataset(args.data_eval, args.batch_size_eval, len(ctx), False, False, 1)
        evaluate(data_eval, model, nsp_loss, mlm_loss, len(tokenizer.vocab), ctx)
