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

# os.environ['GLUON_MIN'] = '8'
sys.path.insert(0, '/home/ubuntu/gluon-nlp/src/')
sys.path.insert(0, '/home/ubuntu/mxnet/python/')

import argparse
import random
import logging
import numpy as np
import mxnet as mx
import time
from mxnet import gluon
from mxnet.gluon.data import ArrayDataset, DataLoader
from gluonnlp.utils import clip_grad_global_norm
from gluonnlp.model import bert_12_768_12
from gluonnlp.data import SimpleDatasetStream, SplitSampler, NumpyDataset
from tokenizer import FullTokenizer
from dataset import MRPCDataset, ClassificationTransform

parser = argparse.ArgumentParser(description='BERT pretraining example.')
parser.add_argument('--num_steps', type=int, default=1000, help='Number of optimization steps')
parser.add_argument('--dtype', type=str, default='float32', help='data dtype')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU.')
parser.add_argument('--dataset_name', type=str, default='book_corpus_wiki_en_uncased',
                    help='The dataset from which the vocabulary is created. '
                         'Options include book_corpus_wiki_en_uncased, book_corpus_wiki_en_cased. '
                         'Default is book_corpus_wiki_en_uncased')
parser.add_argument('--load_ckpt', type=str, default=None, help='Load model from a checkpoint.')
parser.add_argument('--pretrained', action='store_true',
                    help='Load the pretrained model released by Google.')
parser.add_argument('--data', type=str, required=True,
                    help='Path to training data.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--warmup_ratio', type=float, default=0.1,
                    help='ratio of warmup steps used in NOAM\'s stepsize schedule')
parser.add_argument('--log_interval', type=int, default=10, help='report interval')
parser.add_argument('--max_len', type=int, default=512, help='Maximum length of the sentence pairs')
parser.add_argument('--gpu', action='store_true', help='Whether to use GPU')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--do-training', action='store_true',
                    help='Whether to do training on the training set.')
args = parser.parse_args()
# logging
logging.getLogger().setLevel(logging.INFO)
logging.info(args)

def get_model(ctx):
    # model
    pretrained = args.pretrained
    dataset = args.dataset_name
    # TODO API
    model, vocabulary = bert_12_768_12(dataset_name=dataset,
                                       pretrained=pretrained, ctx=ctx, use_pooler=True,
                                       use_decoder=True, use_classifier=True,
                                       for_pretrain=True)
    # load from checkpoint
    if pretrained and args.load_ckpt:
        raise UserWarning('Both pretrained and load_ckpt are set. Do you intend to load from '
                          'the checkpoint instead of the pretrained model from Google?')
    if args.load_ckpt:
        raise NotImplementedError()

    model.cast(args.dtype)
    # TODO skip init?
    model.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    #model.hybridize(static_alloc=True)
    logging.debug(model)

    # losses
    nsp_loss = gluon.loss.SoftmaxCELoss()
    mlm_loss = gluon.loss.SoftmaxCELoss()
    nsp_loss.hybridize(static_alloc=True, static_shape=True)
    mlm_loss.hybridize(static_alloc=True, static_shape=True)

    return model, nsp_loss, mlm_loss, vocabulary

def get_dataset(data):
    mx.nd.waitall()
    t0 = time.time()
    data_train = NumpyDataset(data)
    t1 = time.time()
    logging.info('Loading {} took {:.3f}s'.format(data, t1-t0))
    logging.info(data_train.keys)
    return data_train

def get_dataloader(data):
    dataloader = DataLoader(data, batch_size=args.batch_size,
                                 shuffle=True, last_batch='rollover')
    # prefetch iter
    # preload to GPU
    return dataloader

def train(data_train, model, nsp_loss, mlm_loss, vocab_size, ctx):
    """Training function."""
    mlm_metric = mx.metric.Accuracy()
    nsp_metric = mx.metric.Accuracy()
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

    begin_time = time.time()
    step_mlm_loss = 0
    step_nsp_loss = 0
    batch_num = 0
    step_num = 0
    while step_num < num_train_steps:
        for _, data in enumerate(data_train):
            if step_num >= num_train_steps:
                break
            step_num += 1
            # update learning rate
            if step_num < num_warmup_steps:
                new_lr = lr * step_num / num_warmup_steps
            else:
                offset = (step_num - num_warmup_steps) * lr / (num_train_steps - num_warmup_steps)
                new_lr = lr - offset
            trainer.set_learning_rate(new_lr)

            def as_in_ctx(arrs, ctx):
                if isinstance(arrs, list):
                    return [arr.as_in_context(ctx) for arr in arrs]
                return arrs.as_in_context(ctx)

            data = as_in_ctx(data, ctx[0])
            with mx.autograd.record():
                input_id, masked_lm_id, masked_lm_position, masked_lm_weight, next_sentence_label, segment_id, valid_length = data
                valid_length = valid_length.astype('float32')
                classified, decoded = model(input_id, segment_id, valid_length=valid_length, positions=masked_lm_position)
                decoded = decoded.reshape((-1, vocab_size))
                masked_lm_id = masked_lm_id.reshape((-1))
                ls1 = mlm_loss(decoded, masked_lm_id) * masked_lm_weight.reshape((-1))
                ls2 = nsp_loss(classified, next_sentence_label)
                ls1 = ls1.mean()
                ls2 = ls2.mean()
                ls = ls1 + ls2
            mx.autograd.backward(ls)
            trainer.allreduce_grads()
            clip_grad_global_norm(params, 1)
            trainer.update(1)
            step_mlm_loss += ls1.asscalar()
            step_nsp_loss += ls2.asscalar()
            nsp_metric.update(next_sentence_label, classified)
            mlm_metric.update(masked_lm_id, decoded)
            if (step_num + 1) % (args.log_interval) == 0:
                #if args.profile:
                #    if batch_id + 1 == args.log_interval:
                #        mx.nd.waitall()
                #        mx.profiler.set_config(profile_memory=False,profile_symbolic=True, aggregate_stats=True, profile_all=True, filename='profile_output.json')
                #        mx.profiler.set_state('run')
                #    elif batch_id + 1 == args.log_interval * 2:
                #        mx.nd.waitall()
                #        mx.profiler.set_state('stop')
                #        print(mx.profiler.dumps())
                #        exit()
                end_time = time.time()
                duration = end_time - begin_time
                throughput = args.log_interval * batch_size * args.max_len / 1000.0 / duration
                step_mlm_loss /= args.log_interval
                step_nsp_loss /= args.log_interval
                logging.info('[step {}]\tmlm_loss={:.3f}\tmlm_acc={:.1f}\tnsp_loss={:.3f}\tnsp_acc={:.1f}\tthroughput={:.1f}K tks/s\tlr={:.7f}'
                             .format(step_num, step_mlm_loss, mlm_metric.get()[1] * 100, step_nsp_loss,
                                     nsp_metric.get()[1] * 100, throughput, trainer.learning_rate))
                begin_time = end_time
                step_mlm_loss = 0
                step_nsp_loss = 0
        mx.nd.waitall()

if __name__ == '__main__':
    # random seed
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    mx.random.seed(seed)


    ctx = [mx.gpu() if args.gpu else mx.cpu()]
    model, nsp_loss, mlm_loss, vocabulary = get_model(ctx)

    batch_size = args.batch_size * len(ctx)

    do_lower_case = 'uncased' in args.dataset_name
    tokenizer = FullTokenizer(vocabulary, do_lower_case=do_lower_case)
    dataset_train = get_dataset(args.data)
    data_train = get_dataloader(dataset_train)
    if args.do_training:
        train(data_train, model, nsp_loss, mlm_loss, len(tokenizer.vocab), ctx)
