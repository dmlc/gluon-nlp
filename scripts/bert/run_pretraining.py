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

os.environ['GLUON_MIN'] = '8'
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
from gluonnlp.model import bert_12_768_12
from gluonnlp.data import SimpleDatasetStream
from tokenizer import FullTokenizer
from dataset import MRPCDataset, ClassificationTransform

parser = argparse.ArgumentParser(description='BERT pretraining example.')
parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
parser.add_argument('--dtype', type=str, default='float32', help='data dtype')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU.')
parser.add_argument('--test_batch_size', type=int, default=8, help='Test batch size per GPU')
parser.add_argument('--dataset_name', type=str, default='book_corpus_wiki_en_uncased',
                    help='The dataset from which the vocabulary is created. '
                         'Options include book_corpus_wiki_en_uncased, book_corpus_wiki_en_cased. '
                         'Default is book_corpus_wiki_en_uncased')
parser.add_argument('--optimizer', type=str, default='bertadam', help='optimization algorithm')
parser.add_argument('--load_ckpt', type=str, default=None, help='Load model from a checkpoint.')
parser.add_argument('--pretrained', action='store_true',
                    help='Load the pretrained model released by Google.')
parser.add_argument('--data', type=str, default=None,
                    help='Path to training data.')
parser.add_argument('--eval_data', type=str, default=None, help='Path to eval data.')
parser.add_argument('--kv', type=str, default='device', help='Type of kvstore')
parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate')
parser.add_argument('--warmup_ratio', type=float, default=0.1,
                    help='ratio of warmup steps used in NOAM\'s stepsize schedule')
parser.add_argument('--log_interval', type=int, default=10, help='report interval')
parser.add_argument('--max_len', type=int, default=512, help='Maximum length of the sentence pairs')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--num_gpus', type=int, help='Number of GPUs to use')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--do-training', action='store_true',
                    help='Whether to do training on the training set.')
parser.add_argument('--do-eval', action='store_true',
                    help='Whether to do evaluation on the eval set.')

#parser.add_argument('--profile', action='store_true', help='whether to use gpu for finetuning')
#parser.add_argument('--summary', action='store_true', help='whether to use gpu for finetuning')
args = parser.parse_args()

# random seed
seed = args.seed
np.random.seed(seed)
random.seed(seed)
mx.random.seed(seed)

# logging
logging.getLogger().setLevel(logging.INFO)
logging.info(args)

lr = args.lr

def get_model(args, ctx):
    # model
    pretrained = args.pretrained
    dataset = args.dataset_name
    model, vocabulary = bert_12_768_12(dataset_name=dataset,
                                      pretrained=pretrained, ctx=ctx, use_pooler=True,
                                      use_decoder=True, use_classifier=True, dropout=args.dropout,
                                      embed_dropout=args.dropout, for_pretrain=True)
    # load from checkpoint
    if pretrained and args.load_ckpt:
        raise UserWarning('Both pretrained and load_ckpt are set. Do you intend to load from '
                          'the checkpoint instead of the pretrained model from Google?')
    if args.load_ckpt:
        raise NotImplementedError()

    model.cast(args.dtype)
    # TODO skip init?
    model.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    model.hybridize(static_alloc=True)
    logging.debug(model)

    # losses
    nsp_loss = gluon.loss.SoftmaxCELoss()
    mlm_loss = gluon.loss.SoftmaxCELoss()
    nsp_loss.hybridize(static_alloc=True, static_shape=True)
    mlm_loss.hybridize(static_alloc=True, static_shape=True)

    return model, nsp_loss, mlm_loss

ctx = [mx.gpu(i) for i in range(args.num_gpus)]
model, nsp_loss, mlm_loss = get_model(args, ctx)
metric = mx.metric.Accuracy()

def get_dataset(data, data_eval):

    # stream of stream?
    train_data = SimpleDatasetStream(cls, file_pattern, 'random', kwargs)

    class BERTDataset(ArrayDataset):
        def __init__(self, path):
            self._path = path
            dataset = np.load(path)
            self._next_sentence_labels = dataset["next_sentence_labels"]
            self._input_ids = dataset["input_ids"]
            self._segment_ids = dataset["segment_ids"]
            self._masked_lm_positions = dataset["masked_lm_positions"]
            self._masked_lm_ids = dataset["masked_lm_ids"]
            self._masked_lm_weights = dataset["masked_lm_weights"]
            self._valid_lens = dataset["valid_lengths"]
            arr = [self._next_sentence_labels, self._input_ids, self._segment_ids, self._masked_lm_positions, self._masked_lm_ids, self._masked_lm_weights, self._valid_lens]
            super(BERTDataset, self).__init__(*arr)

    file_path = '/tmp/tf_examples.npz'
    data_train = BERTDataset(file_path)
    data_eval = BERTDataset(file_path)
    return data_train, data_eval

def get_dataloader(data, data_eval):
    #trans = ClassificationTransform(tokenizer, MRPCDataset.get_labels(), args.max_len)
    #data_train = MRPCDataset('train').transform(trans)
    #data_dev = MRPCDataset('dev').transform(trans)

    bert_dataloader = DataLoader(data_train, batch_size=batch_size,
                                   shuffle=True, last_batch='rollover')
    bert_dataloader_dev = DataLoader(data_dev, batch_size=test_batch_size,
                                                   shuffle=False)

    return dataloader_train, dataloader_eval

batch_size = args.batch_size * args.num_gpus
test_batch_size = args.test_batch_size * args.num_gpus

do_lower_case = 'uncased' in dataset
tokenizer = FullTokenizer(vocabulary, do_lower_case=do_lower_case)
dataset_train, dataset_eval = get_dataset(args.data, args.data_eval)
data_train, data_eval = get_dataloader(batch_size, test_batch_size)

def evaluate():
    """Evaluate the model on validation dataset.
    """
    step_loss = 0
    metric.reset()
    nsp_loss.hybridize(static_alloc=True)
    mlm_loss.hybridize(static_alloc=True)
    for _, seqs in enumerate(bert_dataloader_dev):
        Ls = []
        input_ids, valid_len, type_ids, label = seqs
        out = model(input_ids.as_in_context(ctx), type_ids.as_in_context(ctx),
                    valid_len.astype('float32').as_in_context(ctx))
        ls = nsp_loss(out, label.as_in_context(ctx)).mean()
        Ls.append(ls)
        step_loss += sum([L.asscalar() for L in Ls])
        metric.update([label], [out])
    logging.info('validation accuracy: %s', metric.get()[1])

def train():
    """Training function."""
    trainer = gluon.Trainer(model.collect_params(), 'bertadam',
                            {'learning_rate': lr, 'epsilon': 1e-6, 'wd': 0.01},
                            kvstore=args.kv)

    num_train_examples = len(data_train)
    print('num_train_examples=', num_train_examples)
    num_train_steps = int(num_train_examples / batch_size * args.epochs)
    warmup_ratio = args.warmup_ratio
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    step_num = 0
    params = []

    # Do not apply weight decay on LayerNorm and bias terms
    for _, v in model.collect_params('.*beta|.*gamma|.*bias').items():
        v.wd_mult = 0.0

    for p in model.collect_params().values():
        if p.grad_req != 'null':
            params.append(p)
    mx.nd.waitall()

    for epoch_id in range(args.epochs):
        metric.reset()
        step_loss = 0
        t0 = time.time()
        for batch_id, seqs in enumerate(bert_dataloader):
            step_num += 1
            if step_num < num_warmup_steps:
                new_lr = lr * step_num / num_warmup_steps
            else:
                offset = (step_num - num_warmup_steps) * lr / (num_train_steps - num_warmup_steps)
                new_lr = lr - offset
            trainer.set_learning_rate(new_lr)

            def _split_load(arr, dtype, ctx):
                return mx.gluon.utils.split_and_load(arr.astype(args.dtype, copy=False), ctx)

            next_sentence_labels, input_ids, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, valid_lens = seqs

            next_sentence_labels = _split_load(next_sentence_labels, args.dtype, ctx)
            input_ids = _split_load(input_ids, args.dtype, ctx)
            segment_ids = _split_load(segment_ids, args.dtype, ctx)
            masked_lm_positions = _split_load(masked_lm_positions, args.dtype, ctx)
            masked_lm_ids = _split_load(masked_lm_ids, args.dtype, ctx)
            masked_lm_weights = _split_load(masked_lm_weights, args.dtype, ctx)
            valid_lens = _split_load(valid_lens, args.dtype, ctx)
            Ls = []
            classified_list = []

            for x in zip(next_sentence_labels, input_ids, segment_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, valid_lens):
                with mx.autograd.record():
                    next_sentence_label, input_id, segment_id, masked_lm_position, masked_lm_id, masked_lm_weight, valid_len = x

                    classified, decode_out = model(input_id, segment_id, valid_length=valid_len, positions=masked_lm_position)
                    decode_out = decode_out.reshape((-1, len(vocabulary)))
                    #position_id = position_id.reshape((-1,))
                    ls = nsp_loss(classified, next_sentence_label)

                    ls2 = mlm_loss(decode_out, masked_lm_id.reshape((-1))) * masked_lm_weight.reshape((-1))
                    ls = ls.mean() + ls2.mean()
                    Ls.append(ls)
                    classified_list.append(classified)
                mx.autograd.backward(ls)

            grads = [p.grad(c) for p in params for c in ctx]
            gluon.utils.clip_global_norm(grads, 1)
            trainer.step(1)

            for ls in Ls:
                step_loss += ls.asscalar()
            metric.update(next_sentence_labels, classified_list)
            mx.nd.waitall()
            if (batch_id + 1) % (args.log_interval) == 0:
                if args.profile:
                    if batch_id + 1 == args.log_interval:
                        mx.nd.waitall()
                        mx.profiler.set_config(profile_memory=False,profile_symbolic=True, aggregate_stats=True, profile_all=True, filename='profile_output.json')
                        mx.profiler.set_state('run')
                    elif batch_id + 1 == args.log_interval * 2:
                        mx.nd.waitall()
                        mx.profiler.set_state('stop')
                        print(mx.profiler.dumps())
                        exit()

                t1 = time.time()
                logging.info('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f}'
                             .format(epoch_id, batch_id + 1, len(bert_dataloader),
                                     step_loss / args.log_interval,
                                     trainer.learning_rate, metric.get()[1]))
                logging.info('Throughput={:.2f} K tokens/sec'
                             .format(args.log_interval * batch_size * args.max_len * 1.0 / (t1 - t0) / 1000))
                t0 = t1
                step_loss = 0
        mx.nd.waitall()
        evaluate()

if __name__ == '__main__':
    train()
