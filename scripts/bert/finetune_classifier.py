"""
Sentence Pair Classification with Bidirectional Encoder Representations from Transformers

=========================================================================================

This example shows how to implement finetune a model with pre-trained BERT parameters for
sentence pair classification, with Gluon NLP Toolkit.

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

import argparse
import time
import random
import os
import io
import logging
import math
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data import ArrayDataset, SimpleDataset
from mxnet.gluon.data import DataLoader
from gluonnlp.model import bert_12_768_12
from bert import BERTClassifier
from utils import logging_config
from tokenization import FullTokenizer
from dataset import MRPCDataset, SentenceClassificationTrans

np.random.seed(0)
random.seed(0)
mx.random.seed(0)

parser = argparse.ArgumentParser(description='Neural Machine Translation Example.'
                                             'We train the Transformer Model')
parser.add_argument('--vocab_file', type=str, required=True,
                    help='Path to the vocabulary file. e.g. $BERT_BASE_DIR/vocab.txt')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size. Number of examples per gpu in a minibatch')
parser.add_argument('--test_batch_size', type=int, default=8, help='Test batch size')
parser.add_argument('--optimizer', type=str, default='adam', help='optimization algorithm')
parser.add_argument('--lr', type=float, default=5e-5, help='Initial learning rate')
parser.add_argument('--warmup_ratio', type=float, default=0.1,
                    help='ratio of warmup steps used in NOAM\'s stepsize schedule')
parser.add_argument('--log_interval', type=int, default=10, help='report interval')
parser.add_argument('--max_len', type=int, default=128, help='Maximum length of the sentence pairs')
parser.add_argument('--output_dir', type=str, default='classifier_out',
                    help='directory path to save the final model and training log')
parser.add_argument('--gpu', action='store_true', help='whether to use gpu for finetuning')
args = parser.parse_args()
logging_config(args.output_dir)
logging.info(args)
batch_size = args.batch_size
test_batch_size = args.test_batch_size


ctx = [mx.cpu()] if args.gpu is None or args.gpu == '' else [mx.gpu()]

do_lower_case=True
tokenizer = FullTokenizer(vocab_file=args.vocab_file, do_lower_case=do_lower_case)

bert, vocabulary = bert_12_768_12(dataset_name='book_corpus_wiki_en_uncased',
                                  pretrained=True, ctx=mx.cpu(), use_pooler=True,
                                  use_decoder=False, use_classifier=False,
                                  root='/home/ubuntu/gluon-nlp/tests/data/model/')
bert.collect_params().reset_ctx(ctx)

model = BERTClassifier(bert, dropout=0.1)
model.initialize(init=mx.init.Normal(0.02), ctx=ctx)
static_alloc = True
model.hybridize(static_alloc=static_alloc)
logging.info(model)

loss_function = gluon.loss.SoftmaxCELoss()
loss_function.hybridize(static_alloc=static_alloc)

metric = mx.metric.Accuracy()

trans = SentenceClassificationTrans(tokenizer, MRPCDataset.get_labels(), args.max_len)
data_train = MRPCDataset('train').transform(trans)
data_dev = MRPCDataset('dev').transform(trans)

def evaluate():
    """Evaluate given the data loader

    Parameters
    ----------
    data_loader : DataLoader

    Returns
    -------
    avg_loss : float
        Average loss
    real_translation_out : list of list of str
        The translation output
    """
    step_loss = 0
    bert_dataloader_dev = mx.gluon.data.DataLoader(data_dev, batch_size=test_batch_size, shuffle=False)
    metric.reset()
    log_start_time = time.time()

    for batch_id, seqs in enumerate(bert_dataloader_dev):
        Ls = []
        input_ids_nd0, input_len0, segment_ids_nd0, label_ids_nd0 = seqs
        out  = model(input_ids_nd0.reshape((test_batch_size, -1)).as_in_context(mx.gpu()),
                     segment_ids_nd0.reshape((test_batch_size, -1)).as_in_context(mx.gpu()),
                     input_len0.squeeze().astype('float32').as_in_context(mx.gpu()))
        ls = loss_function(out, label_ids_nd0.as_in_context(mx.gpu())).mean()
        Ls.append(ls)
        step_loss += sum([L.asscalar() for L in Ls])
        metric.update([label_ids_nd0], [out])
    print('validation', metric.get())


def train():
    """Training function."""
    trainer = gluon.Trainer(model.collect_params(), args.optimizer,
                            {'learning_rate': args.lr, 'epsilon': 1e-9})

    bert_dataloader = mx.gluon.data.DataLoader(data_train, batch_size=batch_size,
                                               shuffle=True, last_batch='discard')
                                               #shuffle=False, last_batch='discard')
    num_train_examples = len(data_train)
    print('num samples = ', num_train_examples)
    num_train_steps = int(
        num_train_examples / batch_size * args.epochs)
    warmup_ratio = args.warmup_ratio
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    step_num = 0

    for epoch_id in range(args.epochs):
        metric.reset()
        log_avg_loss = 0
        step_loss = 0
        log_start_time = time.time()

        for batch_id, seqs in enumerate(bert_dataloader):
            step_num += 1
            if step_num < num_warmup_steps:
                new_lr = args.lr * step_num / num_warmup_steps
            else:
                new_lr = args.lr - (step_num - num_warmup_steps) * args.lr / (num_train_steps - num_warmup_steps)
            trainer.set_learning_rate(new_lr)
            Ls = []
            #with mx.autograd.pause():
            with mx.autograd.record():
                input_ids_nd0, input_mask_nd0, segment_ids_nd0, label_ids_nd0 = seqs
                #print(input_ids_nd0.sum().asscalar(),input_mask_nd0.sum().asscalar(),
                #      segment_ids_nd0.sum().asscalar(), label_ids_nd0.sum().asscalar())
                out  = model(input_ids_nd0.reshape((batch_size, -1)).as_in_context(mx.gpu()),
                             segment_ids_nd0.reshape((batch_size, -1)).as_in_context(mx.gpu()),
                             input_mask_nd0.squeeze().astype('float32').as_in_context(mx.gpu()))
                ls = loss_function(out, label_ids_nd0.as_in_context(mx.gpu())).mean()
                Ls.append(ls)
            #if batch_id == 0:
            #    mx.nd.waitall()
            #    exit()
            for L in Ls:
                L.backward()
            parameters = model.collect_params()
            differentiable_params = []
            for p in parameters.values():
                if p.grad_req != 'null':
                    differentiable_params.append(p)

            grads = [p.grad(c) for p in differentiable_params for c in [mx.gpu()]]
            gluon.utils.clip_global_norm(grads, 1)
            trainer.step(1)
            step_loss += sum([L.asscalar() for L in Ls])
            metric.update([label_ids_nd0], [out])
            if (batch_id + 1) % (args.log_interval) == 0:
                logging.info('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f}'
                             .format(epoch_id, batch_id + 1, len(bert_dataloader),
                                     step_loss / args.log_interval,
                                     trainer.learning_rate, metric.get()[1]))
                log_start_time = time.time()
                log_avg_loss = 0
                step_loss = 0
        mx.nd.waitall()
        evaluate()

if __name__ == '__main__':
    train()
