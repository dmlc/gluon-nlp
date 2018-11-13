"""
Bidirectional Encoder Representations from Transformers.

=================================

This example shows how to implement the Transformer model with Gluon NLP Toolkit.

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones,
          Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6000--6010},
  year={2017}
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
from bert import BERTModel, get_transformer_encoder, BERTClassifier
from utils import logging_config
import tokenization
from dataset import MRPCDataset, SentenceClassificationTrans

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)

parser = argparse.ArgumentParser(description='Neural Machine Translation Example.'
                                             'We train the Transformer Model')
parser.add_argument('--vocab_file', type=str, required=True,
                    help='Path to the vocabulary file. e.g. $BERT_BASE_DIR/vocab.txt')
parser.add_argument('--epochs', type=int, default=3, help='number of epochs')
parser.add_argument('--num_units', type=int, default=768, help='Dimension of the embedding '
                                                               'vectors and states.')
parser.add_argument('--hidden_size', type=int, default=3072,
                    help='Dimension of the hidden state in position-wise feed-forward networks.')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--num_layers', type=int, default=12,
                    help='number of layers in the encoder and decoder')
parser.add_argument('--num_heads', type=int, default=12,
                    help='number of heads in multi-head attention')
parser.add_argument('--scaled', action='store_true', help='Turn on to use scale in attention')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size. Number of examples per gpu in a minibatch')
parser.add_argument('--test_batch_size', type=int, default=8, help='Test batch size')
parser.add_argument('--max_len', type=int, default=128, help='Maximum length of the sentence pairs')
parser.add_argument('--optimizer', type=str, default='adam', help='optimization algorithm')
parser.add_argument('--lr', type=float, default=5e-5, help='Initial learning rate')
parser.add_argument('--warmup_ratio', type=float, default=0.1,
                    help='ratio of warmup steps used in NOAM\'s stepsize schedule')
parser.add_argument('--log_interval', type=int, default=10, help='report interval')
parser.add_argument('--output_dir', type=str, default='classifier_out',
                    help='directory path to save the final model and training log')
parser.add_argument('--gpu', action='store_true', help='whether to use gpu for finetuning')
args = parser.parse_args()
logging_config(args.output_dir)
logging.info(args)
batch_size = args.batch_size
test_batch_size = args.test_batch_size


ctx = [mx.cpu()] if args.gpu is None or args.gpu == '' else [mx.gpu()]

encoder = get_transformer_encoder(units=args.num_units,
                                  hidden_size=args.hidden_size,
                                  dropout=args.dropout,
                                  num_layers=args.num_layers,
                                  num_heads=args.num_heads,
                                  max_src_length=512,
                                  max_tgt_length=512,
                                  scaled=args.scaled)

import tokenization
do_lower_case=True
tokenizer = tokenization.FullTokenizer(
    vocab_file=args.vocab_file, do_lower_case=do_lower_case)

print(tokenizer)
MAX_SEQ_LENGTH = 128
vocab = tokenizer.vocab
model = BERTModel(encoder=encoder, vocab_size=len(vocab), token_type_vocab_size=2,
                  units=args.num_units, embed_size=args.num_units,
                  embed_initializer=None, prefix='transformer_')

# TODO dropout for classifier
model = BERTClassifier(model)

model.initialize(init=mx.init.Normal(0.02), ctx=ctx)
static_alloc = True
#model.hybridize(static_alloc=static_alloc)
logging.info(model)

ones = mx.nd.ones((1, 128), ctx=mx.gpu())
out = model(ones, ones, mx.nd.ones((1,), ctx=mx.gpu()))
params = model.bert._collect_params_with_prefix()
import pickle
#import pdb; pdb.set_trace()
#print(sorted(params.keys()))
with open('/home/ubuntu/bert/bert.pickle.mx', 'rb') as f:
    tf_params = pickle.load(f)
loaded = {}
for name in params:
    try:
        arr = mx.nd.array(tf_params[name])
        params[name].set_data(arr)
        loaded[name] = 0
    except:
        if name not in tf_params:
            print("cannot initialize %s from bert checkpoint"%(name))
        else:
            print("cannot initialize ", name, params[name].shape, tf_params[name].shape)
print('num_loaded = ', len(loaded), ' total = ', len(tf_params))
for name in tf_params:
    if name not in loaded:
        print('not loading', name)

loss_function = gluon.loss.SoftmaxCELoss()
loss_function.hybridize(static_alloc=static_alloc)

metric = mx.metric.Accuracy()
metric_dev = mx.metric.Accuracy()

trans = SentenceClassificationTrans(tokenizer, MRPCDataset.get_labels(), args.max_len)
train_examples2 = MRPCDataset('train')
train_examples2 = train_examples2.transform(trans)
data_mx = train_examples2

dev_examples2 = MRPCDataset('dev')
dev_examples2 = dev_examples2.transform(trans)
data_mx_dev = dev_examples2

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
    bert_dataloader_dev = mx.gluon.data.DataLoader(data_mx_dev, batch_size=test_batch_size, shuffle=False)
    metric_dev.reset()
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
        metric_dev.update([label_ids_nd0], [out])
    print('validation', metric_dev.get())


def train():
    """Training function."""
    trainer = gluon.Trainer(model.collect_params(), args.optimizer,
                            {'learning_rate': args.lr, 'epsilon': 1e-9})

    bert_dataloader = mx.gluon.data.DataLoader(data_mx, batch_size=batch_size, shuffle=False, last_batch='discard')
    num_train_examples = len(data_mx)
    print('num samples = ', num_train_examples)
    num_train_steps = int(
        num_train_examples / batch_size * args.epochs)
    warmup_ratio = args.warmup_ratio
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    step_num = 0
    metric.reset()

    for epoch_id in range(args.epochs):
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
