"""
Word Language Model
===================

This example shows how to build a word-level language model on WikiText-2 with Gluon NLP Toolkit.
By using the existing data pipeline tools and building blocks, the process is greatly simplified.
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

import argparse
import time
import math
import mxnet as mx
from mxnet import gluon, autograd
import os
import sys
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '..', '..'))
from gluonnlp import datasets, Vocab
from gluonnlp.models.language_model import StandardRNN, AWDRNN
from gluonnlp.data import Counter
from mxnet.gluon.data import SimpleDataset

parser = argparse.ArgumentParser(description='MXNet Autograd RNN/LSTM Language Model on Wikitext-2.')
parser.add_argument('--model', type=str, default='lstm',
                    help='type of recurrent net (rnn_tanh, rnn_relu, lstm, gru)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropout_h', type=float, default=0.3,
                    help='dropout applied to hidden layer (0 = no dropout)')
parser.add_argument('--dropout_i', type=float, default=0.65,
                    help='dropout applied to input layer (0 = no dropout)')
parser.add_argument('--weight_dropout', type=float, default=0.0,
                    help='weight dropout applied to h2h weight matrix (0 = no weight dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.params',
                    help='path to save the final model')
parser.add_argument('--gctype', type=str, default='none',
                    help='type of gradient compression to use, \
                          takes `2bit` or `none` for now.')
parser.add_argument('--gcthreshold', type=float, default=0.5,
                    help='threshold for 2bit gradient compression')
parser.add_argument('--eval_only', action='store_true',
                    help='Whether to only evaluate the trained model')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu. (the result of multi-gpu training might be slightly different compared to single-gpu training, still need to be finalized)')
args = parser.parse_args()

print(args)


###############################################################################
# Load data
###############################################################################

context = [mx.cpu()] if args.gpus is None or args.gpus == "" else \
          [mx.gpu(int(i)) for i in args.gpus.split(',')]

assert args.batch_size % len(context) == 0, "Total batch size must be multiple of the number of devices"

train_dataset, val_dataset, test_dataset = [datasets.WikiText2(segment=segment,
                                                               bos=None, eos='<eos>')
                                            for segment in ['train', 'val', 'test']]

vocab = Vocab(Counter(train_dataset[0]), padding_token=None, bos_token=None)

train_data, val_data, test_data = [x.bptt_batchify(vocab, args.bptt, args.batch_size,
                                                   last_batch='keep')
                                   for x in [train_dataset, val_dataset, test_dataset]]


###############################################################################
# Build the model
###############################################################################


ntokens = len(vocab)

if args.weight_dropout > 0:
    print("Use weight_drop!")
    model = AWDRNN(args.model, len(vocab), args.emsize, args.nhid, args.nlayers,
                   args.tied, args.dropout, args.weight_dropout, args.dropout_h, args.dropout_i)
else:
    model = StandardRNN(args.model, len(vocab), args.emsize, args.nhid, args.nlayers, args.dropout,
                        args.tied)

model.initialize(mx.init.Xavier(), ctx=context)


compression_params = None if args.gctype == 'none' else {'type': args.gctype, 'threshold': args.gcthreshold}
trainer = gluon.Trainer(model.collect_params(), 'sgd',
                        {'learning_rate': args.lr,
                         'momentum': 0,
                         'wd': 0},
                        compression_params=compression_params)
loss = gluon.loss.SoftmaxCrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [detach(i) for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden

def evaluate(data_source, ctx):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(args.batch_size, func=mx.nd.zeros, ctx=ctx)
    for i, (data, target) in enumerate(data_source):
        data = data.as_in_context(ctx)
        target = target.as_in_context(ctx)
        output, hidden = model(data, hidden)
        L = loss(output.reshape(-3, -1),
                 target.reshape(-1))
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal

def train():
    best_val = float("Inf")
    start_train_time = time.time()
    parameters = model.collect_params().values()
    for epoch in range(args.epochs):
        total_L, n_total = 0.0, 0
        start_epoch_time = time.time()
        start_log_interval_time = time.time()
        hiddens = [model.begin_state(args.batch_size//len(context), func=mx.nd.zeros, ctx=ctx) for ctx in context]
        for i, (data, target) in enumerate(train_data):
            data_list = gluon.utils.split_and_load(data, context, batch_axis=1, even_split=True)
            target_list = gluon.utils.split_and_load(target, context, batch_axis=1, even_split=True)
            hiddens = detach(hiddens)
            L = 0
            Ls = []
            with autograd.record():
                for j, (X, y, h) in enumerate(zip(data_list, target_list, hiddens)):
                    output, h = model(X, h)
                    batch_L = loss(output.reshape(-3, -1), y.reshape(-1))
                    L = L + batch_L.as_in_context(context[0]) / X.size
                    Ls.append(batch_L)
                    hiddens[j] = h
            L.backward()
            grads = [p.grad(x.context) for p in parameters for x in data_list]
            gluon.utils.clip_global_norm(grads, args.clip)

            trainer.step(1)

            total_L += sum([mx.nd.sum(l).asscalar() for l in Ls])
            n_total += data.size

            if i % args.log_interval == 0 and i > 0:
                cur_L = total_L / n_total
                print('[Epoch %d Batch %d/%d] loss %.2f, ppl %.2f, throughput %.2f samples/s'%(
                    epoch, i, len(train_data), cur_L, math.exp(cur_L),
                    args.batch_size * args.log_interval / (time.time() - start_log_interval_time)))
                total_L, n_total = 0.0, 0
                start_log_interval_time = time.time()

        mx.nd.waitall()

        print('[Epoch %d] throughput %.2f samples/s'%(
                    epoch, (args.batch_size * len(train_data)) / (time.time() - start_epoch_time)))
        val_L = evaluate(val_data, context[0])
        print('[Epoch %d] time cost %.2fs, valid loss %.2f, valid ppl %.2f'%(
            epoch, time.time()-start_epoch_time, val_L, math.exp(val_L)))

        if val_L < best_val:
            best_val = val_L
            test_L = evaluate(test_data, context[0])
            model.save_params(args.save)
            print('test loss %.2f, test ppl %.2f'%(test_L, math.exp(test_L)))
        else:
            args.lr = args.lr*0.25
            print('Learning rate now %f'%(args.lr))
            trainer.set_learning_rate(args.lr)

    print('Total training throughput %.2f samples/s'%(
                            (args.batch_size * len(train_data) * args.epochs) / (time.time() - start_train_time)))

if __name__ == '__main__':
    start_pipeline_time = time.time()
    if not args.eval_only:
        train()
    model.load_params(args.save, context)
    val_L = evaluate(val_data)
    test_L = evaluate(test_data)
    print('Best validation loss %.2f, test ppl %.2f'%(val_L, math.exp(val_L)))
    print('Best test loss %.2f, test ppl %.2f'%(test_L, math.exp(test_L)))
    print('Total time cost %.2fs'%(time.time()-start_pipeline_time))
