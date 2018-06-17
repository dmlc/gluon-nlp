"""
Word Language Model
===================

This example shows how to build a word-level language model on WikiText-2 with Gluon NLP Toolkit.
By using the existing data pipeline tools and building blocks, the process is greatly simplified.

We implement the AWD LSTM language model proposed in the following work.

@article{merityRegOpt,
  title={{Regularizing and Optimizing LSTM Language Models}},
  author={Merity, Stephen and Keskar, Nitish Shirish and Socher, Richard},
  journal={ICLR},
  year={2018}
}

Note that we are using standard SGD as the optimizer for code simpilification.
Once NT-ASGD in the work is implemented and used as the optimizer.
Our implementation should yield identical results.
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


import time
import math
import os
import sys
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
import gluonnlp as nlp

import argparse
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '..', '..'))

#parser.add_argument('--model', type=str, default='lstm',
#                    help='type of recurrent net (rnn_tanh, rnn_relu, lstm, gru)')
parser = argparse.ArgumentParser(description=
                                 'MXNet Big RNN/LSTM Language Model on Google Billion Words')
parser.add_argument('--save', type=str, default='model.params',
                    help='path to save the final model')
parser.add_argument('--load', type=str, default=None,
                    help='path to load the final model')
parser.add_argument('--vocab', type=str, default='./data/1b_word_vocab.txt',
                    help='location of the corpus vocabulary file')
parser.add_argument('--emsize', type=int, default=512,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=2048,
                    help='number of hidden units per layer')
parser.add_argument('--nproj', type=int, default=512,
                    help='number of projection units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--epochs', type=int, default=8,
                    help='number of epoch for training')
parser.add_argument('--batch-size', type=int, default=256,
                    help='batch size per gpu??')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--eps', type=float, default=1,
                    help='epsilon for adagrad')
parser.add_argument('--bptt', type=int, default=20,
                    help='sequence length')
parser.add_argument('--k', type=int, default=8192,
                    help='number of noise samples for estimation')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--checkpoint-interval', type=int, default=None,
                    help='report interval')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--lr', type=float, default=0.2,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=10.0,
                    help='gradient clipping by global norm.')
parser.add_argument('--unique', action='store_true',
                    help='Whether to use unique samples')
parser.add_argument('--test_mode', action='store_true',
                    help='Whether to run through the script with few examples')
parser.add_argument('--eval', action='store_true',
                    help='Whether to only evaluate the trained model')
args = parser.parse_args()

###############################################################################
# Load data
###############################################################################

mx.random.seed(args.seed)
np.random.seed(args.seed)

context = [mx.cpu()] if args.gpus is None or args.gpus == '' else \
          [mx.gpu(int(x)) for x in args.gpus.split(',')]

assert args.batch_size % len(context) == 0, \
    'Total batch size must be multiple of the number of devices'

train_data_stream, test_data_stream = \
    [nlp.data.GBWStream(segment=segment, skip_empty=False, bos=None, eos='<eos>')
     for segment in ['train', 'test']]

vocab = nlp.data.GBWVocab()

train_batch_size = args.batch_size
train_data = nlp.data.PrefetchingStream(train_data_stream.bptt_batchify(vocab, args.bptt, train_batch_size))
test_batch_size = 1
test_data = nlp.data.PrefetchingStream(test_data_stream.bptt_batchify(vocab, args.bptt, test_batch_size))
checkpoint_interval = args.checkpoint_interval

max_nbatch = None
max_nbatch_eval = None if args.eval else 64

if args.test_mode:
    #args.emsize = 200
    #args.nhid = 200
    #args.nlayers = 1
    #args.epochs = 1
    max_nbatch = 10
    max_nbatch_eval = 10
    checkpoint_interval = 5
    #train_data = test_data_stream.bptt_batchify(vocab, args.bptt, args.batch_size)
print(args)

###############################################################################
# Build the model
###############################################################################


ntokens = len(vocab)
sampler = mx.nd.contrib.rand_zipfian_unique if args.unique else mx.nd.contrib.rand_zipfian
eval_model = nlp.model.language_model.BigRNN(ntokens, args.emsize, args.nhid,
                                             args.nlayers, args.nproj,
                                             dropout=args.dropout)
model = nlp.model.language_model.train.BigRNN(ntokens, args.emsize, args.nhid,
                                              args.nlayers, args.nproj, args.k,
                                              dropout=args.dropout, sampler=sampler)
print(model)
model.initialize(mx.init.Xavier(), ctx=context)
model.hybridize()

trainer_params = {'learning_rate': args.lr, 'wd': 0, 'eps': args.eps}

trainer = gluon.Trainer(model.collect_params(), 'adagrad', trainer_params)
loss = gluon.loss.SoftmaxCrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def train():
    """Training loop for language model.

    """
    best_val = float('Inf')
    start_train_time = time.time()
    parameters = model.collect_params().values()
    encoder_params = model.encoder.collect_params().values()
    embedding_params = model.embedding.collect_params().values()
    for epoch in range(args.epochs):
        total_L = 0.0
        start_epoch_time = time.time()
        start_log_interval_time = time.time()
        hiddens = [model.begin_state(batch_size=args.batch_size//len(context),
                                     func=mx.nd.zeros, ctx=ctx) for ctx in context]
        nbatch = 0
        for data, target, mask in train_data:
            nbatch += 1
            data_list = gluon.utils.split_and_load(data, context, batch_axis=1, even_split=True)
            target_list = gluon.utils.split_and_load(target, context, batch_axis=1, even_split=True)
            mask_list = gluon.utils.split_and_load(mask, context, batch_axis=1, even_split=True)
            hiddens = detach(hiddens)
            Ls = []
            L = 0
            with autograd.record():
                for j, (X, y, m, h) in enumerate(zip(data_list, target_list, mask_list, hiddens)):
                    output, new_target, h = model(X, y, h)
                    l = loss(output, new_target) * m.reshape((-1,))
                    L = L + l.as_in_context(context[0]) / X.size
                    Ls.append(l/X.size)
                    hiddens[j] = h
            L.backward()
            # TODO rescale embedding grad
            for d in data_list:
                x = embedding_params[0].grad(d.context)
                x *= args.batch_size
            encoder_grads = [p.grad(d.context) for p in encoder_params for d in data_list]
            gluon.utils.clip_global_norm(encoder_grads, args.clip)

            trainer.step(len(context))

            total_L += sum([mx.nd.sum(L).asscalar() for L in Ls])
            if nbatch % args.log_interval == 0:
                cur_L = total_L / args.log_interval / len(context)
                ppl = math.exp(cur_L) if cur_L < 100 else 99999999
                print('[Epoch %d Batch %d] loss %.2f, ppl %.2f, '
                      'throughput %.2f samples/s'
                      %(epoch, nbatch, cur_L, ppl,
                        args.batch_size*args.log_interval/(time.time()-start_log_interval_time)))
                total_L = 0.0
                start_log_interval_time = time.time()
            # early stop for testing
            if checkpoint_interval and nbatch % checkpoint_interval == 0:
                sys.stdout.flush()
                model.save_params('/dev/shm/epoch_%d_%s'%(epoch, args.save))
                eval_model.load_params('/dev/shm/epoch_%d_%s'%(epoch, args.save))
                final_test_L = evaluate(test_data, test_batch_size, ctx=mx.cpu())
                print('[Epoch %d Batch %d] test loss %.2f, test ppl %.2f'%(epoch, nbatch, final_test_L, math.exp(final_test_L)))
                sys.stdout.flush()

            if max_nbatch and nbatch > max_nbatch:
                break

        mx.nd.waitall()

        #print('[Epoch %d] throughput %.2f samples/s'%(
        #    epoch, (args.batch_size * len(train_data)) / (time.time() - start_epoch_time)))

        #val_L = evaluate(val_data, val_batch_size, context[0])
        #print('[Epoch %d] time cost %.2fs, valid loss %.2f, valid ppl %.2f'%(
        #    epoch, time.time()-start_epoch_time, val_L, math.exp(val_L)))
        model.save_params('epoch_%d_%s'%(epoch, args.save))
        eval_model.load_params('epoch_%d_%s'%(epoch, args.save))
        final_test_L = evaluate(test_data, test_batch_size, ctx=mx.cpu())
        print('Epoch %d: test loss %.2f, test ppl %.2f'%(epoch, final_test_L, math.exp(final_test_L)))

    #print('Total training throughput %.2f samples/s'
    #      %((args.batch_size * len(train_data) * args.epochs) / (time.time() - start_train_time)))

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [detach(h) for h in hidden]
    else:
        hidden = hidden.detach()
    return hidden

def evaluate(data_stream, batch_size, ctx=None):
    """Evaluate the model on the dataset.

    Parameters
    ----------
    data_stream : DataStream
        The dataset is evaluated on.
    batch_size : int
        The size of the mini-batch.
    ctx : mx.cpu() or mx.gpu()
        The context of the computation.

    Returns
    -------
    loss: float
        The loss on the dataset
    """
    total_L = 0.0
    ntotal = 0
    nbatch = 0
    hidden = eval_model.begin_state(batch_size=batch_size, func=mx.nd.zeros, ctx=ctx)
    for data, target, mask in data_stream:
        data = data.as_in_context(ctx)
        target = target.as_in_context(ctx)
        mask = mask.as_in_context(ctx)
        output, hidden = eval_model(data, hidden)
        hidden = detach(hidden)
        L = loss(output, target.reshape(-1,)) * mask.reshape((-1,))
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
        nbatch += 1
        avg = total_L / ntotal
        if nbatch % args.log_interval == 0:
            print('Eval till batch %d: test loss %.2f, test ppl %.2f'%(nbatch, avg, math.exp(avg)))
        if max_nbatch_eval and nbatch > max_nbatch_eval:
            break
    return total_L / ntotal

def load():
    model.load_params(args.load)

if __name__ == '__main__':
    start_pipeline_time = time.time()
    if args.load:
        load()
    if not args.eval:
        train()
    epoch = args.epochs - 1
    nbatch = -1
    eval_model.load_params('/dev/shm/epoch_%d_%s'%(epoch, args.save))
    final_test_L = evaluate(test_data, test_batch_size, ctx=mx.cpu())
    print('[Epoch %d Batch %d] test loss %.2f, test ppl %.2f'%(epoch, nbatch, final_test_L, math.exp(final_test_L)))
    print('Total time cost %.2fs'%(time.time()-start_pipeline_time))
