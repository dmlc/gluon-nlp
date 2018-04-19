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
import os
import sys
import mxnet as mx
from mxnet import gluon, autograd
from gluonnlp import Vocab
from gluonnlp.model.language_model import StandardRNN, AWDRNN
from gluonnlp.data import language_model, Counter
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '..', '..'))


parser = argparse.ArgumentParser(description=
                                 'MXNet Autograd RNN/LSTM Language Model on Wikitext-2.')
parser.add_argument('--model', type=str, default='lstm',
                    help='type of recurrent net (rnn_tanh, rnn_relu, lstm, gru)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=180,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropout_h', type=float, default=0.2,
                    help='dropout applied to hidden layer (0 = no dropout)')
parser.add_argument('--dropout_i', type=float, default=0.65,
                    help='dropout applied to input layer (0 = no dropout)')
parser.add_argument('--dropout_e', type=float, default=0.1,
                    help='dropout applied to embedding layer (0 = no dropout)')
parser.add_argument('--weight_dropout', type=float, default=0.5,
                    help='weight dropout applied to h2h weight matrix (0 = no weight dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.params',
                    help='path to save the final model')
parser.add_argument('--eval_only', action='store_true',
                    help='Whether to only evaluate the trained model')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.'
                         '(using single gpu is suggested)')
parser.add_argument('--lr_update_interval', type=int, default=30,
                    help='lr udpate interval')
parser.add_argument('--lr_update_factor', type=float, default=0.1,
                    help='lr udpate factor')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--wd', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation '
                         '(alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation '
                         '(beta = 0 means no regularization)')
parser.add_argument('--test_mode', action='store_true',
                    help='Whether to run through the script with few examples')
args = parser.parse_args()

###############################################################################
# Load data
###############################################################################

context = [mx.cpu()] if args.gpus is None or args.gpus == '' else \
          [mx.gpu(int(i)) for i in args.gpus.split(',')]

assert args.batch_size % len(context) == 0, \
    'Total batch size must be multiple of the number of devices'

train_dataset, val_dataset, test_dataset = \
    [language_model.WikiText2(segment=segment,
                              skip_empty=False, bos=None, eos='<eos>')
     for segment in ['train', 'val', 'test']]

vocab = Vocab(counter=Counter(train_dataset[0]), padding_token=None, bos_token=None)

train_data = train_dataset.batchify(vocab, args.batch_size)
val_batch_size = 10
val_data = val_dataset.batchify(vocab, val_batch_size)
test_batch_size = 1
test_data = test_dataset.batchify(vocab, test_batch_size)

if args.test_mode:
    args.emsize = 200
    args.nhid = 200
    args.nlayers = 1
    args.epochs = 3
    train_data = train_data[0:100]
    val_data = val_data[0:100]
    test_data = test_data[0:100]

print(args)

###############################################################################
# Build the model
###############################################################################


ntokens = len(vocab)

if args.weight_dropout > 0:
    print('Use AWDRNN')
    model = AWDRNN(args.model, len(vocab), args.emsize, args.nhid, args.nlayers,
                   args.tied, args.dropout, args.weight_dropout, args.dropout_h,
                   args.dropout_i, args.dropout_e)
else:
    model = StandardRNN(args.model, len(vocab), args.emsize, args.nhid, args.nlayers, args.dropout,
                        args.tied)

model.initialize(mx.init.Xavier(), ctx=context)

if args.optimizer == 'sgd':
    trainer_params = {'learning_rate': args.lr,
                      'momentum': 0,
                      'wd': args.wd}
elif args.optimizer == 'adam':
    trainer_params = {'learning_rate': args.lr,
                      'wd': args.wd,
                      'beta1': 0,
                      'beta2': 0.999,
                      'epsilon': 1e-9}

trainer = gluon.Trainer(model.collect_params(), args.optimizer, trainer_params)
loss = gluon.loss.SoftmaxCrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def get_batch(data_source, i, seq_len=None):
    seq_len = min(seq_len if seq_len else args.bptt, len(data_source) - 1 - i)
    data = data_source[i:i+seq_len]
    target = data_source[i+1:i+1+seq_len]
    return data, target

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [detach(i) for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden

def evaluate(data_source, batch_size, ctx=None):
    """Evaluate the model on the dataset.

    Parameters
    ----------
    data_source : NDArray
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
    hidden = model.begin_state(batch_size, func=mx.nd.zeros, ctx=context[0])
    for i in range(0, len(data_source) - 1, args.bptt):
        data, target = get_batch(data_source, i)
        data = data.as_in_context(ctx)
        target = target.as_in_context(ctx)
        output, hidden = model(data, hidden)
        hidden = detach(hidden)
        L = loss(output.reshape(-3, -1),
                 target.reshape(-1,))
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal


def awd_forward(inputs, begin_state=None):
    """Implement forward computation using awd language model.

    Parameters
    ----------
    inputs : NDArray
        The training dataset.
    begin_state : list
        The initial hidden states.

    Returns
    -------
    out: NDArray
        The output of the model.
    out_states: list
        The list of output states of the model's encoder.
    encoded_raw: list
        The list of outputs of the model's encoder.
    encoded_dropped: list
        The list of outputs with dropout of the model's encoder.
    """
    encoded = model.embedding(inputs)
    if not begin_state:
        begin_state = model.begin_state(batch_size=inputs.shape[1])
    out_states = []
    encoded_raw = []
    encoded_dropped = []
    for i, (e, s) in enumerate(zip(model.encoder, begin_state)):
        encoded, state = e(encoded, s)
        encoded_raw.append(encoded)
        out_states.append(state)
        if model._drop_h and i != len(model.encoder)-1:
            encoded = mx.nd.Dropout(encoded, p=model._drop_h, axes=(0,))
            encoded_dropped.append(encoded)
    if model._dropout:
        encoded = mx.nd.Dropout(encoded, p=model._dropout, axes=(0,))
    encoded_dropped.append(encoded)
    with autograd.predict_mode():
        out = model.decoder(encoded)
    return out, out_states, encoded_raw, encoded_dropped

def criterion(output, target, encoder_hs, dropped_encoder_hs):
    """Compute regularized (optional) loss of the language model in training mode.

        Parameters
        ----------
        output: NDArray
            The output of the model.
        target: list
            The list of output states of the model's encoder.
        encoder_hs: list
            The list of outputs of the model's encoder.
        dropped_encoder_hs: list
            The list of outputs with dropout of the model's encoder.

        Returns
        -------
        l: NDArray
            The loss per word/token.
            If both args.alpha and args.beta are zeros, the loss is the standard cross entropy.
            If args.alpha is not zero, the standard loss is regularized with activation.
            If args.beta is not zero, the standard loss is regularized with temporal activation.
    """
    l = loss(output.reshape(-3, -1), target.reshape(-1,))
    if args.alpha:
        dropped_means = [args.alpha*dropped_encoder_h.__pow__(2).mean()
                         for dropped_encoder_h in dropped_encoder_hs[-1:]]
        l = l + mx.nd.add_n(*dropped_means)
    if args.beta:
        means = [args.beta*(encoder_h[1:] - encoder_h[:-1]).__pow__(2).mean()
                 for encoder_h in encoder_hs[-1:]]
        l = l + mx.nd.add_n(*means)
    return l

def train():
    """Training loop for awd language model.

    """
    best_val = float('Inf')
    start_train_time = time.time()
    parameters = model.collect_params().values()
    for epoch in range(args.epochs):
        total_L = 0.0
        start_epoch_time = time.time()
        start_log_interval_time = time.time()
        hiddens = [model.begin_state(args.batch_size//len(context),
                                     func=mx.nd.zeros, ctx=ctx) for ctx in context]
        batch_i, i = 0, 0
        while i < len(train_data) - 1 - 1:
            bptt = args.bptt if mx.nd.random.uniform().asscalar() < 0.95 else args.bptt / 2
            seq_len = max(5, int(mx.nd.random.normal(bptt, 5).asscalar()))
            lr_batch_start = trainer.learning_rate
            trainer.set_learning_rate(lr_batch_start*seq_len/args.bptt)

            data, target = get_batch(train_data, i, seq_len=seq_len)
            data_list = gluon.utils.split_and_load(data, context, batch_axis=1, even_split=True)
            target_list = gluon.utils.split_and_load(target, context, batch_axis=1, even_split=True)
            hiddens = detach(hiddens)
            Ls = []
            L = 0
            with autograd.record():
                for j, (X, y, h) in enumerate(zip(data_list, target_list, hiddens)):
                    if args.weight_dropout > 0:
                        output, h, encoder_hs, dropped_encoder_hs = awd_forward(X, h)
                        l = criterion(output, y, encoder_hs, dropped_encoder_hs)
                    else:
                        output, h = model(X, h)
                        l = loss(output.reshape(-3, -1), y.reshape(-1,))
                    L = L + l.as_in_context(context[0]) / X.size
                    Ls.append(l/X.size)
                    hiddens[j] = h
            L.backward()
            grads = [p.grad(x.context) for p in parameters for x in data_list]
            gluon.utils.clip_global_norm(grads, args.clip)

            trainer.step(1)

            total_L += sum([mx.nd.sum(L).asscalar() for L in Ls])
            trainer.set_learning_rate(lr_batch_start)
            if batch_i % args.log_interval == 0 and batch_i > 0:
                cur_L = total_L / args.log_interval
                print('[Epoch %d Batch %d/%d] loss %.2f, ppl %.2f, '
                      'throughput %.2f samples/s, lr %.2f'
                      %(epoch, batch_i, len(train_data)//args.bptt, cur_L, math.exp(cur_L),
                        args.batch_size*args.log_interval/(time.time()-start_log_interval_time),
                        lr_batch_start*seq_len/args.bptt))
                total_L = 0.0
                start_log_interval_time = time.time()
            i += seq_len
            batch_i += 1

        mx.nd.waitall()

        print('[Epoch %d] throughput %.2f samples/s'%(
            epoch, (args.batch_size * len(train_data)) / (time.time() - start_epoch_time)))
        val_L = evaluate(val_data, val_batch_size, context[0])
        print('[Epoch %d] time cost %.2fs, valid loss %.2f, valid ppl %.2f'%(
            epoch, time.time()-start_epoch_time, val_L, math.exp(val_L)))

        if val_L < best_val:
            update_lr_epoch = 0
            best_val = val_L
            test_L = evaluate(test_data, test_batch_size, context[0])
            model.save_params(args.save)
            print('test loss %.2f, test ppl %.2f'%(test_L, math.exp(test_L)))
        else:
            update_lr_epoch += 1
            if update_lr_epoch % args.lr_update_interval == 0 and update_lr_epoch != 0:
                lr_scale = trainer.learning_rate * args.lr_update_factor
                print('Learning rate after interval update %f'%(lr_scale))
                trainer.set_learning_rate(lr_scale)
                update_lr_epoch = 0

    print('Total training throughput %.2f samples/s'
          %((args.batch_size * len(train_data) * args.epochs) / (time.time() - start_train_time)))


if __name__ == '__main__':
    start_pipeline_time = time.time()
    if not args.eval_only:
        train()
    model.load_params(args.save, context)
    final_val_L = evaluate(val_data, val_batch_size, context[0])
    final_test_L = evaluate(test_data, test_batch_size, context[0])
    print('Best validation loss %.2f, val ppl %.2f'%(final_val_L, math.exp(final_val_L)))
    print('Best test loss %.2f, test ppl %.2f'%(final_test_L, math.exp(final_test_L)))
    print('Total time cost %.2fs'%(time.time()-start_pipeline_time))
