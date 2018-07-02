"""
Large Word Language Model
===================

This example shows how to build a word-level language model on Google Billion Words dataset
with Gluon NLP Toolkit.
By using the existing data pipeline tools and building blocks, the process is greatly simplified.

We implement the LSTM 2048-512 language model proposed in the following work.

@article{jozefowicz2016exploring,
 title={Exploring the Limits of Language Modeling},
 author={Jozefowicz, Rafal and Vinyals, Oriol and Schuster, Mike and Shazeer, Noam and Wu, Yonghui},
 journal={arXiv preprint arXiv:1602.02410},
 year={2016}
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

import time
import math
import os
import sys
import io
import argparse
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
import gluonnlp as nlp
from sampler import LogUniformSampler

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '..', '..'))

###############################################################################
# Arg parser
###############################################################################
parser = argparse.ArgumentParser(description=
                                 'Gluon-NLP Big LSTM 2048-512 Language Model on GBW')
parser.add_argument('--save', type=str, default='model.params',
                    help='path to save the final model.')
parser.add_argument('--emsize', type=int, default=512,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=2048,
                    help='number of hidden units per layer')
parser.add_argument('--nproj', type=int, default=512,
                    help='number of projection units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--from-epoch', type=int, default=None,
                    help='start training or testing from the provided epoch')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epoch for training')
parser.add_argument('--batch-size', type=int, default=256,
                    help='batch size per gpu')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--eps', type=float, default=1,
                    help='initial history accumulation or adagrad')
parser.add_argument('--bptt', type=int, default=20,
                    help='sequence length')
parser.add_argument('--k', type=int, default=8192,
                    help='number of noise samples for estimation')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.')
parser.add_argument('--log-interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--lr', type=float, default=0.2,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=10.0,
                    help='gradient clipping by global norm.')
parser.add_argument('--test-mode', action='store_true',
                    help='Whether to run through the script with few examples')
parser.add_argument('--eval-only', action='store_true',
                    help='Whether to only run evluation for the trained model')
args = parser.parse_args()

segments = ['train', 'test']
max_nbatch_eval = None

if args.test_mode:
    args.emsize = 200
    args.log_interval = 1
    args.nhid = 200
    args.nlayers = 1
    args.epochs = 20
    max_nbatch_eval = 3
    segments = ['test', 'test']

print(args)
mx.random.seed(args.seed)
np.random.seed(args.seed)

###############################################################################
# Vocab
###############################################################################
with io.open('gbw_vocab.json', 'r', encoding='utf-8') as in_file:
    vocab = nlp.Vocab.from_json(in_file.read())
ntokens = len(vocab)

###############################################################################
# Load data
###############################################################################

context = [mx.cpu()] if args.gpus is None or args.gpus == '' else \
          [mx.gpu(int(x)) for x in args.gpus.split(',')]

train_data_stream, test_data_stream = \
    [nlp.data.GBWStream(segment=segment, skip_empty=True, bos='<eos>', eos='<eos>')
     for segment in segments]

sampler = LogUniformSampler(ntokens, args.k)

def _load(xs):
    ret = []
    for x, ctx in zip(xs, context):
        if isinstance(x, tuple):
            ret.append([y.as_in_context(ctx) for y in x])
        else:
            ret.append(x.as_in_context(ctx))
    return ret

def _split_and_sample(data):
    x, y, m = data
    num_ctx = len(context)
    if num_ctx > 1:
        xs = gluon.utils.split_data(x, num_ctx, batch_axis=1, even_split=True)
        ys = gluon.utils.split_data(y, num_ctx, batch_axis=1, even_split=True)
        ms = gluon.utils.split_data(m, num_ctx, batch_axis=1, even_split=True)
    else:
        xs, ys, ms = [x], [y], [m]
    ss = [sampler(x) for x in xs]
    xs = _load(xs)
    ys = _load(ys)
    ms = _load(ms)
    ss = _load(ss)
    return xs, ys, ms, ss

train_batch_size = args.batch_size * len(context)
train_data = train_data_stream.bptt_batchify(vocab, args.bptt, train_batch_size)
train_data = train_data.transform(_split_and_sample)
train_data = nlp.data.PrefetchingStream(train_data)

test_batch_size = args.batch_size
test_data = test_data_stream.bptt_batchify(vocab, args.bptt, test_batch_size)
test_data = nlp.data.PrefetchingStream(test_data)

###############################################################################
# Build the model
###############################################################################


eval_model = nlp.model.language_model.BigRNN(ntokens, args.emsize, args.nhid,
                                             args.nlayers, args.nproj,
                                             dropout=args.dropout)
model = nlp.model.language_model.train.BigRNN(ntokens, args.emsize, args.nhid,
                                              args.nlayers, args.nproj, args.k,
                                              dropout=args.dropout)
loss = gluon.loss.SoftmaxCrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def train():
    """Training loop for language model.
    """
    print(model)
    from_epoch = 0
    model.initialize(mx.init.Xavier(), ctx=context)
    if args.from_epoch:
        from_epoch = args.from_epoch - 1
        checkpoint_name = '%s.%s'%(args.save, format(from_epoch, '02d'))
        model.load_parameters(checkpoint_name)
        # warning: hidden states are not loaded from the previous checkpoint
        print('Loaded parameters from checkpoint %s'%(checkpoint_name))

    model.hybridize(static_alloc=True, static_shape=True)
    trainer_params = {'learning_rate': args.lr, 'wd': 0, 'eps': args.eps}
    trainer = gluon.Trainer(model.collect_params(), 'adagrad', trainer_params)

    encoder_params = model.encoder.collect_params().values()
    embedding_params = list(model.embedding.collect_params().values())
    for epoch in range(from_epoch, args.epochs):
        sys.stdout.flush()
        total_L = 0.0
        start_epoch_time = time.time()
        start_log_interval_time = time.time()
        hiddens = [model.begin_state(batch_size=args.batch_size,
                                     func=mx.nd.zeros, ctx=ctx) for ctx in context]
        nbatch = 0

        for data, target, mask, sample in train_data:
            nbatch += 1
            hiddens = detach(hiddens)
            Ls = []
            with autograd.record():
                for j, (X, y, m, s, h) in enumerate(zip(data, target, mask, sample, hiddens)):
                    output, new_target, h = model(X, y, h, s)
                    l = loss(output, new_target) * m.reshape((-1,))
                    Ls.append(l/args.batch_size)
                    hiddens[j] = h

            autograd.backward(Ls)

            # rescale embedding grad
            for d in data:
                x = embedding_params[0].grad(d.context)
                x[:] *= args.batch_size
                encoder_grad = [p.grad(d.context) for p in encoder_params]
                # perform gradient clipping per ctx
                gluon.utils.clip_global_norm(encoder_grad, args.clip)

            trainer.step(len(context))

            total_L += sum([mx.nd.sum(L).asscalar() / args.bptt for L in Ls])

            if nbatch % args.log_interval == 0:
                cur_L = total_L / args.log_interval / len(context)
                ppl = math.exp(cur_L) if cur_L < 100 else 99999999
                print('[Epoch %d Batch %d] loss %.2f, ppl %.2f, '
                      'throughput %.2f samples/s'
                      %(epoch, nbatch, cur_L, ppl,
                        train_batch_size*args.log_interval/(time.time()-start_log_interval_time)))
                total_L = 0.0
                start_log_interval_time = time.time()
        end_epoch_time = time.time()
        print('Epoch %d took %.2f seconds.'%(epoch, end_epoch_time - start_epoch_time))
        mx.nd.waitall()
        checkpoint_name = '%s.%s'%(args.save, format(epoch, '02d'))
        model.save_parameters(checkpoint_name)

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [detach(h) for h in hidden]
    else:
        hidden = hidden.detach()
    return hidden

def test(data_stream, batch_size, ctx=None):
    """Evaluate the model on the dataset.

    Parameters
    ----------
    data_stream : DataStream
        The dataset is testd on.
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
        total_L += L.sum()
        ntotal += mask.sum()
        nbatch += 1
        avg = (total_L / ntotal).asscalar()
        if nbatch % args.log_interval == 0:
            print('Evaluation batch %d: test loss %.2f, test ppl %.2f'
                  %(nbatch, avg, math.exp(avg)))
        if max_nbatch_eval and nbatch > max_nbatch_eval:
            print('Quit evaluation early at batch %d'%nbatch)
            break
    return avg

def evaluate():
    """ Evaluate loop for the trained model """
    print(eval_model)
    eval_model.initialize(mx.init.Xavier(), ctx=context[0])
    eval_model.hybridize(static_alloc=True, static_shape=True)
    epoch = args.from_epoch if args.from_epoch else 0
    while epoch < args.epochs:
        checkpoint_name = '%s.%s'%(args.save, format(epoch, '02d'))
        if not os.path.exists(checkpoint_name):
            print('Wait for a new checkpoint...')
            # check again after 600 seconds
            time.sleep(600)
            continue
        eval_model.load_parameters(checkpoint_name)
        print('Loaded parameters from checkpoint %s'%(checkpoint_name))
        final_test_L = test(test_data, test_batch_size, ctx=context[0])
        print('[Epoch %d] test loss %.2f, test ppl %.2f'%
              (epoch, final_test_L, math.exp(final_test_L)))
        sys.stdout.flush()
        epoch += 1

if __name__ == '__main__':
    start_pipeline_time = time.time()
    if args.eval_only:
        evaluate()
    else:
        train()
