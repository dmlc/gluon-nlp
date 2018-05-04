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

import argparse
import time
import math
import os
import sys
import mxnet as mx
from mxnet import gluon, autograd
import gluonnlp as nlp

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '..', '..'))


parser = argparse.ArgumentParser(description=
                                 'MXNet Autograd RNN/LSTM Language Model on Wikitext-2.')
parser.add_argument('--model', type=str, default='lstm',
                    help='type of recurrent net (rnn_tanh, rnn_relu, lstm, gru)')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--save', type=str, default='model.params',
                    help='path to save the final model')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.'
                         '(using single gpu is suggested)')
parser.add_argument('--window', type=int, default=3785,
                    help='pointer window length')
parser.add_argument('--theta', type=float, default=0.6625523432485668,
                    help='mix between uniform distribution and pointer softmax distribution over previous words')
parser.add_argument('--lambdasm', type=float, default=0.12785920428335693,
                    help='linear mix between only pointer (1) and only vocab (0) distribution')
args = parser.parse_args()

###############################################################################
# Load data
###############################################################################

context = [mx.cpu()] if args.gpus is None or args.gpus == '' else \
          [mx.gpu(int(x)) for x in args.gpus.split(',')]

assert args.batch_size % len(context) == 0, \
    'Total batch size must be multiple of the number of devices'

assert args.weight_dropout > 0 or (args.weight_dropout == 0 and args.alpha == 0), \
    'The alpha L2 regularization cannot be used with standard RNN, please set alpha to 0'

train_dataset, val_dataset, test_dataset = \
    [nlp.data.WikiText2(segment=segment,
                        skip_empty=False, bos=None, eos='<eos>')
     for segment in ['train', 'val', 'test']]

vocab = nlp.Vocab(counter=nlp.data.Counter(train_dataset[0]), padding_token=None, bos_token=None)

train_data = train_dataset.batchify(vocab, args.batch_size)
val_batch_size = 10
val_data = val_dataset.batchify(vocab, val_batch_size)
test_batch_size = 1
test_data = test_dataset.batchify(vocab, test_batch_size)

print(args)

###############################################################################
# Build the model
###############################################################################


ntokens = len(vocab)

if args.weight_dropout > 0:
    print('Use AWDRNN')
    model = nlp.model.language_model.AWDRNN(args.model, len(vocab), args.emsize,
                                            args.nhid, args.nlayers, args.tied,
                                            args.dropout, args.weight_dropout, args.dropout_h,
                                            args.dropout_i, args.dropout_e)
else:
    model = nlp.model.language_model.StandardRNN(args.model, len(vocab), args.emsize,
                                                 args.nhid, args.nlayers, args.dropout, args.tied)

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
        hidden = [detach(h) for h in hidden]
    else:
        hidden = hidden.detach()
    return hidden




def cache_forward(inputs, begin_state=None):
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
    if args.weight_dropout > 0:
        for i, (e, s) in enumerate(zip(model.encoder, begin_state)):
            encoded, state = e(encoded, s)
            encoded_raw.append(encoded)
            out_states.append(state)
            if model._drop_h and i != len(model.encoder)-1:
                encoded = mx.nd.Dropout(encoded, p=model._drop_h, axes=(0,))
                encoded_dropped.append(encoded)
    else:
        encoded, state = model.encoder(encoded, begin_state)
        encoded_raw.append(encoded)
    if model._dropout:
        encoded = mx.nd.Dropout(encoded, p=model._dropout, axes=(0,))
    if args.weight_dropout > 0:
        encoded_dropped.append(encoded)
        with autograd.predict_mode():
            out = model.decoder(encoded)
    else:
        out = model.decoder(encoded)
    if args.weight_dropout > 0:
        return out, out_states, encoded_raw, encoded_dropped
    else:
        return out, state, encoded_raw, encoded_dropped

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


# cache model on the fly
# TODO: can add to attention model?
def cache(data_source, batch_size, ctx=None):
    total_L = 0
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=args.batch_size, ctx=context[0])
    next_word_history = None
    cache_history = None
    for i in range(0, len(data_source) - 1, args.bptt):
        data, target = get_batch(data_source, i)
        data = data.as_in_context(ctx)
        target = target.as_in_context(ctx)
        output, hidden, rnn_out, rnn_out_dropped = cache_forward(data, hidden)
        ##rnn_out
        rnn_out = mx.nd.reshape(rnn_out, (-1,))
        output_flat = mx.nd.reshape(output, (-1,))

        start_idx = len(next_word_history) if next_word_history is not None else 0
        next_word_history = mx.nd.concat([mx.nd.one_hot([t], on_value=1, off_value=0) for t in target],
                                         dim=0) if next_word_history is None else mx.nd.concat(
            [next_word_history, mx.nd.one_hot([t], on_value=1, off_value=0) for t in target])

        ##cache_history
        cache_history = rnn_out if cache_history is None else mx.nd.concat(cache_history, rnn_out, dim=0)

        L = 0
        softmax_output_flat = mx.nd.softmax(output_flat, axis=0)
        for idx, vocab_L in enumerate(softmax_output_flat):
            p = vocab_L
            if start_idx + idx > args.window:
                valid_next_word = next_word_history[start_idx + idx - args.window:start_idx + idx]
                valid_cache_history = cache_history[start_idx + idx - args.window:start_idx + idx]
                logits = mx.nd.multiply(valid_cache_history, rnn_out)
                cache_attn = mx.nd.reshape(mx.nd.softmax(args.theta * logits), (-1,))
                # broadcast_to
                # squeeze
                cache_dist = mx.nd.reshape(
                    mx.nd.sum((cache_attn.broadcast_to(valid_next_word.shape) * valid_next_word), axis=0), (-3,))
                p = args.lambdah * cache_dist + (1 - args.lambdah) * vocab_L
            target_L = p[target]
            L += -mx.nd.log(target_L)
        total_L += L / args.batch_size

        next_word_history = next_word_history[-args.window:]
        cache_history = cache_history[-args.window:]
    return total_L / len(datasource)


if __name__ == '__main__':
    start_pipeline_time = time.time()
    model.load_params(args.save, context)
    final_val_L = cache(val_data, val_batch_size, context[0])
    final_test_L = cache(test_data, test_batch_size, context[0])
    print('Best validation loss %.2f, val ppl %.2f'%(final_val_L, math.exp(final_val_L)))
    print('Best test loss %.2f, test ppl %.2f'%(final_test_L, math.exp(final_test_L)))
    print('Total time cost %.2fs'%(time.time()-start_pipeline_time))
