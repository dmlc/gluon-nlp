"""
Neural Cache Language Model
===================
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
parser.add_argument('--bptt', type=int, default=5000,
                    help='sequence length')
parser.add_argument('--save', type=str, default='awd_lstm_lm_1150_wikitext-2',
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
parser.add_argument('--weight_dropout', type=float, default=0.5,
                    help='weight dropout applied to h2h weight matrix (0 = no weight dropout)')
args = parser.parse_args()

###############################################################################
# Load data
###############################################################################

context = [mx.cpu()] if args.gpus is None or args.gpus == '' else \
          [mx.gpu(int(x)) for x in args.gpus.split(',')]

train_dataset, val_dataset, test_dataset = \
    [nlp.data.WikiText2(segment=segment,
                        skip_empty=False, bos=None, eos='<eos>')
     for segment in ['train', 'val', 'test']]

vocab = nlp.Vocab(counter=nlp.data.Counter(train_dataset[0]), padding_token=None, bos_token=None)

val_batch_size = 1
val_data = val_dataset.batchify(vocab, val_batch_size)
test_batch_size = 1
test_data = test_dataset.batchify(vocab, test_batch_size)

print(args)

###############################################################################
# Build the model
###############################################################################


ntokens = len(vocab)

model = nlp.model.language_model.AWDRNN(args.model, len(vocab))
model.initialize(mx.init.Xavier(), ctx=context)

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

def cache(data_source, batch_size, ctx=None):
    total_L = 0
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context[0])
    next_word_history = None
    cache_history = None
    for i in range(0, len(data_source) - 1, args.bptt):
        if i > 0: print(i, len(data_source), math.exp(total_L/i))
        data, target = get_batch(data_source, i)
        data = data.as_in_context(ctx)
        target = target.as_in_context(ctx)
        output, hidden, encoder_hs, dropped_encoder_hs = cache_forward(data, hidden)
        encoder_h = encoder_hs[-1].reshape(-3,-2)
        output = output.reshape(-1, ntokens)

        start_idx = len(next_word_history) if next_word_history is not None else 0
        next_word_history = mx.nd.concat(*[mx.nd.one_hot(t[0], ntokens, on_value=1, off_value=0) for t in target], dim=0) \
            if next_word_history is None else mx.nd.concat(next_word_history, mx.nd.concat(*[mx.nd.one_hot(t[0], ntokens, on_value=1, off_value=0) for t in target], dim=0), dim=0)
        print(next_word_history)
        cache_history = encoder_h if cache_history is None else mx.nd.concat(cache_history, encoder_h, dim=0)
        print(cache_history)

        L = 0
        softmax_output = mx.nd.softmax(output)
        for idx, vocab_L in enumerate(softmax_output):
            joint_p = vocab_L
            if start_idx + idx > args.window:
                valid_next_word = next_word_history[start_idx + idx - args.window:start_idx + idx]
                valid_cache_history = cache_history[start_idx + idx - args.window:start_idx + idx]
                logits = mx.nd.dot(valid_cache_history, encoder_h[idx])
                cache_attn = mx.nd.softmax(args.theta * logits).reshape(-1,1)
                cache_dist = (cache_attn.broadcast_to(valid_next_word.shape) * valid_next_word).sum(axis=0)
                joint_p = args.lambdasm * cache_dist + (1 - args.lambdasm) * vocab_L
            target_L = joint_p[target[idx]]
            L += (-mx.nd.log(target_L)).asscalar()
        total_L += L / batch_size

        hidden = detach(hidden)
        next_word_history = next_word_history[-args.window:]
        cache_history = cache_history[-args.window:]
    return total_L / len(data_source)


if __name__ == '__main__':
    start_pipeline_time = time.time()
    model.load_params(args.save, context)
    final_val_L = cache(val_data, val_batch_size, context[0])
    final_test_L = cache(test_data, test_batch_size, context[0])
    print('Best validation loss %.2f, val ppl %.2f'%(final_val_L, math.exp(final_val_L)))
    print('Best test loss %.2f, test ppl %.2f'%(final_test_L, math.exp(final_test_L)))
    print('Total time cost %.2fs'%(time.time()-start_pipeline_time))
