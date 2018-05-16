"""
Neural Cache Language Model
===================
This example shows how to build a neural cache language model based on pretrained word-level language model
on WikiText-2 with Gluon NLP Toolkit.

We implement the neural cache language model proposed in the following work.
@article{grave2016improving,
  title={Improving neural language models with a continuous cache},
  author={Grave, Edouard and Joulin, Armand and Usunier, Nicolas},
  journal={ICLR},
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

import argparse
import time
import math
import os
import sys
import mxnet as mx
import gluonnlp as nlp

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '..', '..'))


parser = argparse.ArgumentParser(description=
                                 'MXNet Neural Cache Language Model on Wikitext-2.')
# parser.add_argument('--model', type=str, default='lstm',
#                     help='type of recurrent net (rnn_tanh, rnn_relu, lstm, gru)')
parser.add_argument('--bptt', type=int, default=2000,
                    help='sequence length')
parser.add_argument('--save', type=str, default='awd_lstm_lm_1150',
                    help='name of the pretrained language model')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.'
                         '(using single gpu is suggested)')
parser.add_argument('--window', type=int, default=2000,
                    help='pointer window length')
parser.add_argument('--theta', type=float, default=0.662,
                    help='mix between uniform distribution and pointer softmax distribution over previous words')
parser.add_argument('--lambdas', type=float, default=0.1279,
                    help='linear mix between only pointer (1) and only vocab (0) distribution')
# parser.add_argument('--weight_dropout', type=float, default=0.5,
#                     help='weight dropout applied to h2h weight matrix (0 = no weight dropout)')
# parser.add_argument('--use_customerized_pretrained_model', action='store_true',
#                     help='whether to apply cache model to user pretrained model')
args = parser.parse_args()

###############################################################################
# Load data
###############################################################################

context = [mx.cpu()] if args.gpus is None or args.gpus == '' else \
          [mx.gpu(int(x)) for x in args.gpus.split(',')]

print(args)

val_dataset, test_dataset = \
    [nlp.data.WikiText2(segment=segment,
                        skip_empty=False, bos=None, eos='<eos>')
     for segment in ['val', 'test']]

###############################################################################
# Build the model
###############################################################################
model, vocab = nlp.model.get_model(name=args.save,
                                      dataset_name='wikitext-2',
                                      pretrained=True,
                                      ctx=context)

val_batch_size = 1
val_data = val_dataset.batchify(vocab, val_batch_size)
test_batch_size = 1
test_data = test_dataset.batchify(vocab, test_batch_size)

ntokens = len(vocab)

def get_batch(data_source, i, seq_len=None):
    seq_len = min(seq_len if seq_len else args.bptt, len(data_source) - 1 - i)
    data = data_source[i:i+seq_len]
    target = data_source[i+1:i+1+seq_len]
    return data, target

def evaluate(data_source, batch_size, ctx=None):
    total_L = 0
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context[0])
    next_word_history = None
    cache_history = None
    cache_cell = nlp.model.CacheCell(model, ntokens, args.window, args.theta, args.lambdas)
    for i in range(0, len(data_source) - 1, args.bptt):
        if i > 0: print('Batch %d/%d, loss %f'%(i, len(data_source), math.exp(total_L/i)))
        data, target = get_batch(data_source, i)
        data = data.as_in_context(ctx)
        target = target.as_in_context(ctx)
        L = 0
        outs, next_word_history, cache_history = cache_cell(data, target, next_word_history, cache_history, hidden)
        for out in outs:
            L += (-mx.nd.log(out)).asscalar()
        total_L += L / data.shape[1]
        hidden = nlp.model.detach(hidden)
    return total_L / len(data_source)


if __name__ == '__main__':
    start_pipeline_time = time.time()
    # model.load_params(args.save, context)
    final_val_L = evaluate(val_data, val_batch_size, context[0])
    final_test_L = evaluate(test_data, test_batch_size, context[0])
    print('Best validation loss %.2f, val ppl %.2f'%(final_val_L, math.exp(final_val_L)))
    print('Best test loss %.2f, test ppl %.2f'%(final_test_L, math.exp(final_test_L)))
    print('Total time cost %.2fs'%(time.time()-start_pipeline_time))
