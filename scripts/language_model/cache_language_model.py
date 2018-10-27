"""
Neural Cache Language Model
===================
This example shows how to build a neural cache language model based on
pre-trained word-level language model on WikiText-2 with Gluon NLP Toolkit.

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
parser.add_argument('--bptt', type=int, default=2000,
                    help='sequence length')
parser.add_argument('--model_name', type=str, default='awd_lstm_lm_1150',
                    help='name of the pre-trained language model')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.'
                         '(using single gpu is suggested)')
parser.add_argument('--window', type=int, default=2000,
                    help='cache window length')
parser.add_argument('--theta', type=float, default=0.662,
                    help='the scala controls the flatness of the cache distribution '
                         'that predict the next word')
parser.add_argument('--lambdas', type=float, default=0.1279,
                    help='linear scalar between only cache and vocab distribution')
parser.add_argument('--path_to_params_file', type=str, default=None,
                    help='path to the saved params file of user pre-trained model, '
                         'including the params file, e.g., ~/.mxnet/models/awd_lstm_lm_1150.params')
args = parser.parse_args()

###############################################################################
# Load vocabulary
###############################################################################

context = [mx.cpu()] if args.gpus is None or args.gpus == '' else \
          [mx.gpu(int(x)) for x in args.gpus.split(',')]

print(args)

_, vocab = nlp.model.get_model(name=args.model_name,
                               dataset_name='wikitext-2',
                               pretrained=False,
                               ctx=context)
ntokens = len(vocab)

###############################################################################
# Build the cache model and load pre-trained language model
###############################################################################


if not args.path_to_params_file:
    cache_cell = nlp.model.train.get_cache_model(name=args.model_name,
                                                 dataset_name='wikitext-2',
                                                 window=args.window,
                                                 theta=args.theta,
                                                 lambdas=args.lambdas,
                                                 ctx=context)
else:
    model, _ = nlp.model.get_model(name=args.model_name,
                                   dataset_name='wikitext-2',
                                   pretrained=False,
                                   ctx=context)
    cache_cell = nlp.model.train.CacheCell(model, ntokens, args.window, args.theta, args.lambdas)
    cache_cell.load_parameters(args.path_to_params_file, ctx=context)

###############################################################################
# Load data
###############################################################################

val_dataset, test_dataset = \
    [nlp.data.WikiText2(segment=segment,
                        skip_empty=False, bos=None, eos='<eos>')
     for segment in ['val', 'test']]

val_batch_size = 1
val_batchify = nlp.data.batchify.CorpusBatchify(vocab, val_batch_size)
val_data = val_batchify(val_dataset)
test_batch_size = 1
test_batchify = nlp.data.batchify.CorpusBatchify(vocab, test_batch_size)
test_data = test_batchify(test_dataset)

###############################################################################
# Training
###############################################################################


def detach(hidden):
    """Transfer hidden states into new states, to detach them from the history.
    Parameters
    ----------
    hidden : NDArray
        The hidden states
    Returns
    ----------
    hidden: NDArray
        The detached hidden states
    """
    if isinstance(hidden, (tuple, list)):
        hidden = [detach(h) for h in hidden]
    else:
        hidden = hidden.detach()
    return hidden


def get_batch(data_source, i, seq_len=None):
    """Get mini-batches of the dataset.

    Parameters
    ----------
    data_source : NDArray
        The dataset is evaluated on.
    i : int
        The index of the batch, starting from 0.
    seq_len : int
        The length of each sample in the batch.

    Returns
    -------
    data: NDArray
        The context
    target: NDArray
        The words to predict
    """
    seq_len = min(seq_len if seq_len else args.bptt, len(data_source) - 1 - i)
    data = data_source[i:i+seq_len]
    target = data_source[i+1:i+1+seq_len]
    return data, target


def evaluate(data_source, batch_size, ctx=None):
    """Evaluate the model on the dataset with cache model.

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
    total_L = 0
    hidden = cache_cell.\
        begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context[0])
    next_word_history = None
    cache_history = None
    for i in range(0, len(data_source) - 1, args.bptt):
        if i > 0:
            print('Batch %d/%d, ppl %f'%
                  (i, len(data_source), math.exp(total_L/i)))
        data, target = get_batch(data_source, i)
        data = data.as_in_context(ctx)
        target = target.as_in_context(ctx)
        L = 0
        outs, next_word_history, cache_history, hidden = \
            cache_cell(data, target, next_word_history, cache_history, hidden)
        for out in outs:
            L += (-mx.nd.log(out)).asscalar()
        total_L += L / data.shape[1]
        hidden = detach(hidden)
    return total_L / len(data_source)


if __name__ == '__main__':
    start_pipeline_time = time.time()
    final_val_L = evaluate(val_data, val_batch_size, context[0])
    final_test_L = evaluate(test_data, test_batch_size, context[0])
    print('Best validation loss %.2f, val ppl %.2f' % (final_val_L, math.exp(final_val_L)))
    print('Best test loss %.2f, test ppl %.2f' % (final_test_L, math.exp(final_test_L)))
    print('Total time cost %.2fs' % (time.time()-start_pipeline_time))
