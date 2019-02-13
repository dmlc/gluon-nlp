"""
BERT-based Language Models
==================================

Reference paper will be available soon.
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
import random
import numpy as np
import mxnet as mx
from mxnet import gluon

try:
    from .transformer_lm import bert_lm_12_768_12_300_1150, bert_lm_12_768_12_400_2500, \
        bert_lm_24_1024_16_300_1150, bert_lm_24_1024_16_400_2500
    from .transformer_lm_data import WikiText2WordPiece, WikiText103WordPiece, \
        TransformedCorpusBatchify
except ImportError:
    from transformer_lm import bert_lm_12_768_12_300_1150, bert_lm_12_768_12_400_2500, \
        bert_lm_24_1024_16_300_1150, bert_lm_24_1024_16_400_2500
    from transformer_lm_data import WikiText2WordPiece, WikiText103WordPiece, \
        TransformedCorpusBatchify

parser = argparse.ArgumentParser(
    description='BERT based Language Models')
parser.add_argument('--data', type=str, default='wikitext2',
                    help='language model corpus (wikitext2, wikitext103)')
parser.add_argument('--model', type=str, default='bert_lm_12_768_12_300_1150',
                    help='type of pretrained models (bert_lm_12_768_12_300_1150, '
                         'bert_lm_12_768_12_400_2500, '
                         'bert_lm_24_1024_16_300_1150, '
                         'bert_lm_24_1024_16_400_2500)')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--val_batch_size', type=int, default=10, metavar='N',
                    help='validation batch size')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                    help='test batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--dropoutl', type=float, default=-0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation '
                         '(alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation '
                         '(beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--n_experts', type=int, default=10,
                    help='number of experts')
parser.add_argument('--max_seq_len_delta', type=int, default=40,
                    help='max sequence length')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer to use (sgd, adam, openaiadam, bertadam)')
parser.add_argument('--test_mode', action='store_true',
                    help='test mode')
parser.add_argument('--lr_warmup', type=float, default=0.002)
parser.add_argument('--lr_schedule', default='cosine', type=str,
                    choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant', 'warmup_linear'],
                    help='lr scheduler to use.')
parser.add_argument('--b1', type=float, default=0.9)
parser.add_argument('--b2', type=float, default=0.999)
parser.add_argument('--e', type=float, default=1e-8)
parser.add_argument('--l2', type=float, default=0.01)
parser.add_argument('--vector_l2', action='store_true')
parser.add_argument('--max_grad_norm', type=float, default=1)
parser.add_argument('--use_mos_decoder', action='store_false',
                    help='use mos decoder')
parser.add_argument('--warmup_proportion',
                    default=0.1,
                    type=float,
                    help='Proportion of training to perform linear learning rate warmup for. '
                         'E.g., 0.1 = 10%% of training.')
parser.add_argument('--train_step_factor',
                    default=1,
                    type=float,
                    help='train step factor.')
parser.add_argument('--mask_mode', type=int, default=0,
                    help='(0,1,2,3)')
parser.add_argument('--num_blocks', type=int, default=1,
                    help='(1,2)')
parser.add_argument('--eta_min', type=float, default=0.0,
                    help='min learning rate for cosine scheduler')
parser.add_argument('--max_step', type=int, default=100000,
                    help='upper epoch limit')
parser.add_argument('--warmup_step', type=int, default=0,
                    help='upper epoch limit')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
random.seed(args.seed)
mx.random.seed(args.seed)


def logging(s, print_=True):
    if print_:
        print(s)


context = [mx.cpu()] if args.gpus is None or args.gpus == '' else \
    [mx.gpu(int(x)) for x in args.gpus.split(',')]

assert args.batch_size % len(context) == 0, \
    'Total batch size must be multiple of the number of devices'

###############################################################################
# Load data and batchify data
###############################################################################


if args.data == 'wikitext2':
    val_dataset, test_dataset = \
        [WikiText2WordPiece(segment=segment,
                            skip_empty=False, bos=None, eos=None)
         for segment in ['val', 'test']]
elif args.data == 'wikitext103':
    val_dataset, test_dataset = \
        [WikiText103WordPiece(segment=segment,
                              skip_empty=False, bos=None, eos=None)
         for segment in ['val', 'test']]
else:
    raise NotImplementedError

val_batchify = TransformedCorpusBatchify(args.val_batch_size)
val_data = val_batchify(val_dataset)

test_batchify = TransformedCorpusBatchify(args.test_batch_size)
test_data = test_batchify(test_dataset)

###############################################################################
# Load pre-trained model and BERT vocabulary
###############################################################################


if args.model == 'bert_lm_12_768_12_300_1150':
    model_eval, vocab = bert_lm_12_768_12_300_1150(dataset_name=args.data,
                                                   pretrained=True, ctx=context)
elif args.model == 'bert_lm_12_768_12_400_2500':
    model_eval, vocab = bert_lm_12_768_12_400_2500(dataset_name=args.data,
                                                   pretrained=True, ctx=context)
elif args.model == 'bert_lm_24_1024_16_300_1150':
    model_eval, vocab = bert_lm_24_1024_16_300_1150(dataset_name=args.data,
                                                    pretrained=True, ctx=context)
elif args.model == 'bert_lm_24_1024_16_400_2500':
    model_eval, vocab = bert_lm_24_1024_16_400_2500(dataset_name=args.data,
                                                    pretrained=True, ctx=context)

model_eval.hybridize(static_alloc=True)
logging(model_eval.collect_params())

loss = gluon.loss.SoftmaxCrossEntropyLoss()


###############################################################################
# Training code
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
    data = data_source[i:i + seq_len]
    target = data_source[i + 1:i + 1 + seq_len]
    return data, target


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

    hidden = model_eval.begin_state(batch_size=batch_size, func=mx.nd.zeros, ctx=context[0])
    i = 0
    while i < len(data_source) - 1 - 1:
        data, target = get_batch(data_source, i, seq_len=args.bptt)
        data = data.as_in_context(ctx)
        target = target.as_in_context(ctx)
        output, hidden = model_eval(data, hidden)
        L = loss(output.reshape(-3, -1),
                 target.reshape(-1, ))
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
        i += args.bptt
        hidden = detach(hidden)
    return total_L / ntotal


# Run on validation and test data.
if __name__ == '__main__':
    start_pipeline_time = time.time()
    final_val_L = evaluate(val_data, args.val_batch_size, context[0])
    final_test_L = evaluate(test_data, args.test_batch_size, context[0])
    logging('=' * 89)
    logging('Best validation loss %.2f, val ppl %.2f' % (final_val_L, math.exp(final_val_L)))
    logging('=' * 89)
    logging('Best test loss %.2f, test ppl %.2f' % (final_test_L, math.exp(final_test_L)))
    logging('=' * 89)
    logging('Total time cost %.2fs' % (time.time() - start_pipeline_time))
    logging('=' * 89)
