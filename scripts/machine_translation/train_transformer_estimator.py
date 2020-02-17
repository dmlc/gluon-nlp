"""
Transformer
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
import logging
import os
import random

import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.contrib.estimator import ValidationHandler

import gluonnlp as nlp
from gluonnlp.loss import LabelSmoothing, MaskedSoftmaxCELoss
from gluonnlp.model.transformer import ParallelTransformer, get_transformer_encoder_decoder
from gluonnlp.model.translation import NMTModel
from gluonnlp.metric import LengthNormalizedLoss
from gluonnlp.estimator import MachineTranslationEstimator
from gluonnlp.estimator import MTTransformerBatchProcessor, MTTransformerParamUpdateHandler
from gluonnlp.estimator import TransformerLearningRateHandler, MTTransformerMetricHandler
from gluonnlp.estimator import TransformerGradientAccumulationHandler, ComputeBleuHandler
from gluonnlp.estimator import ValBleuHandler, MTCheckpointHandler
from gluonnlp.estimator import MTTransformerLoggingHandler

import dataprocessor
from bleu import _bpe_to_words, compute_bleu
from translation import BeamSearchTranslator
from utils import logging_config

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)

nlp.utils.check_version('0.9.0')

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='Neural Machine Translation Example with the Transformer Model.')
parser.add_argument('--dataset', type=str.upper, default='WMT2016BPE', help='Dataset to use.',
                    choices=['IWSLT2015', 'WMT2016BPE', 'WMT2014BPE', 'TOY'])
parser.add_argument('--src_lang', type=str, default='en', help='Source language')
parser.add_argument('--tgt_lang', type=str, default='de', help='Target language')
parser.add_argument('--epochs', type=int, default=10, help='upper epoch limit')
parser.add_argument('--num_units', type=int, default=512, help='Dimension of the embedding '
                                                               'vectors and states.')
parser.add_argument('--hidden_size', type=int, default=2048,
                    help='Dimension of the hidden state in position-wise feed-forward networks.')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--epsilon', type=float, default=0.1,
                    help='epsilon parameter for label smoothing')
parser.add_argument('--num_layers', type=int, default=6,
                    help='number of layers in the encoder and decoder')
parser.add_argument('--num_heads', type=int, default=8,
                    help='number of heads in multi-head attention')
parser.add_argument('--scaled', action='store_true', help='Turn on to use scale in attention')
parser.add_argument('--batch_size', type=int, default=1024,
                    help='Batch size. Number of tokens per gpu in a minibatch')
parser.add_argument('--beam_size', type=int, default=4, help='Beam size')
parser.add_argument('--lp_alpha', type=float, default=0.6,
                    help='Alpha used in calculating the length penalty')
parser.add_argument('--lp_k', type=int, default=5, help='K used in calculating the length penalty')
parser.add_argument('--test_batch_size', type=int, default=256, help='Test batch size')
parser.add_argument('--num_buckets', type=int, default=10, help='Bucket number')
parser.add_argument('--bucket_scheme', type=str, default='constant',
                    help='Strategy for generating bucket keys. It supports: '
                         '"constant": all the buckets have the same width; '
                         '"linear": the width of bucket increases linearly; '
                         '"exp": the width of bucket increases exponentially')
parser.add_argument('--bucket_ratio', type=float, default=0.0, help='Ratio for increasing the '
                                                                    'throughput of the bucketing')
parser.add_argument('--src_max_len', type=int, default=-1, help='Maximum length of the source '
                                                                'sentence, -1 means no clipping')
parser.add_argument('--tgt_max_len', type=int, default=-1, help='Maximum length of the target '
                                                                'sentence, -1 means no clipping')
parser.add_argument('--optimizer', type=str, default='adam', help='optimization algorithm')
parser.add_argument('--lr', type=float, default=1.0, help='Initial learning rate')
parser.add_argument('--warmup_steps', type=float, default=4000,
                    help='number of warmup steps used in NOAM\'s stepsize schedule')
parser.add_argument('--num_accumulated', type=int, default=1,
                    help='Number of steps to accumulate the gradients. '
                         'This is useful to mimic large batch training with limited gpu memory')
parser.add_argument('--magnitude', type=float, default=3.0,
                    help='Magnitude of Xavier initialization')
parser.add_argument('--average_checkpoint', action='store_true',
                    help='Turn on to perform final testing based on '
                         'the average of last few checkpoints')
parser.add_argument('--num_averages', type=int, default=5,
                    help='Perform final testing based on the '
                         'average of last num_averages checkpoints. '
                         'This is only used if average_checkpoint is True')
parser.add_argument('--average_start', type=int, default=5,
                    help='Perform average SGD on last average_start epochs')
parser.add_argument('--full', action='store_true',
                    help='In default, we use the test dataset in'
                         ' http://statmt.org/wmt14/test-filtered.tgz.'
                         ' When the option full is turned on, we use the test dataset in'
                         ' http://statmt.org/wmt14/test-full.tgz')
parser.add_argument('--bleu', type=str, default='tweaked',
                    help='Schemes for computing bleu score. It can be: '
                    '"tweaked": it uses similar steps in get_ende_bleu.sh in tensor2tensor '
                    'repository, where compound words are put in ATAT format; '
                    '"13a": This uses official WMT tokenization and produces the same results'
                    ' as official script (mteval-v13a.pl) used by WMT; '
                    '"intl": This use international tokenization in mteval-v14a.pl')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save_dir', type=str, default='transformer_out',
                    help='directory path to save the final model and training log')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.'
                         '(using single gpu is suggested)')
args = parser.parse_args()
logging_config(args.save_dir)
logging.info(args)


data_train, data_val, data_test, val_tgt_sentences, test_tgt_sentences, src_vocab, tgt_vocab \
    = dataprocessor.load_translation_data(dataset=args.dataset, bleu=args.bleu, args=args)

dataprocessor.write_sentences(val_tgt_sentences, os.path.join(args.save_dir, 'val_gt.txt'))
dataprocessor.write_sentences(test_tgt_sentences, os.path.join(args.save_dir, 'test_gt.txt'))

data_train = data_train.transform(lambda src, tgt: (src, tgt, len(src), len(tgt)), lazy=False)
data_val = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                                     for i, ele in enumerate(data_val)])
data_test = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                                      for i, ele in enumerate(data_test)])

ctx = [mx.cpu()] if args.gpus is None or args.gpus == '' else \
    [mx.gpu(int(x)) for x in args.gpus.split(',')]
num_ctxs = len(ctx)

data_train_lengths, data_val_lengths, data_test_lengths = [dataprocessor.get_data_lengths(x)
                                                           for x in
                                                           [data_train, data_val, data_test]]

if args.src_max_len <= 0 or args.tgt_max_len <= 0:
    max_len = np.max(
        [np.max(data_train_lengths, axis=0), np.max(data_val_lengths, axis=0),
         np.max(data_test_lengths, axis=0)],
        axis=0)
if args.src_max_len > 0:
    src_max_len = args.src_max_len
else:
    src_max_len = max_len[0]
if args.tgt_max_len > 0:
    tgt_max_len = args.tgt_max_len
else:
    tgt_max_len = max_len[1]
encoder, decoder, one_step_ahead_decoder = get_transformer_encoder_decoder(
    units=args.num_units, hidden_size=args.hidden_size, dropout=args.dropout,
    num_layers=args.num_layers, num_heads=args.num_heads, max_src_length=max(src_max_len, 500),
    max_tgt_length=max(tgt_max_len, 500), scaled=args.scaled)
model = NMTModel(src_vocab=src_vocab, tgt_vocab=tgt_vocab, encoder=encoder, decoder=decoder,
                 one_step_ahead_decoder=one_step_ahead_decoder,
                 share_embed=args.dataset not in ('TOY', 'IWSLT2015'), embed_size=args.num_units,
                 tie_weights=args.dataset not in ('TOY', 'IWSLT2015'), embed_initializer=None,
                 prefix='transformer_')
model.initialize(init=mx.init.Xavier(magnitude=args.magnitude), ctx=ctx)
static_alloc = True
model.hybridize(static_alloc=static_alloc)
logging.info(model)

translator = BeamSearchTranslator(model=model, beam_size=args.beam_size,
                                  scorer=nlp.model.BeamSearchScorer(alpha=args.lp_alpha,
                                                                    K=args.lp_k),
                                  max_length=200)
logging.info('Use beam_size={}, alpha={}, K={}'.format(args.beam_size, args.lp_alpha, args.lp_k))

label_smoothing = LabelSmoothing(epsilon=args.epsilon, units=len(tgt_vocab))
label_smoothing.hybridize(static_alloc=static_alloc)

loss_function = MaskedSoftmaxCELoss(sparse_label=False)
loss_function.hybridize(static_alloc=static_alloc)

test_loss_function = MaskedSoftmaxCELoss()
test_loss_function.hybridize(static_alloc=static_alloc)

rescale_loss = 100.
parallel_model = ParallelTransformer(model, label_smoothing, loss_function, rescale_loss)
detokenizer = nlp.data.SacreMosesDetokenizer()

trainer = gluon.Trainer(model.collect_params(), args.optimizer,
                        {'learning_rate': args.lr, 'beta2': 0.98, 'epsilon': 1e-9})

train_data_loader, val_data_loader, test_data_loader \
    = dataprocessor.make_dataloader(data_train, data_val, data_test, args,
                                    use_average_length=True, num_shards=len(ctx))

if args.bleu == 'tweaked':
    bpe = bool(args.dataset != 'IWSLT2015' and args.dataset != 'TOY')
    split_compound_word = bpe
    tokenized = True
elif args.bleu == '13a' or args.bleu == 'intl':
    bpe = False
    split_compound_word = False
    tokenized = False
else:
    raise NotImplementedError

grad_interval = args.num_accumulated
average_start = (len(train_data_loader) // grad_interval) * (args.epochs - args.average_start)

train_metric = LengthNormalizedLoss(loss_function)
val_metric = LengthNormalizedLoss(test_loss_function)
batch_processor = MTTransformerBatchProcessor(rescale_loss=rescale_loss,
                                              batch_size=args.batch_size,
                                              label_smoothing=label_smoothing,
                                              loss_function=loss_function)

mt_estimator = MachineTranslationEstimator(net=model, loss=loss_function,
                                           train_metrics=train_metric,
                                           val_metrics=val_metric,
                                           trainer=trainer,
                                           context=ctx,
                                           val_loss=test_loss_function,
                                           batch_processor=batch_processor)

param_update_handler = MTTransformerParamUpdateHandler(avg_start=average_start,
                                                       grad_interval=grad_interval)
learning_rate_handler = TransformerLearningRateHandler(lr=args.lr, num_units=args.num_units,
                                                       warmup_steps=args.warmup_steps,
                                                       grad_interval=grad_interval)
gradient_acc_handler = TransformerGradientAccumulationHandler(grad_interval=grad_interval,
                                                              batch_size=args.batch_size,
                                                              rescale_loss=rescale_loss)
metric_handler = MTTransformerMetricHandler(metrics=mt_estimator.train_metrics,
                                            grad_interval=grad_interval)
bleu_handler = ComputeBleuHandler(tgt_vocab=tgt_vocab, tgt_sentence=val_tgt_sentences,
                                  translator=translator, compute_bleu_fn=compute_bleu,
                                  tokenized=tokenized, tokenizer=args.bleu,
                                  split_compound_word=split_compound_word,
                                  bpe=bpe, bleu=args.bleu, detokenizer=detokenizer,
                                  _bpe_to_words=_bpe_to_words)

test_bleu_handler = ComputeBleuHandler(tgt_vocab=tgt_vocab, tgt_sentence=test_tgt_sentences,
                                       translator=translator, compute_bleu_fn=compute_bleu,
                                       tokenized=tokenized, tokenizer=args.bleu,
                                       split_compound_word=split_compound_word,
                                       bpe=bpe, bleu=args.bleu, detokenizer=detokenizer,
                                       _bpe_to_words=_bpe_to_words)

val_bleu_handler = ValBleuHandler(val_data=val_data_loader, val_tgt_vocab=tgt_vocab,
                                  val_tgt_sentences=val_tgt_sentences, translator=translator,
                                  tokenized=tokenized, tokenizer=args.bleu,
                                  split_compound_word=split_compound_word, bpe=bpe,
                                  compute_bleu_fn=compute_bleu,
                                  bleu=args.bleu, detokenizer=detokenizer,
                                  _bpe_to_words=_bpe_to_words)

checkpoint_handler = MTCheckpointHandler(model_dir=args.save_dir,
                                         average_checkpoint=args.average_checkpoint,
                                         num_averages=args.num_averages,
                                         average_start=args.average_start)

val_metric_handler = MTTransformerMetricHandler(metrics=mt_estimator.val_metrics)

val_validation_handler = ValidationHandler(val_data=val_data_loader,
                                           eval_fn=mt_estimator.evaluate,
                                           event_handlers=val_metric_handler)

log_interval = args.log_interval * grad_interval
logging_handler = MTTransformerLoggingHandler(log_interval=log_interval,
                                              metrics=mt_estimator.train_metrics)

event_handlers = [param_update_handler,
                  learning_rate_handler,
                  gradient_acc_handler,
                  metric_handler,
                  val_validation_handler,
                  val_bleu_handler,
                  checkpoint_handler,
                  logging_handler]

mt_estimator.fit(train_data=train_data_loader,
                 val_data=val_data_loader,
                 epochs=args.epochs,
                 event_handlers=event_handlers,
                 batch_axis=0)

val_event_handlers = [val_metric_handler,
                      bleu_handler]

test_event_handlers = [val_metric_handler,
                       test_bleu_handler]

mt_estimator.evaluate(val_data=val_data_loader, event_handlers=val_event_handlers)

mt_estimator.evaluate(val_data=test_data_loader, event_handlers=test_event_handlers)
