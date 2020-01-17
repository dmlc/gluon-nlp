"""
Google Neural Machine Translation
=================================

This example shows how to implement the GNMT model with Gluon NLP Toolkit.

@article{wu2016google,
  title={Google's neural machine translation system:
   Bridging the gap between human and machine translation},
  author={Wu, Yonghui and Schuster, Mike and Chen, Zhifeng and Le, Quoc V and
   Norouzi, Mohammad and Macherey, Wolfgang and Krikun, Maxim and Cao, Yuan and Gao, Qin and
   Macherey, Klaus and others},
  journal={arXiv preprint arXiv:1609.08144},
  year={2016}
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
import time
import random
import os
import logging
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp

from gluonnlp.model.translation import NMTModel
from gluonnlp.loss import MaskedSoftmaxCELoss
from gnmt import get_gnmt_encoder_decoder
from translation import BeamSearchTranslator
from utils import logging_config
from bleu import compute_bleu
import dataprocessor
from gluonnlp.estimator import MachineTranslationEstimator, LengthNormalizedLoss
from gluonnlp.estimator import MTGNMTBatchProcessor, MTGNMTGradientUpdateHandler
from gluonnlp.estimator import ComputeBleuHandler, ValBleuHandler
from gluonnlp.estimator import MTTransformerMetricHandler, MTGNMTLearningRateHandler
from gluonnlp.estimator import MTCheckpointHandler, MTTransformerMetricHandler
from gluonnlp.estimator import MTValidationHandler

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)

nlp.utils.check_version('0.9.0')

parser = argparse.ArgumentParser(description='Neural Machine Translation Example.'
                                             'We train the Google NMT model')
parser.add_argument('--dataset', type=str, default='IWSLT2015', help='Dataset to use.')
parser.add_argument('--src_lang', type=str, default='en', help='Source language')
parser.add_argument('--tgt_lang', type=str, default='vi', help='Target language')
parser.add_argument('--epochs', type=int, default=40, help='upper epoch limit')
parser.add_argument('--num_hidden', type=int, default=128, help='Dimension of the embedding '
                                                                'vectors and states.')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in the encoder'
                                                              ' and decoder')
parser.add_argument('--num_bi_layers', type=int, default=1,
                    help='number of bidirectional layers in the encoder and decoder')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--beam_size', type=int, default=4, help='Beam size')
parser.add_argument('--lp_alpha', type=float, default=1.0,
                    help='Alpha used in calculating the length penalty')
parser.add_argument('--lp_k', type=int, default=5, help='K used in calculating the length penalty')
parser.add_argument('--test_batch_size', type=int, default=32, help='Test batch size')
parser.add_argument('--num_buckets', type=int, default=5, help='Bucket number')
parser.add_argument('--bucket_scheme', type=str, default='constant',
                    help='Strategy for generating bucket keys. It supports: '
                         '"constant": all the buckets have the same width; '
                         '"linear": the width of bucket increases linearly; '
                         '"exp": the width of bucket increases exponentially')
parser.add_argument('--bucket_ratio', type=float, default=0.0, help='Ratio for increasing the '
                                                                    'throughput of the bucketing')
parser.add_argument('--src_max_len', type=int, default=50, help='Maximum length of the source '
                                                                'sentence')
parser.add_argument('--tgt_max_len', type=int, default=50, help='Maximum length of the target '
                                                                'sentence')
parser.add_argument('--optimizer', type=str, default='adam', help='optimization algorithm')
parser.add_argument('--lr', type=float, default=1E-3, help='Initial learning rate')
parser.add_argument('--lr_update_factor', type=float, default=0.5,
                    help='Learning rate decay factor')
parser.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save_dir', type=str, default='out_dir',
                    help='directory path to save the final model and training log')
parser.add_argument('--gpu', type=int, default=None,
                    help='id of the gpu to use. Set it to empty means to use cpu.')
args = parser.parse_args()
print(args)
logging_config(args.save_dir)


data_train, data_val, data_test, val_tgt_sentences, test_tgt_sentences, src_vocab, tgt_vocab\
    = dataprocessor.load_translation_data(dataset=args.dataset, bleu='tweaked', args=args)

dataprocessor.write_sentences(val_tgt_sentences, os.path.join(args.save_dir, 'val_gt.txt'))
dataprocessor.write_sentences(test_tgt_sentences, os.path.join(args.save_dir, 'test_gt.txt'))

data_train = data_train.transform(lambda src, tgt: (src, tgt, len(src), len(tgt)), lazy=False)
data_val = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                                     for i, ele in enumerate(data_val)])
data_test = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                                      for i, ele in enumerate(data_test)])
if args.gpu is None:
    ctx = mx.cpu()
    print('Use CPU')
else:
    ctx = mx.gpu(args.gpu)

encoder, decoder, one_step_ahead_decoder = get_gnmt_encoder_decoder(
    hidden_size=args.num_hidden, dropout=args.dropout, num_layers=args.num_layers,
    num_bi_layers=args.num_bi_layers)
model = NMTModel(src_vocab=src_vocab, tgt_vocab=tgt_vocab, encoder=encoder, decoder=decoder,
                 one_step_ahead_decoder=one_step_ahead_decoder, embed_size=args.num_hidden,
                 prefix='gnmt_')
model.initialize(init=mx.init.Uniform(0.1), ctx=ctx)
static_alloc = True
model.hybridize(static_alloc=static_alloc)
logging.info(model)

translator = BeamSearchTranslator(model=model, beam_size=args.beam_size,
                                  scorer=nlp.model.BeamSearchScorer(alpha=args.lp_alpha,
                                                                    K=args.lp_k),
                                  max_length=args.tgt_max_len + 100)
logging.info('Use beam_size={}, alpha={}, K={}'.format(args.beam_size, args.lp_alpha, args.lp_k))


loss_function = MaskedSoftmaxCELoss()
loss_function.hybridize(static_alloc=static_alloc)
trainer = gluon.Trainer(model.collect_params(), args.optimizer, {'learning_rate': args.lr})

train_data_loader, val_data_loader, test_data_loader \
    = dataprocessor.make_dataloader(data_train, data_val, data_test, args)

train_metric = LengthNormalizedLoss(loss_function)
val_metric = LengthNormalizedLoss(loss_function)
batch_processor = MTGNMTBatchProcessor()
gnmt_estimator = MachineTranslationEstimator(net=model, loss=loss_function,
                                             train_metrics=train_metric,
                                             val_metrics=val_metric,
                                             trainer=trainer,
                                             context=ctx,
                                             batch_processor=batch_processor)

learning_rate_handler = MTGNMTLearningRateHandler(epochs=args.epochs,
                                                  lr_update_factor=args.lr_update_factor)

gradient_update_handler = MTGNMTGradientUpdateHandler(clip=args.clip)

metric_handler = MTTransformerMetricHandler(metrics=gnmt_estimator.train_metrics,
                                            grad_interval=1)

bleu_handler = ComputeBleuHandler(tgt_vocab=tgt_vocab, tgt_sentence=val_tgt_sentences,
                                  translator=translator, compute_bleu_fn=compute_bleu)

test_bleu_handler = ComputeBleuHandler(tgt_vocab=tgt_vocab, tgt_sentence=test_tgt_sentences,
                                       translator=translator, compute_bleu_fn=compute_bleu)

val_bleu_handler = ValBleuHandler(val_data=val_data_loader,
                                  val_tgt_vocab=tgt_vocab, val_tgt_sentences=val_tgt_sentences,
                                  translator=translator, compute_bleu_fn=compute_bleu)

checkpoint_handler = MTCheckpointHandler(model_dir=args.save_dir)

val_metric_handler = MTTransformerMetricHandler(metrics=gnmt_estimator.val_metrics)

val_validation_handler = MTValidationHandler(val_data=val_data_loader,
                                             eval_fn=gnmt_estimator.evaluate,
                                             event_handlers=val_metric_handler)

event_handlers = [learning_rate_handler, gradient_update_handler, metric_handler,
                  val_bleu_handler, checkpoint_handler, val_validation_handler]

gnmt_estimator.fit(train_data=train_data_loader,
                   val_data=val_data_loader,
                   #epochs=args.epochs,
                   batches=5,
                   event_handlers=event_handlers,
                   batch_axis=0)

val_event_handlers = [val_metric_handler, bleu_handler]
test_event_handlers = [val_metric_handler, test_bleu_handler]

gnmt_estimator.evaluate(val_data=val_data_loader, event_handlers=val_event_handlers)
gnmt_estimator.evaluate(val_data=test_data_loader, event_handlers=test_event_handlers)
