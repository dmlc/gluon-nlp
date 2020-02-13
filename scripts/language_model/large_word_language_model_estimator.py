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
import argparse
import re

import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon.contrib.estimator import CheckpointHandler, LoggingHandler
import gluonnlp as nlp
from gluonnlp.utils import Parallel, Parallelizable
from sampler import LogUniformSampler
from gluonnlp.estimator import ParallelLanguageModelBatchProcessor
from gluonnlp.estimator import HiddenStateHandler, MetricResetHandler
from gluonnlp.estimator import LargeRNNGradientUpdateHandler
from gluonnlp.estimator import WordLanguageModelCheckpointHandler
from gluonnlp.estimator import LanguageModelEstimator
from gluonnlp.estimator import ParallelLoggingHandler
from gluonnlp.estimator.length_normalized_loss import LengthNormalizedLoss

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '..', '..'))

nlp.utils.check_version('0.7.0')

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
                    help='number of projection units per layer. Could be different from embsize')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--from-epoch', type=int, default=None,
                    help='start training or testing from the provided epoch')
parser.add_argument('--epochs', type=int, default=50,
                    help='number of epoch for training')
parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size per gpu')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--eps', type=float, default=1,
                    help='initial history accumulation for adagrad')
parser.add_argument('--bptt', type=int, default=20,
                    help='sequence length')
parser.add_argument('--k', type=int, default=8192,
                    help='number of noise samples for estimation')
parser.add_argument('--gpus', type=str,
                    help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu.')
parser.add_argument('--log-interval', type=int, default=1000,
                    help='report interval')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed')
parser.add_argument('--lr', type=float, default=0.2,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clipping by global norm.')
parser.add_argument('--test-mode', action='store_true',
                    help='Whether to run through the script with few examples')
parser.add_argument('--eval-only', action='store_true',
                    help='Whether to only run evaluation for the trained model')
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

context = [mx.cpu()] if args.gpus is None or args.gpus == '' else \
          [mx.gpu(int(x)) for x in args.gpus.split(',')]

os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
os.environ['MXNET_CPU_PARALLEL_RAND_COPY'] = str(len(context))
os.environ['MXNET_CPU_WORKER_NTHREADS'] = str(len(context))

###############################################################################
# Data stream
###############################################################################
train_data_stream, test_data_stream = \
    [nlp.data.GBWStream(segment=segment, skip_empty=True, bos=None, eos='<eos>')
     for segment in segments]
vocab = train_data_stream.vocab
ntokens = len(vocab)

# Sampler for generating negative classes during training with importance sampling
sampler = LogUniformSampler(ntokens, args.k)

# Given a list of (array, context) pairs, load array[i] on context[i]
def _load(xs):
    ret = []
    for x, ctx in zip(xs, context):
        if isinstance(x, tuple):
            ret.append([y.as_in_context(ctx) for y in x])
        else:
            ret.append(x.as_in_context(ctx))
    return ret

# Transformation for a data batch for training.
# First, load the data, target and mask to target contexts.
# Second, the LSTM-2048-512 model performs importance sampling for decoding
# during training, we need to sample negative candidate classes by invoking the
# log uniform sampler.
def _split_and_sample(x, y):
    m = x != vocab[vocab.padding_token]  # mask padding
    num_ctx = len(context)
    if num_ctx > 1:
        xs = gluon.utils.split_data(x, num_ctx, batch_axis=1, even_split=True)
        ys = gluon.utils.split_data(y, num_ctx, batch_axis=1, even_split=True)
        ms = gluon.utils.split_data(m, num_ctx, batch_axis=1, even_split=True)
    else:
        xs, ys, ms = [x], [y], [m]
    xs = _load(xs)
    ys = _load(ys)
    ms = _load(ms)
    ss = [sampler(y) for y in ys]
    ss = _load(ss)
    return xs, ys, ms, ss

train_batch_size = args.batch_size * len(context)
train_batchify = nlp.data.batchify.StreamBPTTBatchify(vocab, args.bptt, train_batch_size)
train_data = train_batchify(train_data_stream)
train_data = train_data.transform(_split_and_sample)

test_batch_size = args.batch_size
test_batchify = nlp.data.batchify.StreamBPTTBatchify(vocab, args.bptt, test_batch_size)
test_data = test_batchify(test_data_stream)
test_data = nlp.data.PrefetchingStream(test_data)

###############################################################################
# Build the model
###############################################################################

model = nlp.model.language_model.train.BigRNN(ntokens, args.emsize, args.nhid,
                                              args.nlayers, args.nproj, args.k,
                                              embed_dropout=args.dropout,
                                              encode_dropout=args.dropout)
eval_model = nlp.model.language_model.BigRNN(ntokens, args.emsize, args.nhid,
                                             args.nlayers, args.nproj,
                                             embed_dropout=args.dropout,
                                             encode_dropout=args.dropout)

loss = gluon.loss.SoftmaxCrossEntropyLoss()
model.initialize(mx.init.Xavier(factor_type='out'), ctx=context)
trainer_params = {'learning_rate': args.lr, 'wd': 0, 'eps': args.eps}
trainer = gluon.Trainer(model.collect_params(), 'adagrad', trainer_params)
if args.from_epoch:
    from_epoch = args.from_epoch
    checkpoint_name = '%s.%s'%(args.save, format(from_epoch - 1, '02d'))
    model.load_parameters(checkpoint_name)
    trainer.load_states('%s.state'%args.save)
    print('Loaded parameters from checkpoint %s'%(checkpoint_name))


model.hybridize(static_alloc=True, static_shape=True)

train_metric = mx.metric.Loss(loss)
val_metric = LengthNormalizedLoss(loss)
batch_processor = ParallelLanguageModelBatchProcessor(loss=loss,
                                                      vocab=vocab,
                                                      batch_size=args.batch_size,
                                                      val_batch_size=args.batch_size)
lm_estimator = LanguageModelEstimator(net=model, loss=loss,
                                      train_metrics=train_metric,
                                      val_metrics=val_metric,
                                      trainer=trainer,
                                      context=context,
                                      val_loss=loss,
                                      val_net=eval_model,
                                      batch_processor=batch_processor,
                                      bptt=args.bptt)

hidden_state_handler = HiddenStateHandler()
gradient_handler = LargeRNNGradientUpdateHandler(batch_size=args.batch_size, clip=args.clip)
metric_handler = MetricResetHandler(metrics=lm_estimator.train_metrics,
                                    log_interval=args.log_interval)
checkpoint_handler = CheckpointHandler(model_dir=args.save, model_prefix='largeRNN')
logging_handler = ParallelLoggingHandler(log_interval=args.log_interval,
                                         metrics=lm_estimator.train_metrics)
val_logging_handler = LoggingHandler(log_interval=args.log_interval,
                                     metrics=lm_estimator.val_metrics)

event_handlers = [hidden_state_handler, gradient_handler,
                  metric_handler, checkpoint_handler, logging_handler]

if not args.eval_only:
    lm_estimator.fit(train_data=train_data,
                     epochs=args.epochs,
                     event_handlers=event_handlers,
                     #batches=5,
                     batch_axis=0)

val_metric_handler = MetricResetHandler(metrics=lm_estimator.val_metrics)
lm_estimator.val_net.initialize(mx.init.Xavier(), ctx=context[0])
lm_estimator.val_net.hybridize(static_alloc=True, static_shape=True)

for epoch_id in range(args.epochs):
    for filename in os.listdir(args.save):
        file_pattern = 'largeRNN-epoch%dbatch\d+.params' % (epoch_id)
        if re.match(file_pattern + '',filename):
            checkpoint_path = args.save + '/' + filename
            lm_estimator.val_net.load_parameters(checkpoint_path)
            lm_estimator.evaluate(val_data=test_data, event_handlers=[val_metric_handler, val_logging_handler])
