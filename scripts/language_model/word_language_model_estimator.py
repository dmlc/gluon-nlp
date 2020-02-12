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
from mxnet.gluon.contrib.estimator import LoggingHandler
from gluonnlp.estimator import JointActivationRegularizationLoss
from gluonnlp.estimator import LanguageModelEstimator
from gluonnlp.estimator import HiddenStateHandler, AvgParamHandler
from gluonnlp.estimator import LearningRateHandler, RNNGradientUpdateHandler
from gluonnlp.estimator import WordLanguageModelCheckpointHandler
from gluonnlp.estimator import LanguageModelBatchProcessor
from gluonnlp.estimator import MetricResetHandler
from mxnet.gluon.data.sampler import BatchSampler

class BatchVariableLenTextSampler(BatchSampler):
    def __init__(self, bptt, length, use_variable_length=True):
        self.bptt = bptt
        self.length = length
        self.index = 0
        self.use_variable_length = use_variable_length

    def __iter__(self):
        self.index = 0
        while self.index < self.length - 2:
            if self.use_variable_length:
                bptt = self.bptt if mx.nd.random.uniform().asscalar() < .95 else self.bptt / 2
                seq_len = max(5, int(mx.nd.random.normal(bptt, 5).asscalar()))
            else:
                seq_len = self.bptt
            seq_len = min(seq_len, self.length - self.index - 1)
            # batch_size = seq_len + 1
            batch = []
            for i in range(self.index, self.index + seq_len + 1):
                batch.append(i)
            self.index += seq_len
            yield batch

    def __len__(self):
        # you may never get real size of the data sampler beforehand. May need some
        # postprocessing after fetching the data batch
        return int(self.length / 5) + 1

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, '..', '..'))

nlp.utils.check_version('0.7.0')

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
parser.add_argument('--epochs', type=int, default=750,
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
parser.add_argument('--gpu', type=str, help='single gpu id')
parser.add_argument('--optimizer', type=str, default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--wd', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation '
                         '(alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activation '
                         '(beta = 0 means no regularization)')
parser.add_argument('--ntasgd', action='store_true',
                    help='Whether to apply ntasgd')
parser.add_argument('--test_mode', action='store_true',
                    help='Whether to run through the script with few examples')
parser.add_argument('--lr_update_interval', type=int, default=30,
                    help='lr udpate interval')
parser.add_argument('--lr_update_factor', type=float, default=0.1,
                    help='lr udpate factor')
args = parser.parse_args()

###############################################################################
# Load data
###############################################################################

context = [mx.cpu()] if not args.gpu else [mx.gpu(int(args.gpu))]

assert args.batch_size % len(context) == 0, \
    'Total batch size must be multiple of the number of devices'

assert args.weight_dropout > 0 or (args.weight_dropout == 0 and args.alpha == 0), \
    'The alpha L2 regularization cannot be used with standard RNN, please set alpha to 0'

train_dataset, val_dataset, test_dataset = \
    [nlp.data.WikiText2(segment=segment,
                        skip_empty=False, bos=None, eos='<eos>')
     for segment in ['train', 'val', 'test']]

vocab = nlp.Vocab(counter=nlp.data.Counter(train_dataset), padding_token=None, bos_token=None)
train_batchify = nlp.data.batchify.CorpusBatchify(vocab, args.batch_size)
train_data = train_batchify(train_dataset)
val_batch_size = 10
val_batchify = nlp.data.batchify.CorpusBatchify(vocab, val_batch_size)
val_data = val_batchify(val_dataset)
test_batch_size = 1
test_batchify = nlp.data.batchify.CorpusBatchify(vocab, test_batch_size)
test_data = test_batchify(test_dataset)

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
    model = nlp.model.train.AWDRNN(args.model, len(vocab), args.emsize, args.nhid, args.nlayers,
                                   args.tied, args.dropout, args.weight_dropout,
                                   args.dropout_h, args.dropout_i, args.dropout_e)
    model.initialize(mx.init.Xavier(), ctx=context)
    model_eval = nlp.model.AWDRNN(args.model, len(vocab), args.emsize, args.nhid, args.nlayers,
                                  args.tied, args.dropout, args.weight_dropout,
                                  args.dropout_h, args.dropout_i, args.dropout_e,
                                  params=model.collect_params())
else:
    model = nlp.model.train.StandardRNN(args.model, len(vocab), args.emsize,
                                        args.nhid, args.nlayers, args.dropout, args.tied)
    model.initialize(mx.init.Xavier(), ctx=context)
    model_eval = nlp.model.StandardRNN(args.model, len(vocab), args.emsize,
                                       args.nhid, args.nlayers, args.dropout, args.tied,
                                       params=model.collect_params())


model.hybridize(static_alloc=True)

print(model)


def check_initialized(net):
    params = net.collect_params()
    for param in params:
        try:
            params[param].list_ctx()
        except RuntimeError:
            return False
    return True
    
print(check_initialized(model))
print(check_initialized(model_eval))
                                    
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

trainer = gluon.Trainer(model.collect_params(), args.optimizer, trainer_params,
                        update_on_kvstore=False)

loss = gluon.loss.SoftmaxCrossEntropyLoss()
train_loss = JointActivationRegularizationLoss(loss, args.alpha, args.beta)

sampler = BatchVariableLenTextSampler(bptt=70, length=len(train_data))
val_sampler = BatchVariableLenTextSampler(bptt=70, length=len(val_data), use_variable_length=False)
test_sampler = BatchVariableLenTextSampler(bptt=70, length=len(test_data),
                                           use_variable_length=False)
train_data_loader = mx.gluon.data.DataLoader(train_data,
                                             batch_sampler=sampler)
val_data_loader = mx.gluon.data.DataLoader(val_data,
                                           batch_sampler=val_sampler)
test_data_loader = mx.gluon.data.DataLoader(test_data,
                                            batch_sampler=test_sampler)

train_metric = mx.metric.Loss(train_loss)
val_metric = mx.metric.Loss(loss)
batch_processor = LanguageModelBatchProcessor()
est = LanguageModelEstimator(net=model, loss=train_loss,
                             train_metrics=train_metric,
                             val_metrics=val_metric,
                             trainer=trainer, context=context,
                             val_loss=loss,
                             val_net=model_eval,
                             batch_processor=batch_processor)
event_handlers = [HiddenStateHandler(), AvgParamHandler(data_length=len(train_data)),
                  LearningRateHandler(lr_update_interval=args.lr_update_interval, lr_update_factor=args.lr_update_factor),
                  RNNGradientUpdateHandler(clip=args.clip),
                  LoggingHandler(log_interval=args.log_interval, metrics=est.train_metrics + est.val_metrics),
                  MetricResetHandler(metrics=est.train_metrics, log_interval=args.log_interval),
                  WordLanguageModelCheckpointHandler(args.save)]
est.fit(train_data=train_data_loader, val_data=val_data_loader,
        epochs=args.epochs,
        event_handlers=event_handlers,
        batch_axis=1)

est.net.load_parameters(args.save)
est.evaluate(val_data=val_data_loader, event_handlers=[HiddenStateHandler()], batch_axis=1)
est.evaluate(val_data=test_data_loader, event_handlers=[HiddenStateHandler()], batch_axis=1)
