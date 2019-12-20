"""
Fine-tune Language Model for Sentiment Analysis
===============================================
This example shows how to load a language model pre-trained on wikitext-2 in Gluon NLP Toolkit model
zoo, and reuse the language model encoder for sentiment analysis on IMDB movie reviews dataset.
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


import argparse
import time
import random
import multiprocessing as mp
import numpy as np
import warnings

import mxnet as mx
from mxnet import gluon, autograd, metric
from mxnet.gluon import HybridBlock
from mxnet.gluon.data import DataLoader
from mxnet.gluon.contrib.estimator import Estimator
from mxnet.gluon.contrib.estimator.event_handler import EarlyStoppingHandler, CheckpointHandler
from mxnet.gluon.contrib.estimator.batch_processor import BatchProcessor
import gluonnlp as nlp

nlp.utils.check_version('0.7.0')

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)

tokenizer = nlp.data.SpacyTokenizer('en')
length_clip = nlp.data.ClipSequence(500)

parser = argparse.ArgumentParser(
    description='MXNet Sentiment Analysis Example on IMDB. '
                'We load a LSTM model that is pre-trained on '
                'WikiText as our encoder.')
parser.add_argument(
    '--lm_model',
    type=str,
    default='standard_lstm_lm_200',
    help='type of the pre-trained model to load, can be "standard_lstm_200", '
         '"standard_lstm_200", etc.')
parser.add_argument(
    '--use-mean-pool',
    type=bool,
    default=True,
    help='whether to use mean pooling to aggregate the states from '
         'different timestamps.')
parser.add_argument('--no_pretrained',
                    action='store_true',
                    help='Turn on the option to just use the structure and '
                         'not load the pre-trained weights.')
parser.add_argument('--lr',
                    type=float,
                    default=2.5E-3,
                    help='initial learning rate')
parser.add_argument('--clip',
                    type=float,
                    default=None,
                    help='gradient clipping')
parser.add_argument('--bucket_type',
                    type=str,
                    default=None,
                    help='Can be "fixed" or "sorted"')
parser.add_argument('--bucket_num',
                    type=int,
                    default=10,
                    help='The bucket_num if bucket_type is "fixed".')
parser.add_argument('--bucket_ratio',
                    type=float,
                    default=0.0,
                    help='The ratio used in the FixedBucketSampler.')
parser.add_argument('--bucket_mult',
                    type=int,
                    default=100,
                    help='The mult used in the SortedBucketSampler.')
parser.add_argument(
    '--valid_ratio',
    type=float,
    default=0.05,
    help='Proportion [0, 1] of training samples to use for validation set.')
parser.add_argument('--epochs', type=int, default=3, help='upper epoch limit')
parser.add_argument('--batch_size',
                    type=int,
                    default=16,
                    metavar='N',
                    help='batch size')
parser.add_argument('--dropout',
                    type=float,
                    default=0.,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--log-interval',
                    type=int,
                    default=30,
                    metavar='N',
                    help='report interval')
parser.add_argument('--save-prefix',
                    type=str,
                    default='sa-model',
                    help='path to save the final model')
parser.add_argument(
    '--gpu',
    type=int,
    default=None,
    help='id of the gpu to use. Set it to empty means to use cpu.')
args = parser.parse_args()
print(args)

pretrained = not args.no_pretrained
if args.gpu is None:
    print('Use cpu')
    context = mx.cpu()
else:
    print('Use gpu%d' % args.gpu)
    context = mx.gpu(int(args.gpu))


class AggregationLayer(HybridBlock):
    """A block for different ways of aggregating encoder features"""

    def __init__(self, use_mean_pool=False, prefix=None, params=None):
        super(AggregationLayer, self).__init__(prefix=prefix, params=params)
        self._use_mean_pool = use_mean_pool

    def hybrid_forward(self, F, data, valid_length):  # pylint: disable=arguments-differ
        """Forward logic"""
        # Data will have shape (T, N, C)
        if self._use_mean_pool:
            masked_encoded = F.SequenceMask(data,
                                            sequence_length=valid_length,
                                            use_sequence_length=True)
            agg_state = F.broadcast_div(F.sum(masked_encoded, axis=0),
                                        F.expand_dims(valid_length, axis=1))
        else:
            agg_state = F.SequenceLast(data,
                                       sequence_length=valid_length,
                                       use_sequence_length=True)
        return agg_state


class SentimentNet(HybridBlock):
    """Network for sentiment analysis."""

    def __init__(self, dropout, use_mean_pool=False, prefix=None, params=None):
        super(SentimentNet, self).__init__(prefix=prefix, params=params)
        self._use_mean_pool = use_mean_pool
        with self.name_scope():
            self.embedding = None
            self.encoder = None
            self.agg_layer = AggregationLayer(use_mean_pool=use_mean_pool)
            self.output = gluon.nn.HybridSequential()
            with self.output.name_scope():
                self.output.add(gluon.nn.Dropout(dropout))
                self.output.add(gluon.nn.Dense(1, flatten=False))

    def hybrid_forward(self, _, data, valid_length):  # pylint: disable=arguments-differ
        encoded = self.encoder(self.embedding(data))  # Shape(T, N, C)
        agg_state = self.agg_layer(encoded, valid_length)
        out = self.output(agg_state)
        return out


net = SentimentNet(dropout=args.dropout, use_mean_pool=args.use_mean_pool)
with net.name_scope():
    lm_model, vocab = nlp.model.get_model(name=args.lm_model,
                                          dataset_name='wikitext-2',
                                          pretrained=pretrained,
                                          ctx=context,
                                          dropout=args.dropout)

net.embedding = lm_model.embedding
net.encoder = lm_model.encoder
net.hybridize()


# Dataset preprocessing
def preprocess(x):
    data, label = x
    label = int(label > 5)
    data = vocab[length_clip(tokenizer(data))]
    return data, label


def get_length(x):
    return float(len(x[0]))


# Load the dataset
train_dataset, test_dataset = [
    nlp.data.IMDB(root='data/imdb', segment=segment)
    for segment in ('train', 'test')
]
train_dataset, valid_dataset = nlp.data.train_valid_split(
    train_dataset, args.valid_ratio)
print('Tokenize using spaCy...')


def preprocess_dataset(dataset):
    start = time.time()
    pool = mp.Pool(8)
    dataset = gluon.data.SimpleDataset(pool.map(preprocess, dataset))
    lengths = gluon.data.SimpleDataset(pool.map(get_length, dataset))
    end = time.time()
    print('Done! Tokenizing Time={:.2f}s, #Sentences={}'.format(
        end - start, len(dataset)))
    return dataset, lengths


# Preprocess the dataset
train_dataset, train_data_lengths = preprocess_dataset(train_dataset)
valid_dataset, valid_data_lengths = preprocess_dataset(valid_dataset)
test_dataset, test_data_lengths = preprocess_dataset(test_dataset)

# Construct the DataLoader. Pad data and stack label
batchify_fn = nlp.data.batchify.Tuple(
    nlp.data.batchify.Pad(axis=0, pad_val=0, ret_length=True),
    nlp.data.batchify.Stack(dtype='float32'))
if args.bucket_type is None:
    print('Bucketing strategy is not used!')
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  batchify_fn=batchify_fn)
else:
    if args.bucket_type == 'fixed':
        print('Use FixedBucketSampler')
        batch_sampler = nlp.data.FixedBucketSampler(
            train_data_lengths,
            batch_size=args.batch_size,
            num_buckets=args.bucket_num,
            ratio=args.bucket_ratio,
            shuffle=True)
        print(batch_sampler.stats())
    elif args.bucket_type == 'sorted':
        print('Use SortedBucketSampler')
        batch_sampler = nlp.data.SortedBucketSampler(
            train_data_lengths,
            batch_size=args.batch_size,
            mult=args.bucket_mult,
            shuffle=True)
    else:
        raise NotImplementedError
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_sampler=batch_sampler,
                                  batchify_fn=batchify_fn)

valid_dataloader = DataLoader(
    dataset=valid_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    sampler=nlp.data.SortedSampler(valid_data_lengths),
    batchify_fn=batchify_fn)

test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             sampler=nlp.data.SortedSampler(test_data_lengths),
                             batchify_fn=batchify_fn)

net.hybridize()
print(net)
if args.no_pretrained:
    net.initialize(mx.init.Xavier(), ctx=context)
else:
    net.output.initialize(mx.init.Xavier(), ctx=context)
trainer = gluon.Trainer(net.collect_params(), 'ftml',
                        {'learning_rate': args.lr})
loss_fn = gluon.loss.SigmoidBCELoss()


class SentimentAnalysisBatchProcessor(BatchProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate_batch(self, estimator, val_batch, batch_axis=0):
        data = mx.nd.transpose(val_batch[0][0])
        valid_length = val_batch[0][1].astype(np.float32)
        label = val_batch[1]
        data_list = gluon.utils.split_and_load(data, estimator.context)
        valid_length_list = gluon.utils.split_and_load(valid_length,
                                                       estimator.context)
        label_list = gluon.utils.split_and_load(label, estimator.context)
        pred = [
            estimator.eval_net(data, valid_length)
            for data, valid_length in zip(data_list, valid_length_list)
        ]
        loss = [
            estimator.evaluation_loss(y_hat, y)
            for y_hat, y in zip(pred, label_list)
        ]
        pre_labels = [(pred_pro > 0.5).reshape(-1, ) for pred_pro in pred]
        return data_list, label_list, pre_labels, loss

    def fit_batch(self, estimator, train_batch, batch_axis=0):
        data = mx.nd.transpose(train_batch[0][0])
        valid_length = train_batch[0][1].astype(np.float32)
        label = train_batch[1]
        data_list = gluon.utils.split_and_load(data, estimator.context)
        valid_length_list = gluon.utils.split_and_load(valid_length,
                                                       estimator.context)
        label_list = gluon.utils.split_and_load(label, estimator.context)
        with autograd.record():
            pred = [
                estimator.net(data, valid_length)
                for data, valid_length in zip(data_list, valid_length_list)
            ]
            loss = [
                estimator.loss(y_hat, y).mean()
                for y_hat, y in zip(pred, label_list)
            ]
        for l in loss:
            l.backward()
        # Clip gradient
        if args.clip is not None:
            grads = [
                p.grad(estimator.context)
                for p in self.net.collect_params().values()
            ]
            gluon.utils.clip_global_norm(grads, args.clip)
        pre_labels = [(pred_pro > 0.5).reshape(-1, ) for pred_pro in pred]
        return data_list, label_list, pre_labels, loss


# Metric to monitor
acc_metric = metric.Accuracy()

earlystopping_handler = EarlyStoppingHandler(monitor=acc_metric,
                                             min_delta=0,
                                             patience=3,
                                             mode='max')

checkpoint_handler = CheckpointHandler(
    model_dir='./',
    model_prefix=args.save_prefix,
    monitor=acc_metric,  # Monitors a metric
    save_best=True,
    mode='max')  # Save the best model in terms of

est = Estimator(net=net,
                loss=loss_fn,
                train_metrics=[mx.metric.Loss()],
                val_metrics=[mx.metric.Accuracy()],
                trainer=trainer,
                context=context,
                batch_processor=SentimentAnalysisBatchProcessor())

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Magic line
    est.fit(train_data=train_dataloader,
            val_data=valid_dataloader,
            epochs=args.epochs,
            event_handlers=[checkpoint_handler,
                            earlystopping_handler])  # Add the event handlers
