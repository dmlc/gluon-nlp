"""
TextCNN Model for Sentiment Analysis
===============================================
This example shows how to use convolutional neural networks (textCNN)
for sentiment analysis on various datasets.
Kim, Y. (2014). Convolutional neural networks for sentence classification.
arXiv preprint arXiv:1408.5882.
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
import random
import numpy as np

import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon.data import DataLoader
from mxnet.gluon.contrib.estimator import Estimator
from mxnet.gluon.contrib.estimator.event_handler import EpochEnd
from mxnet.gluon.contrib.estimator.batch_processor import BatchProcessor
import gluonnlp
import process_data
import text_cnn

gluonnlp.utils.check_version('0.7.0')

np.random.seed(3435)
random.seed(3435)
mx.random.seed(3435)

parser = argparse.ArgumentParser(
    description='Sentiment analysis with the textCNN model on\
                                 various datasets.')
parser.add_argument(
    '--data_name',
    choices=['MR', 'SST-1', 'SST-2', 'Subj', 'TREC', 'CR', 'MPQA'],
    default='MR',
    help='name of the data set')
parser.add_argument('--model_mode',
                    choices=['rand', 'static', 'non-static', 'multichannel'],
                    default='multichannel',
                    help='Variants of the textCNN model (see the paper:\
                    Convolutional Neural Networks for Sentence Classification).'
                    )
parser.add_argument('--epochs',
                    type=int,
                    default=200,
                    help='upper epoch limit')
parser.add_argument('--batch_size',
                    type=int,
                    default=50,
                    metavar='N',
                    help='batch size')
parser.add_argument('--dropout',
                    type=float,
                    default=.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--log-interval',
                    type=int,
                    default=30,
                    metavar='N',
                    help='report interval')
parser.add_argument(
    '--gpu',
    type=int,
    default=None,
    help='id of the gpu to use. Set it to empty means to use cpu.')
args = parser.parse_args()
print(args)

if args.gpu is None:
    print('Use cpu')
    context = mx.cpu()
else:
    print('Use gpu%d' % args.gpu)
    context = mx.gpu(args.gpu)

if args.data_name in ('MR', 'Subj', 'CR', 'MPQA'):
    vocab, max_len, output_size, train_dataset, train_data_lengths = \
			process_data.load_dataset(args.data_name)
elif args.data_name == 'TREC':
    vocab, max_len, output_size, train_dataset, train_data_lengths, test_dataset, test_data_lengths = \
			process_data.load_dataset(args.data_name)
else:
    vocab, max_len, output_size, train_dataset, train_data_lengths, test_dataset, test_data_lengths, dev_dataset, \
			dev_data_lengths = process_data.load_dataset(args.data_name)

model = text_cnn.model(args.dropout, vocab, args.model_mode, output_size)
print(model)


def check_metrics(metrics):
    """check metrics"""
    if isinstance(metrics, mx.metric.CompositeEvalMetric):
        metrics = [
            m for metric in metrics.metrics for m in check_metrics(metric)
        ]
    elif isinstance(metrics, mx.metric.EvalMetric):
        metrics = [metrics]
    else:
        metrics = metrics or []
        if not all(
                [isinstance(metric, mx.metric.EvalMetric) for metric in metrics]):
            raise ValueError(
                'metrics must be a Metric or a list of Metric, \
		refer to mxnet.metric.EvalMetric: {}'.format(metrics))
    return metrics


class GetValMetricHandler(EpochEnd):
    """ track validation metric at the end of every epoch."""
    def __init__(self, metrics):
        self.metrics = check_metrics(metrics)
        self.metric_history = {}

    def epoch_end(self, estimator, *argss, **kwargs):
        for metric in self.metrics:
            metric_name, metric_val = metric.get()
            if 'validation' in metric_name:
                self.metric_history.setdefault(metric_name,
                                               []).append(metric_val)


class SentimentAnalysisCNNBatchProcessor(BatchProcessor):
    """subclass for SentimentAnalysisCNN"""
    def __init__(self):
        super().__init__()

    def evaluate_batch(self, estimator, val_batch, batch_axis=0):
        data = mx.nd.transpose(val_batch[0])
        label = val_batch[1]
        data_list = gluon.utils.split_and_load(data, estimator.context)
        label_list = gluon.utils.split_and_load(label, estimator.context)
        pred = [estimator.eval_net(data) for data in data_list]
        loss = [
            estimator.evaluation_loss(y_hat, y)
            for y_hat, y in zip(pred, label_list)
        ]
        pre_labels = [nd.argmax(pred_pro, axis=1) for pred_pro in pred]
        return data_list, label_list, pre_labels, loss

    def fit_batch(self, estimator, train_batch, batch_axis=0):
        data = mx.nd.transpose(train_batch[0])
        label = train_batch[1]
        data_list = gluon.utils.split_and_load(data, estimator.context)
        label_list = gluon.utils.split_and_load(label, estimator.context)
        with autograd.record():
            pred = [estimator.net(data) for data in data_list]
            loss = [
                estimator.loss(y_hat, y).mean()
                for y_hat, y in zip(pred, label_list)
            ]
        for l in loss:
            l.backward()
        pre_labels = [nd.argmax(pred_pro, axis=1) for pred_pro in pred]
        return data_list, label_list, pre_labels, loss


def k_fold_cross_valid(k, all_dataset):
    """k fold cross valid"""
    val_acc = []
    fold_size = len(all_dataset) // k
    random.shuffle(all_dataset)

    for i in range(k):
        print('Fold-%d starts...' % (i + 1))
        _val_dataset = all_dataset[i * fold_size:(i + 1) * fold_size]
        _train_dataset = all_dataset[:i * fold_size] + all_dataset[(i + 1) *
                                                                  fold_size:]
        _train_dataloader = DataLoader(dataset=_train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True)
        _val_dataloader = DataLoader(dataset=_val_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False)
        _est = Estimator(net=net,
                        loss=loss_fn,
                        train_metrics=[mx.metric.Loss()],
                        trainer=trainer,
                        context=context,
                        evaluation_loss=loss_fn,
                        val_metrics=[mx.metric.Accuracy()],
                        batch_processor=SentimentAnalysisCNNBatchProcessor())
        get_val_metric_handler = GetValMetricHandler(_est.val_metrics)
        _est.fit(train_data=_train_dataloader,
                val_data=_val_dataloader,
                epochs=args.epochs,
                event_handlers=[get_val_metric_handler
                                ])  # Add the event handlers
        for metric_name, metric_vals in get_val_metric_handler.metric_history.items(
        ):
            if metric_name == 'validation accuracy':
                val_acc.append(max(metric_vals))
    print('K-fold(%d-fold) cross valid avg acc' % (k), sum(val_acc) / k)


net, trainer = text_cnn.init(model, vocab, args.model_mode, context)
loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
est = Estimator(net=net,
                loss=loss_fn,
                train_metrics=[mx.metric.Loss()],
                trainer=trainer,
                context=context,
                evaluation_loss=loss_fn,
                val_metrics=[mx.metric.Accuracy()],
                batch_processor=SentimentAnalysisCNNBatchProcessor())
if __name__ == '__main__':
    if args.data_name == 'TREC':
        random.shuffle(train_dataset)
        sp = len(train_dataset) // 10
        train_dataloader = DataLoader(dataset=train_dataset[sp:],
                                      batch_size=args.batch_size,
                                      shuffle=True)
        val_dataloader = DataLoader(dataset=train_dataset[:sp],
                                    batch_size=args.batch_size,
                                    shuffle=False)
        est.fit(train_data=train_dataloader,
                val_data=val_dataloader,
                epochs=args.epochs)  # Add the event handlers
    elif args.data_name == 'SST-1' or args.data_name == 'SST-2':
        train_dataloader = DataLoader(dataset=train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True)
        val_dataloader = DataLoader(dataset=dev_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False)
        est.fit(train_data=train_dataloader,
                val_data=val_dataloader,
                epochs=args.epochs)  # Add the event handlers
    else:
        k_fold_cross_valid(10, train_dataset)
