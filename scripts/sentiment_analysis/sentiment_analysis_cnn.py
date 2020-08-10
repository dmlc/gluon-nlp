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
import time
import random

import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon.data import DataLoader
import gluonnlp
import process_data
import text_cnn

gluonnlp.utils.check_version('0.7.0')

seed = 3435
gluonnlp.utils.set_seed(seed)

parser = argparse.ArgumentParser(description='Sentiment analysis with the textCNN model on\
                                 various datasets.')
parser.add_argument('--data_name', choices=['MR', 'SST-1', 'SST-2', 'Subj', 'TREC', 'CR', 'MPQA'],
                    default='MR', help='name of the data set')
parser.add_argument('--model_mode', choices=['rand', 'static', 'non-static', 'multichannel'],
                    default='multichannel', help='Variants of the textCNN model (see the paper:\
                    Convolutional Neural Networks for Sentence Classification).')
parser.add_argument('--epochs', type=int, default=200,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--log-interval', type=int, default=30, metavar='N',
                    help='report interval')
parser.add_argument('--gpu', type=int, default=None,
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
    vocab, max_len, output_size, train_dataset, train_data_lengths \
    = process_data.load_dataset(args.data_name, args.model_mode)
elif args.data_name == 'TREC':
    vocab, max_len, output_size, train_dataset, train_data_lengths, \
    test_dataset, test_data_lengths = process_data.load_dataset(args.data_name, args.model_mode)
else:
    vocab, max_len, output_size, train_dataset, train_data_lengths, test_dataset, \
    test_data_lengths, dev_dataset, dev_data_lengths = process_data.load_dataset(args.data_name,
                                                                                 args.model_mode)

model = text_cnn.model(args.dropout, vocab, args.model_mode, output_size)
print(model)

loss = gluon.loss.SoftmaxCrossEntropyLoss()

def evaluate(net, dataloader):
    """Evaluate network on the specified dataset"""
    total_L = 0.0
    total_sample_num = 0
    total_correct_num = 0
    start_log_interval_time = time.time()
    print('Begin Testing...')
    for i, (data, label) in enumerate(dataloader):
        data = mx.nd.transpose(data.as_in_context(context))
        label = label.as_in_context(context)
        output = net(data)
        L = loss(output, label)
        pred = nd.argmax(output, axis=1)
        total_L += L.sum().asscalar()
        total_sample_num += label.shape[0]
        total_correct_num += (pred.astype('int') == label).sum().asscalar()
        if (i + 1) % args.log_interval == 0:
            print('[Batch {}/{}] elapsed {:.2f} s'.format(
                i + 1, len(dataloader), time.time() - start_log_interval_time))
            start_log_interval_time = time.time()
    avg_L = total_L / float(total_sample_num)
    acc = total_correct_num / float(total_sample_num)
    return avg_L, acc

def train(net, train_data, test_data, dev_data=None):
    """Train textCNN model for sentiment analysis."""
    start_pipeline_time = time.time()
    net, trainer = text_cnn.init(net, vocab, args.model_mode, context)
    if dev_data is None:
        random.shuffle(train_data)
        sp = len(train_data) // 10
        train_dataloader = DataLoader(dataset=train_data[sp:],
                                      batch_size=args.batch_size,
                                      shuffle=True)
        val_dataloader = DataLoader(dataset=train_data[:sp],
                                    batch_size=args.batch_size,
                                    shuffle=False)
    else:
        train_dataloader = DataLoader(dataset=train_data,
                                      batch_size=args.batch_size,
                                      shuffle=True)
        val_dataloader = DataLoader(dataset=dev_data,
                                    batch_size=args.batch_size,
                                    shuffle=False)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=args.batch_size,
                                 shuffle=False)
    # Training/Testing.
    best_val_acc = 0
    for epoch in range(args.epochs):
        # Epoch training stats.
        start_epoch_time = time.time()
        epoch_L = 0.0
        epoch_sent_num = 0
        epoch_wc = 0
        # Log interval training stats.
        start_log_interval_time = time.time()
        log_interval_wc = 0
        log_interval_sent_num = 0
        log_interval_L = 0.0
        for i, (data, label) in enumerate(train_dataloader):
            data = mx.nd.transpose(data.as_in_context(context))
            label = label.as_in_context(context)
            wc = max_len
            log_interval_wc += wc
            epoch_wc += wc
            log_interval_sent_num += data.shape[1]
            epoch_sent_num += data.shape[1]

            with autograd.record():
                output = net(data)
                L = loss(output, label).mean()
            L.backward()
            # Update parameter.
            trainer.step(args.batch_size)
            log_interval_L += L.asscalar()
            epoch_L += L.asscalar()
            if (i + 1) % args.log_interval == 0:
                print('[Epoch %d Batch %d/%d] avg loss %g, throughput %gK wps' % (
                    epoch, i + 1, len(train_dataloader),
                    log_interval_L / log_interval_sent_num,
                    log_interval_wc / 1000 / (time.time() - start_log_interval_time)))
                # Clear log interval training stats.
                start_log_interval_time = time.time()
                log_interval_wc = 0
                log_interval_sent_num = 0
                log_interval_L = 0
        end_epoch_time = time.time()
        val_avg_L, val_acc = evaluate(net, val_dataloader)
        print('[Epoch %d] train avg loss %g, '
              'dev acc %.4f, dev avg loss %g, throughput %gK wps' % (
                  epoch, epoch_L / epoch_sent_num,
                  val_acc, val_avg_L,
                  epoch_wc / 1000 / (end_epoch_time - start_epoch_time)))

        if val_acc >= best_val_acc:
            print('Observed Improvement.')
            best_val_acc = val_acc
            test_avg_L, test_acc = evaluate(net, test_dataloader)

    print('Test loss %g, test acc %.4f'%(test_avg_L, test_acc))
    print('Total time cost %.2fs'%(time.time()-start_pipeline_time))
    return test_acc

def k_fold_cross_valid(k, net, all_dataset):
    test_acc = []
    fold_size = len(all_dataset) // k
    random.shuffle(all_dataset)
    for test_i in range(10):
        test_data = all_dataset[test_i * fold_size: (test_i + 1) * fold_size]
        train_data = all_dataset[: test_i * fold_size] + all_dataset[(test_i + 1) * fold_size:]
        test_acc.append(train(net, train_data, test_data))
    print('K-fold cross valid avg acc', sum(test_acc) / k)

if __name__ == '__main__':
    if args.data_name == 'TREC':
        train(model, train_dataset, test_dataset)
    elif args.data_name == 'SST-1' or args.data_name == 'SST-2':
        train(model, train_dataset, test_dataset, dev_dataset)
    else:
        k_fold_cross_valid(10, model, train_dataset)
