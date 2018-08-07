"""
Fine-tune Language Model for Sentiment Analysis
===============================================

This example shows how to use Convolutional neural network in Gluon NLP Toolkit model
zoo, and use the model encoder for sentiment analysis on various datasets.
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
import random
import glob
import multiprocessing as mp
import numpy as np

import mxnet as mx
from mxnet import nd, gluon, autograd
from mxnet.gluon import HybridBlock
from mxnet.gluon.data import DataLoader

import gluonnlp as nlp

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)

parser = argparse.ArgumentParser(description='MXNet Sentiment Analysis Example on various datasets. '
                                             'We load textCNN as our model.')
parser.add_argument('--data_name', choices=['MR', 'SST-1', 'SST-2', 'Subj', 'TREC'], default='MR',
                    help='specified data set')
parser.add_argument('--model_mode', choices=['rand', 'static', 'non-static', 'multichannel'],
                    default='multichannel', help='the used model')
parser.add_argument('--lr', type=float, default=2.5E-3,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--log-interval', type=int, default=30, metavar='N',
                    help='report interval')
parser.add_argument('--save-prefix', type=str, default='sa-model',
                    help='path to save the final model')
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


class SentimentNet(HybridBlock):
    """Network for sentiment analysis."""
    def __init__(self, dropout, embed_size=300, vocab_size=100, prefix=None,
                 params=None, num_filters=(100, 100, 100), ngram_filter_sizes=(3, 4, 5)):
        super(SentimentNet, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.embedding = gluon.nn.Embedding(vocab_size, embed_size)
            if args.model_mode == 'multichannel':
                self.embedding_extend = gluon.nn.Embedding(vocab_size, embed_size)
                embed_size *= 2
            self.encoder = nlp.model.ConvolutionalEncoder(embed_size=embed_size,
                                                          num_filters=num_filters,
                                                          ngram_filter_sizes=ngram_filter_sizes,
                                                          conv_layer_activation='relu',
                                                          num_highway=None)
            self.output = gluon.nn.HybridSequential()
            with self.output.name_scope():
                self.output.add(gluon.nn.Dropout(dropout))
                self.output.add(gluon.nn.Dense(output_size, flatten=False))

    def hybrid_forward(self, F, data): # pylint: disable=arguments-differ
        if args.model_mode == 'multichannel':
            embedded = F.concat(self.embedding(data), self.embedding_extend(data), dim=2)
        else:
            embedded = self.embedding(data)
        encoded = self.encoder(embedded)  # Shape(T, N, C)
        out = self.output(encoded)
        return out

if args.data_name == 'MR':
    train_dataset, test_dataset = [nlp.data.MR(root='data/mr', segment=segment)
                                   for segment in ('train', 'test')]
    output_size = 2
elif args.data_name == 'SST-1':
    train_dataset, test_dataset = [nlp.data.SST_1(root='data/sst-1', segment=segment)
                                   for segment in ('train', 'test')]
    output_size = 5
elif args.data_name == 'SST-2':
    train_dataset, test_dataset = [nlp.data.SST_2(root='data/sst-2', segment=segment)
                                   for segment in ('train', 'test')]
    output_size = 2
elif args.data_name == 'Subj':
    train_dataset, test_dataset = [nlp.data.SUBJ(root='data/Subj', segment=segment)
                                   for segment in ('train', 'test')]
    output_size = 2
elif args.data_name == 'TREC':
    train_dataset, test_dataset = [nlp.data.TREC(root='data/trec', segment=segment)
                                   for segment in ('train', 'test')]
    output_size = 6

all_token = []
max_len = 0 
for line in train_dataset: 
    line = line[0].split(' ') 
    max_len = max_len if max_len > len(line) else len(line) 
    all_token.extend(line) 
vocab = nlp.Vocab(nlp.data.count_tokens(all_token))
vocab.set_embedding(nlp.embedding.create('Word2Vec', source='GoogleNews-vectors-negative300'))
net = SentimentNet(dropout=args.dropout, vocab_size=len(vocab))

# Dataset preprocessing
def preprocess(x):
    data, label = x
    data = vocab[data.split(' ')]
    if len(data) > max_len:
        data = data[:max_len]
    else:
        while len(data) < max_len:
            data.append(0)
    return data, label

def get_length(x):
    return float(len(x[0]))

def preprocess_dataset(dataset):
    start = time.time()
    pool = mp.Pool(8)
    dataset = gluon.data.SimpleDataset(pool.map(preprocess, dataset))
    lengths = gluon.data.SimpleDataset(pool.map(get_length, dataset))
    end = time.time()
    print('Done! Tokenizing Time={:.2f}s, #Sentences={}'.format(end - start, len(dataset)))
    return dataset, lengths

# Preprocess the dataset
train_dataset, train_data_lengths = preprocess_dataset(train_dataset)
test_dataset, test_data_lengths = preprocess_dataset(test_dataset)

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)


net.hybridize()
print(net)

net.initialize(mx.init.Xavier(), ctx=context)
if args.model_mode != 'rand':
    net.embedding.weight.set_data(vocab.embedding.idx_to_vec)
if args.model_mode == 'multichannel':
    net.embedding_extend.weight.set_data(vocab.embedding.idx_to_vec)
if args.model_mode == 'static' or args.model_mode == 'multichannel':
    net.embedding.collect_params().setattr('grad_req', 'null')
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': args.lr, 'wd':0.004})
loss = gluon.loss.SoftmaxCrossEntropyLoss()


def evaluate(dataloader):
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


def train():
    """Training process"""
    start_pipeline_time = time.time()

    # Training/Testing
    best_valid_acc = 0
    stop_early = 0
    for epoch in range(args.epochs):
        # Epoch training stats
        start_epoch_time = time.time()
        epoch_L = 0.0
        epoch_sent_num = 0
        epoch_wc = 0
        # Log interval training stats
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
            # Update parameter
            trainer.step(1)
            log_interval_L += L.asscalar()
            epoch_L += L.asscalar()
            if (i + 1) % args.log_interval == 0:
                print('[Epoch %d Batch %d/%d] avg loss %g, throughput %gK wps' % (
                    epoch, i + 1, len(train_dataloader),
                    log_interval_L / log_interval_sent_num,
                    log_interval_wc / 1000 / (time.time() - start_log_interval_time)))
                # Clear log interval training stats
                start_log_interval_time = time.time()
                log_interval_wc = 0
                log_interval_sent_num = 0
                log_interval_L = 0
        end_epoch_time = time.time()
        test_avg_L, test_acc = evaluate(test_dataloader)
        print('[Epoch %d] train avg loss %g, '
              'test acc %.4f, test avg loss %g, throughput %gK wps' % (
                  epoch, epoch_L / epoch_sent_num,
                  test_acc, test_avg_L,
                  epoch_wc / 1000 / (end_epoch_time - start_epoch_time)))

        if test_acc < best_valid_acc:
            print('No Improvement.')
            stop_early += 1
            if stop_early == 30:
                break
        else:
            # Reset stop_early if the validation loss finds a new low value
            print('Observed Improvement.')
            stop_early = 0
            net.save_params(args.save_prefix + '_{:04d}.params'.format(epoch))
            best_valid_acc = test_acc

    net.load_params(glob.glob(args.save_prefix+'_*.params')[-1], context)
    test_avg_L, test_acc = evaluate(test_dataloader)
    print('Best test loss %g, test acc %.4f'%(test_avg_L, test_acc))
    print('Total time cost %.2fs'%(time.time()-start_pipeline_time))


if __name__ == '__main__':
    train()
