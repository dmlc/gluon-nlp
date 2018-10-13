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

# pylint: disable=global-variable-undefined,wrong-import-position
"""Fasttext Classification Training Model model
===========================
This example shows how to train a FastText Classification model with the
Gluon NLP Toolkit.
The FastText Classification model was introduced by
- Joulin, Armand, et al. "Bag of tricks for efficient text classification."
... arXiv preprint arXiv:1607.01759 (2016).
For larger datasets please refrain from using -ngrams to value > 2
"""
import argparse
import logging
import math

from collections import Counter
import numpy as np
from mxnet import nd, autograd
from mxnet.gluon import nn, HybridBlock
import mxnet as mx
import mxnet.gluon as gluon
import gluonnlp.data.batchify as btf

import gluonnlp
import evaluation


class FastTextClassificationModel(HybridBlock):
    """
    The Model Block for FastTextClassification Model.
    The trained embeddings layer, is averaged and then softmax
    layer is applied on top of it.
    """

    def __init__(self, vocab_size, embedding_dim, num_classes, **kwargs):
        super(FastTextClassificationModel, self).__init__(**kwargs)
        with self.name_scope():
            self.vs = vocab_size
            self.ed = embedding_dim
            self.embedding = nn.Embedding(
                vocab_size,
                embedding_dim,
                weight_initializer=mx.init.Xavier(),
                dtype='float32')
            num_output_units = num_classes
            if num_classes == 2:
                num_output_units = 1
            logging.info('Number of output units in the last layer :%s',
                         num_output_units)
            self.dense = nn.Dense(num_output_units)

    def hybrid_forward(self, F, x):  # pylint: disable=arguments-differ
        embeddings = self.embedding(x)
        dense_output = self.dense(embeddings.mean(axis=1))
        return F.Dropout(dense_output, 0.1)


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


def evaluate_accuracy(data_iterator, net, ctx, loss_fun, num_classes):
    """
    This function is used for evaluating accuracy of
    a given data iterator. (Either Train/Test data)
    It takes in the loss function used too!
    """
    acc = mx.metric.Accuracy()
    loss_avg = 0.
    for i, (data, labels) in enumerate(data_iterator):
        data = data.as_in_context(ctx)  #.reshape((-1,784))
        labels = labels.as_in_context(ctx)
        output = net(data)
        loss = loss_fun(output, labels)
        preds = []
        if (num_classes == 2):
            preds = (nd.sign(output) + 1) / 2
            preds = preds.reshape(-1)
        else:
            preds = nd.argmax(output, axis=1)
        acc.update(preds=preds, labels=labels)
        loss_avg = loss_avg * i / (i + 1) + nd.mean(loss).asscalar() / (i + 1)
    return acc.get()[1], loss_avg


def read_input_data(filename):
    """Helper function to get training data"""
    logging.info('Opening file %s for reading input', filename)
    input_file = open(filename, 'r')
    data = []
    labels = []
    for line in input_file:
        tokens = line.split(',', 1)
        labels.append(tokens[0].strip())
        data.append(tokens[1].strip())
    return labels, data


###############################################################################
# Utils
###############################################################################
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Text Classification with FastText',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Computation options
    group = parser.add_argument_group('Computation arguments')
    group.add_argument('--input', type=str, help='Input file location')
    group.add_argument(
        '--validation', type=str, help='Validation file Location ')
    group.add_argument(
        '--output', type=str, help='Location to save trained model')
    group.add_argument(
        '--ngrams', type=int, default=1, help='NGrams used for training')
    group.add_argument(
        '--batch-size', type=int, default=16, help='Batch size for training.')
    group.add_argument('--epochs', type=int, default=10, help='Epoch limit')
    group.add_argument(
        '--gpu',
        type=int,
        help=('Number (index) of GPU to run on, e.g. 0. '
              'If not specified, uses CPU.'))
    group.add_argument(
        '--no-hybridize',
        action='store_true',
        help='Disable hybridization of gluon HybridBlocks.')

    # Model
    group = parser.add_argument_group('Model arguments')
    group.add_argument(
        '--emsize', type=int, default=100, help='Size of embedding vectors.')

    # Optimization options
    group = parser.add_argument_group('Optimization arguments')
    group.add_argument('--optimizer', type=str, default='adam')
    group.add_argument('--lr', type=float, default=0.05)
    group.add_argument('--batch_size', type=float, default=16)

    # Evaluation options
    evaluation.add_parameters(parser)

    args = parser.parse_args()
    evaluation.validate_args(args)
    return args


def get_label_mapping(train_labels):
    """
    Create the mapping from label to numeric label
    """
    sorted_labels = np.sort(np.unique(train_labels))
    label_mapping = {}
    for i, label in enumerate(sorted_labels):
        label_mapping[label] = i
    logging.info('Label mapping:%s', format(label_mapping))
    return label_mapping


def save_model(net, output_file):
    """This method saves the model to file"""
    net.save_parameters(output_file)


def get_context(args):
    """ This method gets context of execution"""
    context = None
    if args.gpu is None or args.gpu == '':
        context = mx.cpu()
    if isinstance(args.gpu, int):
        context = mx.gpu(args.gpu)
    return context


###############################################################################
# Training code
###############################################################################
def train(args):  # Load and clean data
    """
    Training function that orchestrates the Classification!
    """
    train_file = args.input
    test_file = args.validation
    ngram_range = args.ngrams
    logging.info('Ngrams range for the training run : %s', ngram_range)
    logging.info('Loading Training data')
    train_labels, train_data = read_input_data(train_file)
    tokens_list = []
    for x in train_data:
        tokens_list.extend(x.split())

    cntr = Counter(tokens_list)
    train_vocab = gluonnlp.Vocab(cntr)
    logging.info('Vocabulary size: %s', len(train_vocab))
    logging.info('Training data converting to sequences...')
    train_sequences = [train_vocab.to_indices(x.split()) for x in train_data]
    logging.info('Reading test dataset')
    test_labels, test_data = read_input_data(test_file)
    test_sequences = [train_vocab.to_indices(x.split()) for x in test_data]

    if ngram_range >= 2:
        logging.info('Adding %s-gram features', ngram_range)
        # Create set of unique n-gram from the training set.
        ngram_set = set()
        for input_list in train_sequences:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)
        start_index = len(cntr)
        token_indices = {v: k + start_index for k, v in enumerate(ngram_set)}
        train_sequences = add_ngram(train_sequences, token_indices,
                                    ngram_range)
        test_sequences = add_ngram(test_sequences, token_indices, ngram_range)
        logging.info('Added n-gram features to train and test datasets!! ')
    logging.info('Encoding labels')

    label_mapping = get_label_mapping(train_labels)
    y_train_final = list(map(lambda x: label_mapping[x], train_labels))
    y_test_final = list(map(lambda x: label_mapping[x], test_labels))

    num_classes = len(np.unique(train_labels))
    logging.info('Number of labels: %s', num_classes)
    logging.info('Initializing network')
    ctx = get_context(args)
    logging.info('Running Training on ctx:%s', ctx)
    embedding_dim = args.emsize
    net = FastTextClassificationModel(
        len(train_vocab), embedding_dim, num_classes)
    net.hybridize()
    net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    logging.info('Network initialized')

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    sigmoid_loss_fn = gluon.loss.SigmoidBinaryCrossEntropyLoss()

    loss_function = softmax_cross_entropy
    if num_classes == 2:
        logging.info(
            'Changing the loss function to sigmoid since its Binary Classification'
        )
        loss_function = sigmoid_loss_fn
    logging.info('Loss function for training:%s', loss_function)
    num_epochs = args.epochs
    batch_size = args.batch_size
    logging.info('Starting Training!')
    learning_rate = args.lr
    trainer1 = gluon.Trainer(net.embedding.collect_params(), 'adam',
                             {'learning_rate': learning_rate})
    trainer2 = gluon.Trainer(net.dense.collect_params(), 'adam',
                             {'learning_rate': learning_rate})
    train_batchify_fn = btf.Tuple(btf.Pad(), mx.nd.array)
    logging.info('Loading the training data to memory and creating sequences!')
    train_data_iter = mx.gluon.data.DataLoader(
        mx.gluon.data.ArrayDataset(train_sequences,
                                   mx.nd.array(y_train_final)),
        batch_size=batch_size,
        shuffle=False,
        batchify_fn=train_batchify_fn)
    logging.info('Loading the test data to memory and creating sequences')
    test_data_iter = mx.gluon.data.DataLoader(
        mx.gluon.data.ArrayDataset(test_sequences, mx.nd.array(y_test_final)),
        batch_size=2048,
        shuffle=False,
        batchify_fn=train_batchify_fn)

    num_batches = len(train_data) / batch_size
    display_batch_cadence = int(math.ceil(num_batches / 10))
    logging.info('Training on %s samples and testing on %s samples',
                 len(train_data), len(test_data))
    logging.info('Number of batches for each epoch : %s, Display cadence: %s',
                 num_batches, display_batch_cadence)
    for e in range(num_epochs):
        for batch, (data, label) in enumerate(train_data_iter):
            #num_batches += 1
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss = loss_function(output, label)
            loss.backward()
            trainer1.step(data.shape[0])
            trainer2.step(data.shape[0])
            if (batch % display_batch_cadence == 0):
                logging.info('Epoch : %s, Batches complete :%s', e, batch)
        logging.info('Epoch complete :%s, Computing Accuracy', e)

        test_accuracy, test_loss = evaluate_accuracy(
            test_data_iter, net, ctx, loss_function, num_classes)
        logging.info('Epochs completed : %s Test Accuracy: %s, Test Loss: %s',
                     e, test_accuracy, test_loss)
        learning_rate = learning_rate * 0.5
        trainer1.set_learning_rate(learning_rate)
        trainer2.set_learning_rate(learning_rate)
    save_model(net, args.output)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    args_ = parse_args()
    train(args_)
