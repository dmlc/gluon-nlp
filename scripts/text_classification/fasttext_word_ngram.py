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
import time

from collections import Counter
import multiprocessing as mp
import numpy as np
from mxnet import nd, autograd
from mxnet.gluon import nn, HybridBlock
import mxnet as mx
import mxnet.gluon as gluon

import gluonnlp


class MeanPoolingLayer(gluon.HybridBlock):
    """A block for mean pooling of encoder features"""

    def __init__(self, prefix=None, params=None):
        super(MeanPoolingLayer, self).__init__(prefix=prefix, params=params)

    def hybrid_forward(self, F, data, valid_length):  # pylint: disable=arguments-differ
        """Forward logic"""
        # Data will have shape (T, N, C)
        masked_encoded = F.SequenceMask(data,
                                        sequence_length=valid_length,
                                        use_sequence_length=True)
        agg_state = F.broadcast_div(F.sum(masked_encoded, axis=0),
                                    F.expand_dims(valid_length, axis=1))
        return agg_state


class FastTextClassificationModel(HybridBlock):
    """
    The Model Block for FastTextClassification Model.
    The trained embeddings layer, is averaged and then softmax
    layer is applied on top of it.
    """

    def __init__(self, vocab_size, embedding_dim, num_classes, **kwargs):
        super(FastTextClassificationModel, self).__init__(**kwargs)
        with self.name_scope():
            self.vocab_size = vocab_size
            self.embedding_dim = embedding_dim
            self.embedding = nn.Embedding(
                self.vocab_size,
                self.embedding_dim,
                weight_initializer=mx.init.Xavier(),
                dtype='float32')
            num_output_units = num_classes
            if num_classes == 2:
                num_output_units = 1
            logging.info('Number of output units in the last layer :%s',
                         num_output_units)
            self.agg_layer = MeanPoolingLayer()
            self.dense = nn.Dense(num_output_units)

    def hybrid_forward(self, F, doc, valid_length):  # pylint: disable=arguments-differ
        doc = doc.swapaxes(dim1=0, dim2=1)
        embeddings = self.embedding(doc)
        mean_pooled = self.agg_layer(embeddings, valid_length)
        dense_output = self.dense(mean_pooled)
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
    for i, ((data, length), label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)  # .reshape((-1,784))
        length = length.astype('float32').as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data, length)
        loss = loss_fun(output, label)
        preds = []
        if num_classes == 2:
            preds = (nd.sign(output) + 1) / 2
            preds = preds.reshape(-1)
        else:
            preds = nd.argmax(output, axis=1)
        acc.update(preds=preds, labels=label)
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
        '--batch_size', type=int, default=16, help='Batch size for training.')
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

    args = parser.parse_args()
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


def get_length(inp):
    """Returns the length"""
    return float(len(inp[0]))


def get_sequence(inpt):
    """Transforms input to vocab id's"""
    document = inpt[0]
    vocab = inpt[1]
    return vocab[document.split()]


def convert_to_sequences(dataset, vocab):
    """This function takes a dataset and converts
    it into sequences via multiprocessing
    """
    start = time.time()
    dataset_vocab = map(lambda x: (x, vocab), dataset)
    with mp.Pool() as pool:
        # Each sample is processed in an asynchronous manner.
        output = pool.map(get_sequence, dataset_vocab)
    end = time.time()
    logging.info('Done! Sequence conversion Time={:.2f}s, #Sentences={}'
                 .format(end - start, len(dataset)))
    return output


def preprocess_dataset(dataset, labels):
    """ Preprocess and prepare a dataset"""
    start = time.time()
    with mp.Pool() as pool:
        # Each sample is processed in an asynchronous manner.
        dataset = gluon.data.SimpleDataset(list(zip(dataset, labels)))
        lengths = gluon.data.SimpleDataset(pool.map(get_length, dataset))
    end = time.time()
    logging.info('Done! Preprocessing Time={:.2f}s, #Sentences={}'
                 .format(end - start, len(dataset)))
    return dataset, lengths


def get_dataloader(train_dataset, train_data_lengths,
                   test_dataset, batch_size):
    """ Construct the DataLoader. Pad data, stack label and lengths"""
    bucket_num, bucket_ratio = 20, 0.2
    batchify_fn = gluonnlp.data.batchify.Tuple(
        gluonnlp.data.batchify.Pad(axis=0, ret_length=True),
        gluonnlp.data.batchify.Stack(dtype='float32'))
    batch_sampler = gluonnlp.data.sampler.FixedBucketSampler(
        train_data_lengths,
        batch_size=batch_size,
        num_buckets=bucket_num,
        ratio=bucket_ratio,
        shuffle=True)
    train_dataloader = gluon.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=batch_sampler,
        batchify_fn=batchify_fn)
    test_dataloader = gluon.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        batchify_fn=batchify_fn)
    return train_dataloader, test_dataloader

###############################################################################
# Training code
###############################################################################
def train(args):
    """Training function that orchestrates the Classification! """
    train_file = args.input
    test_file = args.validation
    ngram_range = args.ngrams
    logging.info('Ngrams range for the training run : %s', ngram_range)
    logging.info('Loading Training data')
    train_labels, train_data = read_input_data(train_file)
    logging.info('Loading Test data')
    test_labels, test_data = read_input_data(test_file)
    tokens_list = []
    for line in train_data:
        tokens_list.extend(line.split())

    cntr = Counter(tokens_list)
    train_vocab = gluonnlp.Vocab(cntr)
    logging.info('Vocabulary size: %s', len(train_vocab))
    logging.info('Training data converting to sequences...')
    embedding_matrix_len = len(train_vocab)
    # Preprocess the dataset
    train_sequences = convert_to_sequences(train_data, train_vocab)
    test_sequences = convert_to_sequences(test_data, train_vocab)

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
        embedding_matrix_len = embedding_matrix_len + len(token_indices)
        train_sequences = add_ngram(train_sequences, token_indices,
                                    ngram_range)
        test_sequences = add_ngram(test_sequences, token_indices, ngram_range)
        logging.info('Added n-gram features to train and test datasets!! ')
    logging.info('Encoding labels')

    label_mapping = get_label_mapping(train_labels)
    y_train_final = list(map(lambda x: label_mapping[x], train_labels))
    y_test_final = list(map(lambda x: label_mapping[x], test_labels))

    train_sequences, train_data_lengths = preprocess_dataset(
        train_sequences, y_train_final)
    test_sequences, _ = preprocess_dataset(
        test_sequences, y_test_final)
    train_dataloader, test_dataloader = get_dataloader(train_sequences, train_data_lengths,
                                                       test_sequences, args.batch_size)

    num_classes = len(np.unique(train_labels))
    logging.info('Number of labels: %s', num_classes)
    logging.info('Initializing network')
    ctx = get_context(args)
    logging.info('Running Training on ctx:%s', ctx)
    embedding_dim = args.emsize
    logging.info('Embedding Matrix Length:%s', embedding_matrix_len)
    net = FastTextClassificationModel(
        embedding_matrix_len, embedding_dim, num_classes)
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
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': learning_rate})
    num_batches = len(train_data) / batch_size
    display_batch_cadence = int(math.ceil(num_batches / 10))
    logging.info('Training on %s samples and testing on %s samples',
                 len(train_data), len(test_data))
    logging.info('Number of batches for each epoch : %s, Display cadence: %s',
                 num_batches, display_batch_cadence)
    for epoch in range(num_epochs):
        for batch, ((data, length), label) in enumerate(train_dataloader):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            length = length.astype('float32').as_in_context(ctx)
            with autograd.record():
                output = net(data, length)
                loss = loss_function(output, label)
            loss.backward()
            trainer.step(data.shape[0])
            if batch % display_batch_cadence == 0:
                logging.info('Epoch : %s, Batches complete :%s', epoch, batch)
        logging.info('Epoch complete :%s, Computing Accuracy', epoch)
        test_accuracy, test_loss = evaluate_accuracy(
            test_dataloader, net, ctx, loss_function, num_classes)
        logging.info('Epochs completed : %s Test Accuracy: %s, Test Loss: %s',
                     epoch, test_accuracy, test_loss)
        learning_rate = learning_rate * 0.5
        trainer.set_learning_rate(learning_rate)
    save_model(net, args.output)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    arguments = parse_args()
    train(arguments)
