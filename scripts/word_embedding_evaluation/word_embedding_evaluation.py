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
# pylint: disable=eval-used, redefined-outer-name
"""Word Embeddings
===============

This example shows how to load and perform intrinsic evaluation of word
embeddings using a variety of datasets all part of the Gluon NLP Toolkit.

"""

import argparse
import time
import sys
import logging

import numpy as np
import mxnet as mx
from scipy import stats

import gluonnlp as nlp

try:
    import progressbar
except ImportError:
    logging.warning(
        'progressbar not installed. '
        ' Install via `pip install progressbar2` for better usability.')
    progressbar = None

parser = argparse.ArgumentParser(
    description='Word embedding training with Gluon.')

# Embeddings arguments
group = parser.add_argument_group('Embedding arguments')
group.add_argument('--embedding-name', type=str, default='fasttext',
                   help=('Name of embedding type to load. '
                         'Valid entries: {}'.format(', '.join(
                             nlp.embedding.list_sources().keys()))))
group.add_argument('--embedding-source', type=str, default='wiki.simple.vec',
                   help=('Source from which to initialize the embedding.'
                         'Pass --list-embedding-sources to get a list of '
                         'valid sources for a given --embedding-name.'))
group.add_argument('--list-embedding-sources', action='store_true')

# Evaluation arguments
group = parser.add_argument_group('Evaluation arguments')
group.add_argument('--ignore_oov', action='store_true',
                   help='Drop OOV words from evaluation datasets.')
## Datasets
group.add_argument(
    '--similarity_datasets', type=str, default='*', nargs='*',
    help='Word similarity datasets to use for intrinsic evaluation. '
    '"*" selects all datasets. Empty argument disables.  Or specify by name: '
    '{}'.format(' '.join(
        nlp.data.word_embedding_evaluation.word_similarity_datasets)))
group.add_argument(
    '--analogy_datasets', type=str, default='*', nargs='*',
    help='Word similarity datasets to use for intrinsic evaluation. '
    '"*" selects all datasets. Empty argument disables. Or specify by name: '
    '{}'.format(' '.join(
        nlp.data.word_embedding_evaluation.word_analogy_datasets)))
## Analogy evaluation specific arguments
group.add_argument(
    '--analogy_dont_exclude_inputs', action='store_true',
    help=('Exclude input words from valid output analogies.'
          'The performance of word embeddings on the analogy task '
          'is around 0% accuracy if input words are not excluded.'))

# Computation options
group = parser.add_argument_group('Computation arguments')
group.add_argument('--batch-size', type=int, default=128,
                   help='Batch size to use on analogy task.')
group.add_argument('--gpu', type=int,
                   help=('Number (index) of GPU to run on, e.g. 0. '
                         'If not specified, uses CPU.'))
group.add_argument('--dont-hybridize', action='store_true',
                   help='Disable hybridization of gluon HybridBlocks.')

args = parser.parse_args()

###############################################################################
# Parse arguments
###############################################################################
if args.list_embedding_sources:
    print('Listing all sources for {} embeddings.'.format(args.embedding_name))
    print('Specify --embedding-name if you wish to '
          'list sources of other embeddings')
    print('')
    if args.embedding_name not in nlp.embedding.list_sources().keys():
        print("Invalid embedding name.")
        print("Only {} are supported.".format(', '.join(
            nlp.embedding.list_sources().keys())))
        sys.exit(1)
    print(' '.join(nlp.embedding.list_sources()[args.embedding_name]))
    sys.exit(0)

print(args)

# Load word similarity datasets
similarity_datasets = []
similarity_datasets_classnames = \
    nlp.data.word_embedding_evaluation.word_similarity_datasets
if args.similarity_datasets == '*':
    args.similarity_datasets = similarity_datasets_classnames
for dataset_name in args.similarity_datasets:
    if dataset_name not in similarity_datasets_classnames:
        print('{} is not a supported dataset.'.format(dataset_name))
        sys.exit(1)
    # TODO use registry instead
    ds_class = eval('nlp.data.{}'.format(dataset_name))
    similarity_datasets.append(ds_class)

# Load word analogy datasets
analogy_datasets = []
analogy_datasets_classnames = \
    nlp.data.word_embedding_evaluation.word_analogy_datasets
if args.analogy_datasets == '*':
    args.analogy_datasets = analogy_datasets_classnames
for dataset_name in args.analogy_datasets:
    if dataset_name not in analogy_datasets_classnames:
        print('{} is not a supported dataset.'.format(dataset_name))
        sys.exit(1)
    # TODO use registry instead
    ds_class = eval('nlp.data.{}'.format(dataset_name))
    analogy_datasets.append(ds_class)

# Context
if args.gpu is None or args.gpu == '':
    context = mx.cpu()
else:
    context = mx.gpu(int(args.gpu))

# Set up logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

###############################################################################
# Evaluation
###############################################################################
def evaluate_similarity(token_embedding, dataset):
    # Closed vocabulary: Only need the words occuring in the dataset
    counter = nlp.data.utils.Counter(w for wpair in dataset for w in wpair[:2])
    vocab = nlp.vocab.Vocab(counter)
    vocab.set_embedding(token_embedding)

    if args.ignore_oov:
        initial_length = len(dataset)
        dataset = [d for d in dataset if d[0] in vocab and d[1] in vocab]
        num_dropped = initial_length - len(dataset)
        if num_dropped:
            logging.warning("Dropped {} pairs from {} as the were OOV.".format(
                num_dropped, dataset.__class__.__name__))

    dataset_coded = [[vocab[d[0]], vocab[d[1]], d[2]] for d in dataset]
    words1, words2, scores = zip(*dataset_coded)

    evaluator = nlp.model.WordEmbeddingSimilarity(
        idx_to_vec=vocab.embedding.idx_to_vec)
    evaluator.initialize(ctx=context)
    if not args.dont_hybridize:
        evaluator.hybridize()

    pred_similarity = evaluator(
        mx.nd.array(words1, ctx=context), mx.nd.array(words2, ctx=context))

    sr = nlp.metric.SpearmanRankCorrelation()
    sr.update(mx.nd.array(scores), pred_similarity.as_in_context(mx.cpu()))
    logging.info('Spearman rank correlation on {}: {}'.format(
        dataset.__class__.__name__,
        sr.get()[1]))


def evaluate_analogy(token_embedding, dataset):
    # Open vocabulary: Use all known words
    counter = nlp.data.utils.Counter(token_embedding.idx_to_token)
    vocab = nlp.vocab.Vocab(counter)
    vocab.set_embedding(token_embedding)

    if args.ignore_oov:
        initial_length = len(dataset)
        dataset = [
            d for d in dataset if d[0] in vocab and d[1] in vocab
            and d[2] in vocab and d[3] in vocab
        ]
        num_dropped = initial_length - len(dataset)
        if num_dropped:
            logging.warning("Dropped {} pairs from {} as the were OOV.".format(
                num_dropped, dataset.__class__.__name__))

    dataset_coded = [[vocab[d[0]], vocab[d[1]], vocab[d[2]], vocab[d[3]]]
                     for d in dataset]
    dataset_coded_batched = mx.gluon.data.DataLoader(
        dataset_coded, batch_size=args.batch_size)
    exclude_inputs = not args.analogy_dont_exclude_inputs
    evaluator = nlp.model.WordEmbeddingAnalogy(
        idx_to_vec=vocab.embedding.idx_to_vec, exclude_inputs=exclude_inputs)
    evaluator.initialize(ctx=context)
    if not args.dont_hybridize:
        evaluator.hybridize()

    acc = mx.metric.Accuracy()
    if progressbar is not None:
        dataset_coded_batched = progressbar.progressbar(dataset_coded_batched)
    for batch in dataset_coded_batched:
        batch = batch.as_in_context(context)
        words1, words2, words3, words4 = (batch[:, 0], batch[:, 1],
                                          batch[:, 2], batch[:, 3])
        pred_idxs = evaluator(words1, words2, words3)
        acc.update(pred_idxs[:, 0], words4.astype(np.float32))

    logging.info('Accuracy on {}: {}'.format(dataset.__class__.__name__,
                                             acc.get()[1]))


def evaluate():
    # Load pretrained embeddings
    print('Loading embedding ', args.embedding_name, ' from ',
          args.embedding_source)
    token_embedding = nlp.embedding.create(args.embedding_name,
                                           source=args.embedding_source)

    # Similarity based evaluation
    for dataset_class in similarity_datasets:
        logging.info('Starting evaluation of {}'.format(
            dataset_class.__name__))
        dataset = dataset_class()
        evaluate_similarity(token_embedding, dataset)

    # Analogy based evaluation
    for dataset_class in analogy_datasets:
        logging.info('Starting evaluation of {}'.format(
            dataset_class.__name__))
        dataset = dataset_class()
        evaluate_analogy(token_embedding, dataset)


if __name__ == '__main__':
    start_pipeline_time = time.time()
    evaluate()
