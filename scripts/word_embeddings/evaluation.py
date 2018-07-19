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
"""Evaluation
=============

Functions to perform evaluation of TokenEmbeddings on the datasets included in
the GluonNLP toolkit.

"""
import itertools
import sys
import logging
import json

import mxnet as mx
import numpy as np
from scipy import stats

import gluonnlp as nlp


def add_parameters(parser):
    """Add evaluation specific parameters to parser."""
    group = parser.add_argument_group('Evaluation arguments')

    group.add_argument('--eval-batch-size', type=int, default=1024)

    # Datasets
    group.add_argument(
        '--similarity-datasets', type=str,
        default=nlp.data.word_embedding_evaluation.word_similarity_datasets,
        nargs='*',
        help='Word similarity datasets to use for intrinsic evaluation.')
    group.add_argument(
        '--similarity-functions', type=str,
        default=nlp.embedding.evaluation.list_evaluation_functions(
            'similarity'), nargs='+',
        help='Word similarity functions to use for intrinsic evaluation.')
    group.add_argument(
        '--analogy-datasets', type=str, default=['GoogleAnalogyTestSet'],
        nargs='*',
        help='Word similarity datasets to use for intrinsic evaluation.')
    group.add_argument(
        '--analogy-functions', type=str,
        default=nlp.embedding.evaluation.list_evaluation_functions('analogy'),
        nargs='+',
        help='Word analogy functions to use for intrinsic evaluation. ')

    ## Analogy evaluation specific arguments
    group.add_argument(
        '--analogy-dont-exclude-question-words', action='store_true',
        help=('Exclude input words from valid output analogies.'
              'The performance of word embeddings on the analogy task '
              'is around 0% accuracy if input words are not excluded.'))


def validate_args(args):
    """Validate provided arguments and act on --help."""
    # Check correctness of similarity dataset names
    for dataset_name in args.similarity_datasets:
        if dataset_name.lower() not in map(
                str.lower,
                nlp.data.word_embedding_evaluation.word_similarity_datasets):
            print('{} is not a supported dataset.'.format(dataset_name))
            sys.exit(1)

    # Check correctness of analogy dataset names
    for dataset_name in args.analogy_datasets:
        if dataset_name.lower() not in map(
                str.lower,
                nlp.data.word_embedding_evaluation.word_analogy_datasets):
            print('{} is not a supported dataset.'.format(dataset_name))
            sys.exit(1)


def iterate_similarity_datasets(args):
    """Generator over all similarity evaluation datasets.

    Iteratos over dataset names, keyword arguments for their creation and the
    created dataset.

    """
    for dataset_name in args.similarity_datasets:
        parameters = nlp.data.list_datasets(dataset_name)
        for key_values in itertools.product(*parameters.values()):
            kwargs = dict(zip(parameters.keys(), key_values))
            yield dataset_name, kwargs, nlp.data.create(dataset_name, **kwargs)


def iterate_analogy_datasets(args):
    """Generator over all analogy evaluation datasets.

    Iteratos over dataset names, keyword arguments for their creation and the
    created dataset.

    """
    for dataset_name in args.analogy_datasets:
        parameters = nlp.data.list_datasets(dataset_name)
        for key_values in itertools.product(*parameters.values()):
            kwargs = dict(zip(parameters.keys(), key_values))
            yield dataset_name, kwargs, nlp.data.create(dataset_name, **kwargs)


def get_tokens_in_evaluation_datasets(args):
    """Returns a set of all tokens occuring the evaluation datasets."""
    tokens = set()
    for _, _, dataset in iterate_similarity_datasets(args):
        tokens.update(
            itertools.chain.from_iterable((d[0], d[1]) for d in dataset))

    for _, _, dataset in iterate_analogy_datasets(args):
        tokens.update(
            itertools.chain.from_iterable(
                (d[0], d[1], d[2], d[3]) for d in dataset))

    return tokens


def evaluate_similarity(args, token_embedding, ctx, logfile=None,
                        global_step=0):
    """Evaluate on specified similarity datasets."""

    results = []
    for similarity_function in args.similarity_functions:
        evaluator = nlp.embedding.evaluation.WordEmbeddingSimilarity(
            idx_to_vec=token_embedding.idx_to_vec,
            similarity_function=similarity_function)
        evaluator.initialize(ctx=ctx)
        if not args.no_hybridize:
            evaluator.hybridize()

        # Evaluate all datasets
        for (dataset_name, dataset_kwargs,
             dataset) in iterate_similarity_datasets(args):
            initial_length = len(dataset)
            dataset_coded = [[
                token_embedding.token_to_idx[d[0]],
                token_embedding.token_to_idx[d[1]], d[2]
            ] for d in dataset if d[0] in token_embedding.token_to_idx
                             and d[1] in token_embedding.token_to_idx]
            num_dropped = initial_length - len(dataset_coded)

            words1, words2, scores = zip(*dataset_coded)
            pred_similarity = evaluator(
                mx.nd.array(words1, ctx=ctx), mx.nd.array(words2, ctx=ctx))
            sr = stats.spearmanr(pred_similarity.asnumpy(), np.array(scores))
            logging.info('Spearman rank correlation on %s %s with %s:\t%s',
                         dataset.__class__.__name__, str(dataset_kwargs),
                         similarity_function, sr.correlation)

            result = dict(
                task='similarity',
                dataset_name=dataset_name,
                dataset_kwargs=dataset_kwargs,
                similarity_function=similarity_function,
                spearmanr=sr.correlation,
                num_dropped=num_dropped,
                global_step=global_step,
            )
            log_similarity_result(logfile, result)
            results.append(result)

    return results


def evaluate_analogy(args, token_embedding, ctx, logfile=None, global_step=0):
    """Evaluate on specified analogy datasets.

    The analogy task is an open vocabulary task, make sure to pass a
    token_embedding with a sufficiently large number of supported tokens.

    """
    results = []
    exclude_question_words = not args.analogy_dont_exclude_question_words
    for analogy_function in args.analogy_functions:
        evaluator = nlp.embedding.evaluation.WordEmbeddingAnalogy(
            idx_to_vec=token_embedding.idx_to_vec,
            exclude_question_words=exclude_question_words,
            analogy_function=analogy_function)
        evaluator.initialize(ctx=ctx)
        if not args.no_hybridize:
            evaluator.hybridize()

        for (dataset_name, dataset_kwargs,
             dataset) in iterate_analogy_datasets(args):
            initial_length = len(dataset)
            dataset_coded = [[
                token_embedding.token_to_idx[d[0]],
                token_embedding.token_to_idx[d[1]],
                token_embedding.token_to_idx[d[2]],
                token_embedding.token_to_idx[d[3]]
            ] for d in dataset if d[0] in token_embedding.token_to_idx
                             and d[1] in token_embedding.token_to_idx
                             and d[2] in token_embedding.token_to_idx
                             and d[3] in token_embedding.token_to_idx]
            num_dropped = initial_length - len(dataset_coded)

            dataset_coded_batched = mx.gluon.data.DataLoader(
                dataset_coded, batch_size=args.eval_batch_size)

            acc = mx.metric.Accuracy()
            for batch in dataset_coded_batched:
                batch = batch.as_in_context(ctx)
                words1, words2, words3, words4 = (batch[:, 0], batch[:, 1],
                                                  batch[:, 2], batch[:, 3])
                pred_idxs = evaluator(words1, words2, words3)
                acc.update(pred_idxs[:, 0], words4.astype(np.float32))

            logging.info(
                'Accuracy on %s %s with %s:\t%s', dataset.__class__.__name__,
                str(dataset_kwargs), analogy_function,
                acc.get()[1])

            result = dict(
                task='analogy',
                dataset_name=dataset_name,
                dataset_kwargs=dataset_kwargs,
                analogy_function=analogy_function,
                accuracy=acc.get()[1],
                num_dropped=num_dropped,
                global_step=global_step,
            )
            log_analogy_result(logfile, result)
            results.append(result)
    return results


def log_similarity_result(logfile, result):
    """Log a similarity evaluation result dictionary as TSV to logfile."""
    assert result['task'] == 'similarity'

    if not logfile:
        return

    with open(logfile, 'a') as f:
        f.write('\t'.join([
            str(result['global_step']),
            result['task'],
            result['dataset_name'],
            json.dumps(result['dataset_kwargs']),
            result['similarity_function'],
            str(result['spearmanr']),
            str(result['num_dropped']),
        ]))

        f.write('\n')


def log_analogy_result(logfile, result):
    """Log a analogy evaluation result dictionary as TSV to logfile."""
    assert result['task'] == 'analogy'

    if not logfile:
        return

    with open(logfile, 'a') as f:
        f.write('\t'.join([
            str(result['global_step']),
            result['task'],
            result['dataset_name'],
            json.dumps(result['dataset_kwargs']),
            result['analogy_function'],
            str(result['accuracy']),
            str(result['num_dropped']),
        ]))
        f.write('\n')
