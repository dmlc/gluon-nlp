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
# pylint: disable=logging-too-many-args
"""Evaluation of pretrained word embeddings
===========================================

This example shows how to load and perform intrinsic evaluation of word
embeddings using a variety of datasets all part of the Gluon NLP Toolkit.

"""

import argparse
import logging
import os
import sys

import evaluation
import gluonnlp as nlp
import utils


def get_args():
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        description='Word embedding evaluation with Gluon.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Embeddings arguments
    group = parser.add_argument_group('Embedding arguments')
    group.add_argument('--embedding-path', type=str,
                       help='Path to a .vec in Word2Vec text foramt or '
                       '.bin binary fastText model file. ')
    group.add_argument('--embedding-name', type=str,
                       help=('Name of embedding type to load. '
                             'Valid entries: {}'.format(
                                 ', '.join(
                                     nlp.embedding.list_sources().keys()))))
    group.add_argument('--embedding-source', type=str,
                       help=('Source from which to initialize the embedding.'
                             'Pass --list-embedding-sources to get a list of '
                             'valid sources for a given --embedding-name.'))
    group.add_argument(
        '--max-vocab-size', type=int, default=None,
        help=('Only retain the X first tokens from the pretrained embedding. '
              'The tokens are ordererd by decreasing frequency.'
              'As the analogy task takes the whole vocabulary into account, '
              'removing very infrequent words improves performance.'))
    group.add_argument('--list-embedding-sources', action='store_true')

    # Computation options
    group = parser.add_argument_group('Computation arguments')
    group.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size to use on analogy task. '
                       'Decrease batch size if evaluation crashes.')
    group.add_argument('--gpu', type=int,
                       help=('Number (index) of GPU to run on, e.g. 0. '
                             'If not specified, uses CPU.'))
    group.add_argument('--no-hybridize', action='store_true',
                       help='Disable hybridization of gluon HybridBlocks.')

    # Logging
    group = parser.add_argument_group('Logging arguments')
    group.add_argument('--logdir', type=str, default='logs',
                       help='Directory to store logs.')

    # Evaluation options
    evaluation.add_parameters(parser)

    args = parser.parse_args()

    validate_args(args)
    evaluation.validate_args(args)

    return args


def validate_args(args):
    """Validate provided arguments and act on --help."""
    if args.list_embedding_sources:
        print('Listing all sources for {} embeddings.'.format(
            args.embedding_name))
        print('Specify --embedding-name if you wish to '
              'list sources of other embeddings')
        print('')
        if args.embedding_name not in nlp.embedding.list_sources().keys():
            print('Invalid embedding name.')
            print('Only {} are supported.'.format(', '.join(
                nlp.embedding.list_sources().keys())))
            sys.exit(1)
        print(' '.join(nlp.embedding.list_sources()[args.embedding_name]))
        sys.exit(0)

    if not (args.embedding_path or args.embedding_name):
        print('You must specify either --embedding-path or --embedding-name ')
        print('Use --embedding-path to load and evaluate '
              'word embeddings from a Word2Vec text format '
              'or fastText binary format file')
        print('Use --embedding-name or to download one of '
              'the pretrained embedding files included in GluonNLP.')
        sys.exit(1)

    if args.embedding_name and not args.embedding_source:
        print('Please also specify --embedding-source'
              ' to select the version of the pretrained embedding. '
              'Use --list-embedding-sources to see all available sources')
        sys.exit(1)

    print(args)


def load_embedding_from_path(args):
    """Load a TokenEmbedding."""
    if 'bin' in args.embedding_path:
        with utils.print_time('load fastText model.'):
            model = \
                nlp.model.train.FasttextEmbeddingModel.load_fasttext_format(
                    args.embedding_path)

        # Add OOV words if the token_embedding can impute them
        token_set = set()
        token_set.update(
            filter(lambda x: x in model,
                   evaluation.get_tokens_in_evaluation_datasets(args)))

        # OOV words will be imputed and added to the
        # token_embedding.idx_to_token etc.
        with utils.print_time('compute vectors from subwords '
                              'for {} words.'.format(len(token_set))):
            embedding = nlp.embedding.TokenEmbedding(unknown_token=None,
                                                     allow_extend=True)
            idx_to_tokens = list(token_set)
            embedding[idx_to_tokens] = model[idx_to_tokens]

    else:
        embedding = nlp.embedding.TokenEmbedding.from_file(args.embedding_path)

    return embedding


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    args_ = get_args()
    ctx = utils.get_context(args_)[0]
    os.makedirs(args_.logdir, exist_ok=True)

    # Load pretrained embeddings
    if not args_.embedding_path:
        print('Loading embedding ', args_.embedding_name, ' from ',
              args_.embedding_source)
        token_embedding = nlp.embedding.create(args_.embedding_name,
                                               source=args_.embedding_source)
        name = '-' + args_.embedding_name + '-' + args_.embedding_source
    else:
        token_embedding = load_embedding_from_path(args_)
        name = ''

    if args_.max_vocab_size:
        if args_.embedding_path and '.bin' in args_.embedding_path:
            raise NotImplementedError(
                'Not implemented for binary fastText model.')

        size = min(len(token_embedding._idx_to_token), args_.max_vocab_size)
        token_embedding._idx_to_token = token_embedding._idx_to_token[:size]
        token_embedding._idx_to_vec = token_embedding._idx_to_vec[:size]
        token_embedding._token_to_idx = {
            token: idx
            for idx, token in enumerate(token_embedding._idx_to_token)
        }

    similarity_results = evaluation.evaluate_similarity(
        args_, token_embedding, ctx, logfile=os.path.join(
            args_.logdir, 'similarity{}.tsv'.format(name)))
    analogy_results = evaluation.evaluate_analogy(
        args_, token_embedding, ctx, logfile=os.path.join(
            args_.logdir, 'analogy{}.tsv'.format(name)))
