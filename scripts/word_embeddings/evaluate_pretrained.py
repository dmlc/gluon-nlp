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
    group.add_argument('--embedding-name', type=str, default='fasttext',
                       help=('Name of embedding type to load. '
                             'Valid entries: {}'.format(
                                 ', '.join(
                                     nlp.embedding.list_sources().keys()))))
    group.add_argument('--embedding-source', type=str, default='wiki.simple',
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

    print(args)


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    args_ = get_args()
    ctx = utils.get_context(args_)[0]
    os.makedirs(args_.logdir, exist_ok=True)

    # Load pretrained embeddings
    print('Loading embedding ', args_.embedding_name, ' from ',
          args_.embedding_source)
    token_embedding = nlp.embedding.create(args_.embedding_name,
                                           source=args_.embedding_source)

    if args_.max_vocab_size:
        token_embedding._idx_to_token = \
            token_embedding._idx_to_token[:args_.max_vocab_size]
        token_embedding._idx_to_vec = \
            token_embedding._idx_to_vec[:args_.max_vocab_size]
        token_embedding._token_to_idx = {
            token: idx
            for idx, token in enumerate(token_embedding._idx_to_token)
        }

    similarity_results = evaluation.evaluate_similarity(
        args_, token_embedding, ctx, logfile=os.path.join(
            args_.logdir, 'similarity-{}.tsv'.format(args_.embedding_name)))
    analogy_results = evaluation.evaluate_analogy(
        args_, token_embedding, ctx, logfile=os.path.join(
            args_.logdir, 'analogy-{}.tsv'.format(args_.embedding_name)))
