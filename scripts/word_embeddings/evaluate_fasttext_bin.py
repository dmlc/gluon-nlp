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
"""Loading and evaluation of pretrained fasttext word embeddings
================================================================

This example shows how to load the binary format containing word and subword
information and perform intrinsic evaluation of word embeddings trained with
the facebookresearch/fasttext implementation using a variety of datasets all
part of the Gluon NLP Toolkit. The example makes use of gensim for reading the
binary file format.

"""

import argparse
import logging
import os
import struct

import mxnet as mx
import numpy as np

import evaluation
import gensim
import gluonnlp as nlp
import utils


def get_args():
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        description='Word embedding evaluation with Gluon.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Embeddings arguments
    group = parser.add_argument_group('Embedding arguments')
    group.add_argument('path', type=str,
                       help='Path to pretrained TokenEmbedding file.')
    group.add_argument(
        '--max-vocab-size', type=int, default=None,
        help=('Only retain the X first tokens from the pretrained embedding. '
              'The tokens are ordererd by decreasing frequency.'
              'As the analogy task takes the whole vocabulary into account, '
              'removing very infrequent words improves performance.'))
    group.add_argument('--list-embedding-sources', action='store_true')

    # Computation options
    group = parser.add_argument_group('Computation arguments')
    group.add_argument('--batch-size', type=int, default=32,
                       help='Batch size to use on analogy task.'
                       'Decrease batch size if evaluation crashes.')
    group.add_argument('--gpu', type=int, nargs='+',
                       help=('Number (index) of GPU to run on, e.g. 0. '
                             'If not specified, uses CPU.'))
    group.add_argument('--no-hybridize', action='store_true',
                       help='Disable hybridization of gluon HybridBlocks '
                       'used for evaluation.')

    # Logging
    group = parser.add_argument_group('Logging arguments')
    group.add_argument('--logdir', type=str, default='logs',
                       help='Directory to store logs.')

    # Evaluation options
    evaluation.add_parameters(parser)

    args = parser.parse_args()

    evaluation.validate_args(args)

    return args


def get_model(args):
    """Load the pretrained model."""
    context = utils.get_context(args)

    assert '.bin' in args.path  # Assume binary fasttext format

    gensim_fasttext = gensim.models.FastText()
    gensim_fasttext.file_name = args.path
    with open(args.path, 'rb') as f:
        gensim_fasttext._load_model_params(f)
        gensim_fasttext._load_dict(f)

        if gensim_fasttext.new_format:
            # quant input
            gensim_fasttext.struct_unpack(f, '@?')
        num_vectors, dim = gensim_fasttext.struct_unpack(f, '@2q')
        assert gensim_fasttext.wv.vector_size == dim
        dtype = np.float32 if struct.calcsize('@f') == 4 else np.float64
        matrix = np.fromfile(f, dtype=dtype, count=num_vectors * dim)
        matrix = matrix.reshape((-1, dim))

        num_words = len(gensim_fasttext.wv.vocab)
        num_subwords = gensim_fasttext.bucket
        assert num_words + num_subwords == num_vectors

    if args.max_vocab_size:
        idx_to_token = list(
            gensim_fasttext.wv.vocab.keys())[:args.max_vocab_size]
        idx_to_vec = mx.nd.array(matrix[:len(idx_to_token)])
        token_to_idx = {(token, idx) for idx, token in enumerate(idx_to_token)}
    else:
        idx_to_token = list(gensim_fasttext.wv.vocab.keys())
        idx_to_vec = mx.nd.array(matrix[:num_words])
        token_to_idx = {(token, idx) for idx, token in enumerate(idx_to_token)}

    if num_subwords:
        subword_function = nlp.vocab.create_subword_function(
            'NGramHashes', num_subwords=num_subwords)

        embedding = nlp.model.train.FasttextEmbeddingModel(
            token_to_idx=token_to_idx,
            subword_function=subword_function,
            embedding_size=dim,
        )

        embedding.initialize(ctx=context[0])
        embedding.embedding.weight.set_data(idx_to_vec)
        embedding.subword_embedding.embedding.weight.set_data(
            mx.nd.array(matrix[num_words:]))
    else:
        print('Loaded model does not contain subwords.')

        embedding = nlp.model.train.SimpleEmbeddingModel(
            token_to_idx=token_to_idx,
            embedding_size=dim,
        )

        embedding.initialize(ctx=context[0])
        embedding.embedding.weight.set_data(idx_to_vec)

    return embedding, idx_to_token


def load_and_evaluate(args):
    """Load the pretrained model and run evaluate."""
    context = utils.get_context(args)
    embedding, model_idx_to_token = get_model(args)

    idx_to_token_set = evaluation.get_tokens_in_evaluation_datasets(args)
    idx_to_token_set.update(model_idx_to_token)
    idx_to_token = list(idx_to_token_set)

    # Compute their word vectors
    token_embedding = embedding.to_token_embedding(idx_to_token,
                                                   ctx=context[0])

    os.makedirs(args.logdir, exist_ok=True)
    results = evaluation.evaluate_similarity(
        args, token_embedding, context[0], logfile=os.path.join(
            args.logdir, 'similarity.tsv'))
    results += evaluation.evaluate_analogy(args, token_embedding, context[0],
                                           logfile=os.path.join(
                                               args.logdir, 'analogy.tsv'))


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    args_ = get_args()
    load_and_evaluate(args_)
