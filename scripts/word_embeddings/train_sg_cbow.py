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
"""SkipGram and CBOW embedding models
=====================================

This example shows how to train SkipGram (SG) and Continuous Bag of Words
(CBOW) embedding models with the Gluon NLP Toolkit. Including fastText style
subword information is supported.

The SG and CBOW models were introduced by "Mikolov et al. Efficient estimation
of word representations in vector space. ICLR Workshop, 2013". The fastText
model was introduced by "Bojanowski et al. Enriching word vectors with subword
information. TACL 2017"

"""
import argparse
import logging
import os
import random
import sys
import time

import mxnet as mx
import numpy as np

import gluonnlp as nlp
import evaluation
from utils import get_context, print_time
from model import SG, CBOW
from data import transform_data_word2vec, transform_data_fasttext, preprocess_dataset, wiki


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Word embedding training with Gluon.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data options
    group = parser.add_argument_group('Data arguments')
    group.add_argument('--data', type=str, default='text8',
                       help='Training dataset.')
    group.add_argument('--wiki-root', type=str, default='text8',
                       help='Root under which preprocessed wiki dump.')
    group.add_argument('--wiki-language', type=str, default='text8',
                       help='Language of wiki dump.')
    group.add_argument('--wiki-date', help='Date of wiki dump.')

    # Computation options
    group = parser.add_argument_group('Computation arguments')
    group.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size for training.')
    group.add_argument('--epochs', type=int, default=5, help='Epoch limit')
    group.add_argument(
        '--gpu', type=int, nargs='+',
        help=('Number (index) of GPU to run on, e.g. 0. '
              'If not specified, uses CPU.'))
    group.add_argument('--no-prefetch-batch', action='store_true',
                       help='Disable multi-threaded nogil batch prefetching.')
    group.add_argument('--num-prefetch-epoch', type=int, default=3,
                       help='Start data pipeline for next N epochs when beginning current epoch.')
    group.add_argument('--no-hybridize', action='store_true',
                       help='Disable hybridization of gluon HybridBlocks.')

    # Model
    group = parser.add_argument_group('Model arguments')
    group.add_argument('--emsize', type=int, default=300,
                       help='Size of embedding vectors.')
    group.add_argument('--ngrams', type=int, nargs='+', default=[3, 4, 5, 6])
    group.add_argument(
        '--ngram-buckets', type=int, default=2000000,
        help='Size of word_context set of the ngram hash function. '
        'Set this to 0 for Word2Vec style training.')
    group.add_argument('--model', type=str, default='skipgram',
                       help='SkipGram or CBOW.')
    group.add_argument('--window', type=int, default=5,
                       help='Context window size.')
    group.add_argument(
        '--negative', type=int, default=5, help='Number of negative samples '
        'per source-context word pair.')
    group.add_argument('--frequent-token-subsampling', type=float,
                       default=1E-4,
                       help='Frequent token subsampling constant.')
    group.add_argument(
        '--max-vocab-size', type=int,
        help='Limit the number of words considered. '
        'OOV words will be ignored.')

    # Optimization options
    group = parser.add_argument_group('Optimization arguments')
    group.add_argument('--optimizer', type=str, default='groupadagrad')
    group.add_argument('--lr', type=float, default=0.1)
    group.add_argument('--seed', type=int, default=1, help='random seed')

    # Logging
    group = parser.add_argument_group('Logging arguments')
    group.add_argument('--logdir', type=str, default='logs',
                       help='Directory to store logs.')
    group.add_argument('--log-interval', type=int, default=100)
    group.add_argument(
        '--eval-interval', type=int,
        help='Evaluate every --eval-interval iterations '
        'in addition to at the end of every epoch.')
    group.add_argument('--no-eval-analogy', action='store_true',
                       help='Don\'t evaluate on the analogy task.')

    # Evaluation options
    evaluation.add_parameters(parser)

    args = parser.parse_args()
    evaluation.validate_args(args)

    random.seed(args.seed)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    return args


def train(args):
    """Training helper."""
    if not args.model.lower() in ['cbow', 'skipgram']:
        logging.error('Unsupported model %s.', args.model)
        sys.exit(1)

    if args.data.lower() == 'toy':
        data = mx.gluon.data.SimpleDataset(nlp.data.Text8(segment='train')[:2])
        data, vocab, idx_to_counts = preprocess_dataset(
            data, max_vocab_size=args.max_vocab_size)
    elif args.data.lower() == 'text8':
        data = nlp.data.Text8(segment='train')
        data, vocab, idx_to_counts = preprocess_dataset(
            data, max_vocab_size=args.max_vocab_size)
    elif args.data.lower() == 'fil9':
        data = nlp.data.Fil9(max_sentence_length=10000)
        data, vocab, idx_to_counts = preprocess_dataset(
            data, max_vocab_size=args.max_vocab_size)
    elif args.data.lower() == 'wiki':
        data, vocab, idx_to_counts = wiki(args.wiki_root, args.wiki_date,
                                          args.wiki_language,
                                          args.max_vocab_size)

    if args.ngram_buckets > 0:
        data, batchify_fn, subword_function = transform_data_fasttext(
            data, vocab, idx_to_counts, cbow=args.model.lower() == 'cbow',
            ngram_buckets=args.ngram_buckets, ngrams=args.ngrams,
            batch_size=args.batch_size, window_size=args.window,
            frequent_token_subsampling=args.frequent_token_subsampling)
    else:
        subword_function = None
        data, batchify_fn = transform_data_word2vec(
            data, vocab, idx_to_counts, cbow=args.model.lower() == 'cbow',
            batch_size=args.batch_size, window_size=args.window,
            frequent_token_subsampling=args.frequent_token_subsampling)

    num_tokens = float(sum(idx_to_counts))

    model = CBOW if args.model.lower() == 'cbow' else SG
    embedding = model(token_to_idx=vocab.token_to_idx, output_dim=args.emsize,
                      batch_size=args.batch_size, num_negatives=args.negative,
                      negatives_weights=mx.nd.array(idx_to_counts),
                      subword_function=subword_function)
    context = get_context(args)
    embedding.initialize(ctx=context)
    if not args.no_hybridize:
        embedding.hybridize(static_alloc=True, static_shape=True)

    optimizer_kwargs = dict(learning_rate=args.lr)
    try:
        trainer = mx.gluon.Trainer(embedding.collect_params(), args.optimizer,
                                   optimizer_kwargs)
    except ValueError as e:
        if args.optimizer == 'groupadagrad':
            logging.warning('MXNet <= v1.3 does not contain '
                            'GroupAdaGrad support. Falling back to AdaGrad')
            trainer = mx.gluon.Trainer(embedding.collect_params(), 'adagrad',
                                       optimizer_kwargs)
        else:
            raise e

    try:
        if args.no_prefetch_batch:
            data = data.transform(batchify_fn)
        else:
            from executors import LazyThreadPoolExecutor
            num_cpu = len(os.sched_getaffinity(0))
            ex = LazyThreadPoolExecutor(num_cpu)
    except (ImportError, SyntaxError, AttributeError):
        # Py2 - no async prefetching is supported
        logging.warning(
            'Asynchronous batch prefetching is not supported on Python 2. '
            'Consider upgrading to Python 3 for improved performance.')
        data = data.transform(batchify_fn)

    num_update = 0
    prefetched_iters = []
    for _ in range(min(args.num_prefetch_epoch, args.epochs)):
        prefetched_iters.append(iter(data))
    for epoch in range(args.epochs):
        if epoch + len(prefetched_iters) < args.epochs:
            prefetched_iters.append(iter(data))
        data_iter = prefetched_iters.pop(0)
        try:
            batches = ex.map(batchify_fn, data_iter)
        except NameError:  # Py 2 or batch prefetching disabled
            batches = data_iter

        # Logging variables
        log_wc = 0
        log_start_time = time.time()
        log_avg_loss = 0

        for i, batch in enumerate(batches):
            ctx = context[i % len(context)]
            batch = [array.as_in_context(ctx) for array in batch]
            with mx.autograd.record():
                loss = embedding(*batch)
            loss.backward()

            num_update += loss.shape[0]
            if len(context) == 1 or (i + 1) % len(context) == 0:
                trainer.step(batch_size=1)

            # Logging
            log_wc += loss.shape[0]
            log_avg_loss += loss.mean().as_in_context(context[0])
            if (i + 1) % args.log_interval == 0:
                # Forces waiting for computation by computing loss value
                log_avg_loss = log_avg_loss.asscalar() / args.log_interval
                wps = log_wc / (time.time() - log_start_time)
                # Due to subsampling, the overall number of batches is an upper
                # bound
                num_batches = num_tokens // args.batch_size
                if args.model.lower() == 'skipgram':
                    num_batches = (num_tokens * args.window * 2) // args.batch_size
                else:
                    num_batches = num_tokens // args.batch_size
                logging.info('[Epoch {} Batch {}/{}] loss={:.4f}, '
                             'throughput={:.2f}K wps, wc={:.2f}K'.format(
                                 epoch, i + 1, num_batches, log_avg_loss,
                                 wps / 1000, log_wc / 1000))
                log_start_time = time.time()
                log_avg_loss = 0
                log_wc = 0

            if args.eval_interval and (i + 1) % args.eval_interval == 0:
                with print_time('mx.nd.waitall()'):
                    mx.nd.waitall()
                with print_time('evaluate'):
                    evaluate(args, embedding, vocab, num_update)

    # Evaluate
    with print_time('mx.nd.waitall()'):
        mx.nd.waitall()
    with print_time('evaluate'):
        evaluate(args, embedding, vocab, num_update,
                 eval_analogy=not args.no_eval_analogy)

    # Save params
    with print_time('save parameters'):
        embedding.save_parameters(os.path.join(args.logdir, 'embedding.params'))


def evaluate(args, embedding, vocab, global_step, eval_analogy=False):
    """Evaluation helper"""
    if 'eval_tokens' not in globals():
        global eval_tokens

        eval_tokens_set = evaluation.get_tokens_in_evaluation_datasets(args)
        if not args.no_eval_analogy:
            eval_tokens_set.update(vocab.idx_to_token)

        if not args.ngram_buckets:
            # Word2Vec does not support computing vectors for OOV words
            eval_tokens_set = filter(lambda t: t in vocab, eval_tokens_set)

        eval_tokens = list(eval_tokens_set)

    if not os.path.isdir(args.logdir):
        os.makedirs(args.logdir)

    # Compute their word vectors
    context = get_context(args)
    mx.nd.waitall()

    token_embedding = nlp.embedding.TokenEmbedding(unknown_token=None,
                                                   allow_extend=True)
    token_embedding[eval_tokens] = embedding[eval_tokens]

    results = evaluation.evaluate_similarity(
        args, token_embedding, context[0], logfile=os.path.join(
            args.logdir, 'similarity.tsv'), global_step=global_step)
    if eval_analogy:
        assert not args.no_eval_analogy
        results += evaluation.evaluate_analogy(
            args, token_embedding, context[0], logfile=os.path.join(
                args.logdir, 'analogy.tsv'))

    return results


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    args_ = parse_args()
    train(args_)
