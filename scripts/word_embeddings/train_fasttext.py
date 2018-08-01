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
"""Fasttext embedding model
===========================

This example shows how to train a FastText embedding model on Text8 with the
Gluon NLP Toolkit.

The FastText embedding model was introduced by

- Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word
  vectors with subword information. TACL, 5(), 135–146.

When setting --ngram-buckets to 0, a Word2Vec embedding model is trained. The
Word2Vec embedding model was introduced by

- Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation
  of word representations in vector space. ICLR Workshop , 2013

"""
# Set a few mxnet specific environment variables
import os
# Workaround for https://github.com/apache/incubator-mxnet/issues/11314
os.environ['MXNET_FORCE_ADDTAKEGRAD'] = '1'

import argparse
import functools
import itertools
import logging
import random
import sys
import tempfile
import time
import warnings

import mxnet as mx
import numpy as np
import gluonnlp as nlp
from gluonnlp.base import numba_njit, numba_prange

import evaluation
from candidate_sampler import remove_accidental_hits
from utils import get_context, print_time, prune_sentences


###############################################################################
# Utils
###############################################################################
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Word embedding training with Gluon.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Computation options
    group = parser.add_argument_group('Computation arguments')
    group.add_argument('--batch-size', type=int, default=2048,
                       help='Batch size for training.')
    group.add_argument('--epochs', type=int, default=5, help='Epoch limit')
    group.add_argument('--gpu', type=int, nargs='+',
                       help=('Number (index) of GPU to run on, e.g. 0. '
                             'If not specified, uses CPU.'))
    group.add_argument('--no-hybridize', action='store_true',
                       help='Disable hybridization of gluon HybridBlocks.')
    group.add_argument(
        '--no-static-alloc', action='store_true',
        help='Disable static memory allocation for HybridBlocks.')
    group.add_argument('--no-sparse-grad', action='store_true',
                       help='Disable sparse gradient support.')

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
    group.add_argument('--negative', type=int, default=5,
                       help='Number of negative samples '
                       'per source-context word pair.')
    group.add_argument('--frequent-token-subsampling', type=float,
                       default=1E-4,
                       help='Frequent token subsampling constant.')

    # Optimization options
    group = parser.add_argument_group('Optimization arguments')
    group.add_argument('--optimizer', type=str, default='adagrad')
    group.add_argument('--lr', type=float, default=0.05)
    group.add_argument('--optimizer-subwords', type=str, default='adagrad')
    group.add_argument('--lr-subwords', type=float, default=0.5)

    # Logging
    group = parser.add_argument_group('Logging arguments')
    group.add_argument('--logdir', type=str, default='logs',
                       help='Directory to store logs.')
    group.add_argument('--log-interval', type=int, default=100)
    group.add_argument('--eval-interval', type=int,
                       help='Evaluate every --eval-interval iterations '
                       'in addition to at the end of every epoch.')
    group.add_argument('--no-eval-analogy', action='store_true',
                       help='Don\'t evaluate on the analogy task.')

    # Evaluation options
    evaluation.add_parameters(parser)

    args = parser.parse_args()
    evaluation.validate_args(args)
    return args


def get_train_data(args):
    """Helper function to get training data."""
    with print_time('load training dataset'):
        dataset = nlp.data.Text8(segment='train')

    with print_time('count tokens'):
        counter = nlp.data.count_tokens(itertools.chain.from_iterable(dataset))

    vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                      bos_token=None, eos_token=None, min_freq=5)

    idx_to_counts = np.array([counter[w] for w in vocab.idx_to_token])
    negatives_weights = idx_to_counts**0.75
    negatives_sampler = nlp.data.UnigramCandidateSampler(
        weights=mx.nd.array(negatives_weights))

    # Skip "unknown" tokens
    with print_time('code dataset'):
        coded_dataset = [[
            vocab[token] for token in sentence if token in vocab
        ] for sentence in dataset]
        coded_dataset = [
            sentence for sentence in coded_dataset if len(sentence)
        ]

    with print_time('prune frequent words from sentences'):
        f = idx_to_counts / np.sum(idx_to_counts)
        idx_to_pdiscard = 1 - np.sqrt(args.frequent_token_subsampling / f)

        prune_sentences_ = functools.partial(prune_sentences,
                                             idx_to_pdiscard=idx_to_pdiscard)
        coded_dataset = list(map(prune_sentences_, coded_dataset))

    if args.ngram_buckets:  # Fasttext model
        with print_time('prepare subwords'):
            subword_function = nlp.vocab.create_subword_function(
                'NGramHashes', ngrams=args.ngrams,
                num_subwords=args.ngram_buckets)

            # Store subword indices for all words in vocabulary
            idx_to_subwordidxs = list(subword_function(vocab.idx_to_token))
            get_subwords_masks = get_subwords_masks_factory(idx_to_subwordidxs)
            max_subwordidxs_len = max(len(s) for s in idx_to_subwordidxs)
            if max_subwordidxs_len > 500:
                warnings.warn(
                    'The word with largest number of subwords '
                    'has {} subwords, suggesting there are '
                    'some noisy words in your vocabulary. '
                    'You should filter out very long words '
                    'to avoid memory issues.'.format(max_subwordidxs_len))

        return (coded_dataset, negatives_sampler, vocab, subword_function,
                get_subwords_masks)
    else:
        return coded_dataset, negatives_sampler, vocab


def get_subwords_masks_factory(idx_to_subwordidxs):
    idx_to_subwordidxs = [
        np.array(i, dtype=np.int_) for i in idx_to_subwordidxs
    ]

    def get_subwords_masks(indices):
        subwords = [idx_to_subwordidxs[i] for i in indices]
        return _get_subwords_masks(subwords)

    return get_subwords_masks


@numba_njit
def _get_subwords_masks(subwords):
    lengths = np.array([len(s) for s in subwords])
    length = np.max(lengths)
    subwords_arr = np.zeros((len(subwords), length))
    mask = np.zeros((len(subwords), length))
    for i in numba_prange(len(subwords)):
        s = subwords[i]
        subwords_arr[i, :len(s)] = s
        mask[i, :len(s)] = 1
    return subwords_arr, mask


def save_params(args, embedding, embedding_out):
    f, path = tempfile.mkstemp(dir=args.logdir)
    os.close(f)

    # write to temporary file; use os.replace
    embedding.collect_params().save(path)
    os.replace(path, os.path.join(args.logdir, 'embedding.params'))
    embedding_out.collect_params().save(path)
    os.replace(path, os.path.join(args.logdir, 'embedding_out.params'))


###############################################################################
# Training code
###############################################################################
def train(args):
    """Training helper."""
    if args.ngram_buckets:  # Fasttext model
        coded_dataset, negatives_sampler, vocab, subword_function, \
            get_subwords_masks = get_train_data(args)
        embedding = nlp.model.train.FasttextEmbeddingModel(
            token_to_idx=vocab.token_to_idx,
            subword_function=subword_function,
            embedding_size=args.emsize,
            weight_initializer=mx.init.Uniform(scale=1 / args.emsize),
            sparse_grad=not args.no_sparse_grad,
        )
    else:
        coded_dataset, negatives_sampler, vocab = get_train_data(args)
        embedding = nlp.model.train.SimpleEmbeddingModel(
            token_to_idx=vocab.token_to_idx,
            embedding_size=args.emsize,
            weight_initializer=mx.init.Uniform(scale=1 / args.emsize),
            sparse_grad=not args.no_sparse_grad,
        )
    embedding_out = nlp.model.train.SimpleEmbeddingModel(
        token_to_idx=vocab.token_to_idx,
        embedding_size=args.emsize,
        weight_initializer=mx.init.Zero(),
        sparse_grad=not args.no_sparse_grad,
    )
    loss_function = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()

    context = get_context(args)
    embedding.initialize(ctx=context)
    embedding_out.initialize(ctx=context)
    if not args.no_hybridize:
        embedding.hybridize(static_alloc=not args.no_static_alloc)
        embedding_out.hybridize(static_alloc=not args.no_static_alloc)

    optimizer_kwargs = dict(learning_rate=args.lr)
    params = list(embedding.embedding.collect_params().values()) + \
        list(embedding_out.collect_params().values())
    trainer = mx.gluon.Trainer(params, args.optimizer, optimizer_kwargs)

    if args.ngram_buckets:  # Fasttext model
        optimizer_subwords_kwargs = dict(learning_rate=args.lr_subwords)
        params_subwords = list(
            embedding.subword_embedding.collect_params().values())
        trainer_subwords = mx.gluon.Trainer(params_subwords,
                                            args.optimizer_subwords,
                                            optimizer_subwords_kwargs)

    num_update = 0
    for epoch in range(args.epochs):
        random.shuffle(coded_dataset)
        context_sampler = nlp.data.ContextSampler(coded=coded_dataset,
                                                  batch_size=args.batch_size,
                                                  window=args.window)
        num_batches = len(context_sampler)

        # Logging variables
        log_wc = 0
        log_start_time = time.time()
        log_avg_loss = 0

        for i, batch in enumerate(context_sampler):
            progress = (epoch * num_batches + i) / (args.epochs * num_batches)
            (center, word_context, word_context_mask) = batch
            negatives_shape = (word_context.shape[0],
                               word_context.shape[1] * args.negative)
            negatives, negatives_mask = remove_accidental_hits(
                negatives_sampler(negatives_shape), word_context,
                word_context_mask)

            if args.ngram_buckets:  # Fasttext model
                if args.model.lower() == 'skipgram':
                    unique, inverse_unique_indices = np.unique(
                        center.asnumpy(), return_inverse=True)
                    inverse_unique_indices = mx.nd.array(
                        inverse_unique_indices, ctx=context[0])
                    subwords, subwords_mask = get_subwords_masks(
                        unique.astype(int))
                    subwords = mx.nd.array(subwords, ctx=context[0])
                    subwords_mask = mx.nd.array(subwords_mask, ctx=context[0])
                elif args.model.lower() == 'cbow':
                    unique, inverse_unique_indices = np.unique(
                        word_context.asnumpy(), return_inverse=True)
                    inverse_unique_indices = mx.nd.array(
                        inverse_unique_indices, ctx=context[0])
                    subwords, subwords_mask = get_subwords_masks(
                        unique.astype(int))
                    subwords = mx.nd.array(subwords, ctx=context[0])
                    subwords_mask = mx.nd.array(subwords_mask, ctx=context[0])
                else:
                    logging.error('Unsupported model %s.', args.model)
                    sys.exit(1)

            num_update += len(center)

            # To GPU
            center = center.as_in_context(context[0])
            word_context = word_context.as_in_context(context[0])
            word_context_mask = word_context_mask.as_in_context(context[0])
            negatives = negatives.as_in_context(context[0])
            negatives_mask = negatives_mask.as_in_context(context[0])

            with mx.autograd.record():
                # Combine subword level embeddings with word embeddings
                if args.model.lower() == 'skipgram':
                    if args.ngram_buckets:
                        emb_in = embedding(center, subwords,
                                           subwordsmask=subwords_mask,
                                           words_to_unique_subwords_indices=
                                           inverse_unique_indices)
                    else:
                        emb_in = embedding(center)

                    with mx.autograd.pause():
                        word_context_negatives = mx.nd.concat(
                            word_context, negatives, dim=1)
                        word_context_negatives_mask = mx.nd.concat(
                            word_context_mask, negatives_mask, dim=1)

                    emb_out = embedding_out(word_context_negatives,
                                            word_context_negatives_mask)

                    # Compute loss
                    pred = mx.nd.batch_dot(emb_in, emb_out.swapaxes(1, 2))
                    pred = pred.squeeze() * word_context_negatives_mask
                    label = mx.nd.concat(word_context_mask,
                                         mx.nd.zeros_like(negatives), dim=1)

                elif args.model.lower() == 'cbow':
                    word_context = word_context.reshape((-3, 1))
                    word_context_mask = word_context_mask.reshape((-3, 1))
                    if args.ngram_buckets:
                        emb_in = embedding(word_context, subwords,
                                           word_context_mask, subwords_mask,
                                           inverse_unique_indices)
                    else:
                        emb_in = embedding(word_context, word_context_mask)

                    with mx.autograd.pause():
                        center = center.tile(args.window * 2).reshape((-1, 1))
                        negatives = negatives.reshape((-1, args.negative))

                        center_negatives = mx.nd.concat(
                            center, negatives, dim=1)
                        center_negatives_mask = mx.nd.concat(
                            mx.nd.ones_like(center), negatives_mask, dim=1)

                    emb_out = embedding_out(center_negatives,
                                            center_negatives_mask)

                    # Compute loss
                    pred = mx.nd.batch_dot(emb_in, emb_out.swapaxes(1, 2))
                    pred = pred.squeeze() * word_context_mask
                    label = mx.nd.concat(
                        mx.nd.ones_like(word_context),
                        mx.nd.zeros_like(negatives), dim=1)

                loss = loss_function(pred, label)

            loss.backward()

            if args.optimizer.lower() != 'adagrad':
                trainer.set_learning_rate(
                    max(0.0001, args.lr * (1 - progress)))

            if (args.optimizer_subwords.lower() != 'adagrad'
                    and args.ngram_buckets):
                trainer_subwords.set_learning_rate(
                    max(0.0001, args.lr_subwords * (1 - progress)))

            trainer.step(batch_size=1)
            if args.ngram_buckets:
                trainer_subwords.step(batch_size=1)

            # Logging
            log_wc += loss.shape[0]
            log_avg_loss += loss.mean()
            if (i + 1) % args.log_interval == 0:
                # Forces waiting for computation by computing loss value
                log_avg_loss = log_avg_loss.asscalar() / args.log_interval
                wps = log_wc / (time.time() - log_start_time)
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
        save_params(args, embedding, embedding_out)


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

    os.makedirs(args.logdir, exist_ok=True)

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
