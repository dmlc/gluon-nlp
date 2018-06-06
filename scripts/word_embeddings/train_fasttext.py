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
"""Fasttext embedding model
===========================

This example shows how to train a FastText embedding model on Text8 with the
Gluon NLP Toolkit.

The FastText embedding model was introduced by

- Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word
  vectors with subword information. TACL, 5(), 135–146.

"""
import sys
import argparse
import itertools
import logging
import os
import random
import tempfile
import time
from contextlib import contextmanager

import mxnet as mx
from mxnet import gluon
import numpy as np
import tqdm
from mxboard import SummaryWriter
from scipy import stats

import gluonnlp as nlp


###############################################################################
# Utils
###############################################################################
@contextmanager
def print_time(task):
    start_time = time.time()
    logging.info('Starting to %s', task)
    yield
    logging.info('Finished to {} in {:.2f} seconds'.format(
        task,
        time.time() - start_time))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Word embedding training with Gluon.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Computation options
    group = parser.add_argument_group('Computation arguments')
    group.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size for training.')
    group.add_argument('--epochs', type=int, default=5, help='Epoch limit')
    group.add_argument('--gpu', type=int, nargs='+',
                       help=('Number (index) of GPU to run on, e.g. 0. '
                             'If not specified, uses CPU.'))
    group.add_argument('--no-deduplicate-subwords', action='store_true',
                       help='Disable deduplication '
                       'of words for subword computation in batch.')
    group.add_argument('--no-hybridize', action='store_true',
                       help='Disable hybridization of gluon HybridBlocks.')
    group.add_argument(
        '--no-static-alloc', action='store_true',
        help='Disable static memory allocation for HybridBlocks.')
    group.add_argument('--sparse-grad', action='store_true',
                       help='Enable sparse gradient support.')

    # Model
    group = parser.add_argument_group('Model arguments')
    group.add_argument('--emsize', type=int, default=300,
                       help='Size of embedding vectors.')
    group.add_argument('--ngrams', type=int, nargs='+', default=[3, 4, 5, 6])
    group.add_argument(
        '--ngram-buckets', type=int, default=2000000,
        help='Size of word_context set of the ngram hash function.')
    group.add_argument('--model', type=str, default='skipgram',
                       help='SkipGram or CBOW.')
    group.add_argument('--window', type=int, default=5,
                       help='Context window size.')
    group.add_argument('--negative', type=int, default=5,
                       help='Number of negative samples.')

    # Optimization options
    group = parser.add_argument_group('Optimization arguments')
    group.add_argument('--optimizer', type=str, default='sgd')
    group.add_argument('--lr', type=float, default=0.1)

    # Logging
    group = parser.add_argument_group('Logging arguments')
    group.add_argument('--logdir', type=str, default='logs',
                       help='Directory to store logs.')
    group.add_argument('--eval-interval', type=int, default=10000)

    args = parser.parse_args()
    return args


def get_train_data(args):
    """Helper function to get training data."""
    with print_time('load training dataset'):
        dataset = nlp.data.Text8(segment='train')

    with print_time('count tokens'):
        counter = nlp.data.count_tokens(itertools.chain.from_iterable(dataset))

    vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                      bos_token=None, eos_token=None, min_freq=5)

    # Skip "unknown" tokens
    with print_time('code dataset'):
        coded_dataset = [[
            vocab[token] for token in sentence if token in vocab
        ] for sentence in dataset]

    with print_time('prepare subwords'):
        subword_function = nlp.vocab.create('NGramHashes', ngrams=args.ngrams,
                                            num_subwords=args.ngram_buckets)
        subword_vocab = nlp.SubwordVocab(idx_to_token=vocab.idx_to_token,
                                         subword_function=subword_function)

    return coded_dataset, vocab, subword_vocab


def get_context(args):
    if args.gpu is None or args.gpu == '':
        context = [mx.cpu()]
    else:
        context = [mx.gpu(int(i)) for i in args.gpu]
    return context


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
    coded_dataset, vocab, subword_vocab = get_train_data(args)
    if args.no_deduplicate_subwords:
        FasttextEmbeddingModel = nlp.model.FasttextEmbeddingModel
    else:
        FasttextEmbeddingModel = nlp.model.DeduplicatedFasttextEmbeddingModel

    embedding = FasttextEmbeddingModel(
        num_tokens=len(vocab),
        num_subwords=len(subword_vocab),
        embedding_size=args.emsize,
        weight_initializer=mx.init.Uniform(scale=1 / args.emsize),
        sparse_grad=args.sparse_grad,
    )
    embedding_out = nlp.model.SimpleEmbeddingModel(
        num_tokens=len(vocab),
        embedding_size=args.emsize,
        weight_initializer=mx.init.Uniform(scale=1 / args.emsize),
        sparse_grad=args.sparse_grad,
    )
    loss_function = gluon.loss.SigmoidBinaryCrossEntropyLoss()

    context = get_context(args)
    embedding.initialize(ctx=context)
    embedding_out.initialize(ctx=context)
    if not args.no_hybridize:
        embedding.hybridize(static_alloc=not args.no_static_alloc)
        embedding_out.hybridize(static_alloc=not args.no_static_alloc)

    params = list(embedding.collect_params().values()) + \
        list(embedding_out.collect_params().values())
    optimizer = mx.optimizer.Optimizer.create_optimizer(
        args.optimizer, learning_rate=args.lr)
    trainer = gluon.Trainer(params, optimizer)

    # Logging writer
    sw = SummaryWriter(logdir=args.logdir)

    num_update = 0
    for epoch in range(args.epochs):
        random.shuffle(coded_dataset)
        context_sampler = nlp.data.ContextSampler(coded=coded_dataset,
                                                  batch_size=args.batch_size,
                                                  window=args.window)
        negatives_sampler = nlp.data.NegativeSampler(
            num_samples=context_sampler.num_samples,
            batch_size=args.batch_size, vocab=vocab, negative=args.negative)
        num_batches = len(context_sampler)

        for i, (batch, negatives) in tqdm.tqdm(
                enumerate(zip(context_sampler,
                              negatives_sampler)), total=num_batches,
                ascii=True, smoothing=1, dynamic_ncols=True):
            progress = (epoch * num_batches + i) / (args.epochs * num_batches)
            (center, word_context, word_context_mask) = batch
            if args.model.lower() == 'skipgram':
                if args.no_deduplicate_subwords:
                    subwords, subwords_mask = \
                        subword_vocab.indices_to_subwordindices_mask(center)
                else:
                    unique, inverse_unique_indices = np.unique(
                        center, return_inverse=True)
                    subwords, subwords_mask = \
                        subword_vocab.indices_to_subwordindices_mask(unique)
            elif args.model.lower() == 'cbow':
                if args.no_deduplicate_subwords:
                    subwords, subwords_mask = \
                        subword_vocab.indices_to_subwordindices_mask(word_context)
                else:
                    unique, inverse_unique_indices = np.unique(
                        word_context, return_inverse=True)
                    inverse_unique_indices = inverse_unique_indices.reshape(
                        word_context.shape)
                    subwords, subwords_mask = \
                        subword_vocab.indices_to_subwordindices_mask(unique)
            else:
                logging.error('Unsupported model %s.', args.model)
                sys.exit(1)
            num_update += len(center)

            # To GPU
            center = mx.nd.array(center, ctx=context[0])
            center_mask = mx.nd.ones((center.shape[0], ), ctx=center.context)
            subwords = mx.nd.array(subwords, ctx=context[0])
            subwords_mask = mx.nd.array(subwords_mask,
                                        dtype=np.float32).as_in_context(
                                            context[0])
            word_context = mx.nd.array(word_context, ctx=context[0])
            word_context_mask = mx.nd.array(word_context_mask, ctx=context[0])
            negatives = mx.nd.array(negatives, ctx=context[0])

            if not args.no_deduplicate_subwords:
                inverse_unique_indices = mx.nd.array(inverse_unique_indices,
                                                     ctx=context[0])

            with mx.autograd.record():
                # Combine subword level embeddings with word embeddings
                if args.model.lower() == 'skipgram':
                    if args.no_deduplicate_subwords:
                        emb_in = embedding(center, center_mask, subwords,
                                           subwords_mask)
                    else:
                        emb_in = embedding(center, center_mask, subwords,
                                           subwords_mask,
                                           inverse_unique_indices)
                    emb_out_pos = embedding_out(word_context,
                                                word_context_mask)
                    emb_out_neg = embedding_out(negatives,
                                                mx.nd.ones_like(negatives))

                    # Compute loss
                    pred_pos = mx.nd.batch_dot(
                        emb_in.expand_dims(1), emb_out_pos.swapaxes(1, 2))
                    pred_pos = pred_pos.squeeze() * word_context_mask
                    pred_neg = mx.nd.batch_dot(
                        emb_in.expand_dims(1), emb_out_neg.swapaxes(1, 2))
                    pred_neg = pred_neg.squeeze()

                    pred = mx.nd.concat(pred_pos, pred_neg, dim=1)
                    label = mx.nd.concat(word_context_mask,
                                         mx.nd.zeros_like(pred_neg), dim=1)
                elif args.model.lower() == 'cbow':
                    if args.no_deduplicate_subwords:
                        emb_in = embedding(word_context, word_context_mask,
                                           subwords,
                                           subwords_mask).sum(axis=-2)
                    else:
                        emb_in = embedding(
                            word_context, word_context_mask, subwords,
                            subwords_mask,
                            inverse_unique_indices).sum(axis=-2)
                    emb_out_pos = embedding_out(center, center_mask)
                    emb_out_neg = embedding_out(negatives,
                                                mx.nd.ones_like(negatives))

                    # Compute loss
                    pred_pos = mx.nd.batch_dot(
                        emb_in.expand_dims(1), emb_out_pos.expand_dims(2))
                    pred_pos = pred_pos.reshape((-1, 1))
                    pred_neg = mx.nd.batch_dot(
                        emb_in.expand_dims(1), emb_out_neg.swapaxes(1, 2))
                    pred_neg = pred_neg.reshape((-1, args.negative))

                    pred = mx.nd.concat(pred_pos, pred_neg, dim=1)
                    label = mx.nd.concat(
                        mx.nd.ones_like(pred_pos), mx.nd.zeros_like(pred_neg),
                        dim=1)
                loss = loss_function(pred, label)

            loss.backward()

            trainer.set_learning_rate(args.lr * (1 - progress))
            trainer.step(batch_size=1)

            # Logging
            if i % args.eval_interval == 0:
                with print_time('mx.nd.waitall()'):
                    mx.nd.waitall()

                log(args, sw, embedding, embedding_out, loss, num_update,
                    vocab, subword_vocab)

        # Log at the end of every epoch
        with print_time('mx.nd.waitall()'):
            mx.nd.waitall()
        log(args, sw, embedding, embedding_out, loss, num_update, vocab,
            subword_vocab)

        # Save params at end of epoch
        save_params(args, embedding, embedding_out)

    sw.close()


def log(args, sw, embedding, embedding_out, loss, num_update, vocab,
        subword_vocab):
    """Logging helper"""
    context = get_context(args)

    # Word embeddings
    embedding_norm = embedding.embedding.weight.data(
        ctx=context[0]).as_in_context(
            mx.cpu()).tostype('default').norm(axis=1)
    sw.add_histogram(tag='embedding_norm', values=embedding_norm,
                     global_step=num_update, bins=200)
    if embedding.embedding.weight.grad(ctx=context[0]).stype == 'row_sparse':
        embedding_grad_norm = embedding.embedding.weight.grad(
            ctx=context[0]).data.as_in_context(
                mx.cpu()).tostype('default').norm(axis=1)
        sw.add_histogram(tag='embedding_grad_norm', values=embedding_grad_norm,
                         global_step=num_update, bins=200)

    # Subword embeddings
    embedding_norm = embedding.subword_embedding.embedding.weight.data(
        ctx=context[0]).as_in_context(
            mx.cpu()).tostype('default').norm(axis=1)
    sw.add_histogram(tag='subword_embedding_norm', values=embedding_norm,
                     global_step=num_update, bins=200)
    if embedding.subword_embedding.embedding.weight.grad(
            ctx=context[0]).stype == 'row_sparse':
        embedding_grad_norm = \
            embedding.subword_embedding.embedding.weight.grad(
                ctx=context[0]).data.as_in_context(
                    mx.cpu()).tostype('default').norm(axis=1)
        sw.add_histogram(tag='subword_embedding_grad_norm',
                         values=embedding_grad_norm, global_step=num_update,
                         bins=200)

    embedding_out_norm = embedding_out.embedding.weight.data(
        ctx=context[0]).as_in_context(
            mx.cpu()).tostype('default').norm(axis=1)
    sw.add_histogram(tag='embedding_out_norm', values=embedding_out_norm,
                     global_step=num_update, bins=200)
    if embedding_out.embedding.weight.grad(
            ctx=context[0]).stype == 'row_sparse':
        embedding_out_grad_norm = embedding_out.embedding.weight.grad(
            ctx=context[0]).data.as_in_context(
                mx.cpu()).tostype('default').norm(axis=1)
        sw.add_histogram(tag='embedding_out_grad_norm',
                         values=embedding_out_grad_norm,
                         global_step=num_update, bins=200)

    if not isinstance(loss, int):
        sw.add_scalar(tag='loss', value=loss.mean().asscalar(),
                      global_step=num_update)

    eval_dict = evaluate(args, embedding, vocab, subword_vocab)
    for k, v in eval_dict.items():
        sw.add_scalar(tag=k, value=float(v), global_step=num_update)

    sw.flush()


def evaluate(args, embedding, vocab, subword_vocab):
    """Evaluation helper"""
    # Collect all words in the evaluation datasets
    counter = nlp.data.utils.Counter()
    for dataset_name in \
            nlp.data.word_embedding_evaluation.word_similarity_datasets:
        dataset = nlp.data.create(dataset_name)
        counter.update(
            itertools.chain.from_iterable((d[0], d[1]) for d in dataset))

    # Compute their word vectors
    context = get_context(args)
    idx_to_token = list(counter)
    token_to_idx = {token: idx for idx, token in enumerate(counter)}
    mx.nd.waitall()
    token_embedding = embedding.to_token_embedding(
        idx_to_token, vocab, subword_vocab, ctx=context[0])

    # Initialize evaluator
    evaluator = nlp.embedding.evaluation.WordEmbeddingSimilarity(
        idx_to_vec=token_embedding.idx_to_vec,
        similarity_function='CosineSimilarity')
    evaluator.initialize(ctx=context[0])
    if not args.no_hybridize:
        evaluator.hybridize()

    # Evaluate all datasets
    eval_dict = {}
    for dataset_name in \
            nlp.data.word_embedding_evaluation.word_similarity_datasets:
        dataset = nlp.data.create(dataset_name)
        dataset_coded = [[token_to_idx[d[0]], token_to_idx[d[1]], d[2]]
                         for d in dataset]
        words1, words2, scores = zip(*dataset_coded)
        pred_similarity = evaluator(
            mx.nd.array(words1, ctx=context[0]),
            mx.nd.array(words2, ctx=context[0]))
        sr = stats.spearmanr(pred_similarity.asnumpy(), np.array(scores))
        logging.info('Spearman rank correlation on %s: %s',
                     dataset.__class__.__name__, sr.correlation)
        eval_dict[dataset_name + '-sr'] = sr.correlation
    return eval_dict


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    args_ = parse_args()
    train(args_)
