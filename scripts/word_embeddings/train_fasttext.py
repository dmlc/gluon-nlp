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

# pylint: disable=global-variable-undefined
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

import mxnet as mx
from mxnet import gluon
import numpy as np
import tqdm
from mxboard import SummaryWriter

import gluonnlp as nlp

import evaluation
from utils import clip_embeddings_gradients, get_context, print_time


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
    group.add_argument('--batch-size', type=int, default=1024,
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
        help='Size of word_context set of the ngram hash function.')
    group.add_argument('--model', type=str, default='skipgram',
                       help='SkipGram or CBOW.')
    group.add_argument('--window', type=int, default=5,
                       help='Context window size.')
    group.add_argument('--negative', type=int, default=5,
                       help='Number of negative samples.')

    # Optimization options
    group = parser.add_argument_group('Optimization arguments')
    group.add_argument('--optimizer', type=str, default='adagrad')
    group.add_argument('--lr', type=float, default=0.1)
    group.add_argument('--elementwise-clip-gradient', type=float, default=-1,
                       help='Clip embedding matrix gradients elementwise. '
                       'Disable by setting to a value <= 0.')
    group.add_argument(
        '--groupwise-clip-gradient', type=float, default=-1,
        help='Clip embedding matrix gradients '
        'such that the norm of the gradient for one embedding vector '
        'does not surpass --groupwise-clip-gradient.'
        'Disable by setting to a value <= 0.')

    # Logging
    group = parser.add_argument_group('Logging arguments')
    group.add_argument('--logdir', type=str, default='logs',
                       help='Directory to store logs.')
    group.add_argument('--eval-interval', type=int, default=10000)
    group.add_argument('--eval-analogy', action='store_true')

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

    # Skip "unknown" tokens
    with print_time('code dataset'):
        coded_dataset = [[
            vocab[token] for token in sentence if token in vocab
        ] for sentence in dataset]

    with print_time('prepare subwords'):
        subword_function = nlp.vocab.create_subword_function(
            'NGramHashes', ngrams=args.ngrams, num_subwords=args.ngram_buckets)

        # Precompute a idx to subwordidxs mapping to support fast lookup
        idx_to_subwordidxs = list(subword_function(vocab.idx_to_token))
        max_subwordidxs_len = max(len(s) for s in idx_to_subwordidxs)

        # Padded max_subwordidxs_len + 1 so each row contains at least one -1
        # element which can be found by np.argmax below.
        idx_to_subwordidxs = np.stack(
            np.pad(b, (0, max_subwordidxs_len - len(b) + 1), \
                   constant_values=-1, mode='constant')
            for b in idx_to_subwordidxs).astype(np.float32)

        logging.info('Using %s to obtain subwords. '
                     'The word with largest number of subwords '
                     'has %s subwords.', subword_function, max_subwordidxs_len)

    return coded_dataset, vocab, subword_function, idx_to_subwordidxs


def save_params(args, embedding, embedding_out):
    f, path = tempfile.mkstemp(dir=args.logdir)
    os.close(f)

    # write to temporary file; use os.replace
    embedding.collect_params().save(path)
    os.replace(path, os.path.join(args.logdir, 'embedding.params'))
    embedding_out.collect_params().save(path)
    os.replace(path, os.path.join(args.logdir, 'embedding_out.params'))


def indices_to_subwordindices_mask(indices, idx_to_subwordidxs):
    """Return array of subwordindices for indices.

    A padded numpy array and a mask is returned. The mask is used as
    indices map to varying length subwords.

    Parameters
    ----------
    indices : list of int, numpy array or mxnet NDArray
        Token indices that should be mapped to subword indices.

    Returns
    -------
    Array of subword indices.

    """
    if isinstance(indices, mx.nd.NDArray):
        indices = indices.asnumpy().astype(np.int)
    else:
        indices = np.array(indices, dtype=np.int)
    subwords = idx_to_subwordidxs[indices]
    mask = np.zeros_like(subwords)
    mask += subwords != -1
    subwords += subwords == -1
    lengths = np.argmax(subwords == -1, axis=1)

    new_length = max(np.max(lengths), 1)  # Return at least 1
    subwords = subwords[:, :new_length]
    mask = mask[:, :new_length]

    return subwords, mask


###############################################################################
# Training code
###############################################################################
def train(args):
    """Training helper."""
    coded_dataset, vocab, subword_function, idx_to_subwordidxs = \
        get_train_data(args)
    embedding = nlp.model.train.FasttextEmbeddingModel(
        num_tokens=len(vocab),
        num_subwords=len(subword_function),
        embedding_size=args.emsize,
        weight_initializer=mx.init.Uniform(scale=1 / args.emsize),
        sparse_grad=not args.no_sparse_grad,
    )
    embedding_out = nlp.model.train.SimpleEmbeddingModel(
        num_tokens=len(vocab),
        embedding_size=args.emsize,
        weight_initializer=mx.init.Uniform(scale=1 / args.emsize),
        sparse_grad=not args.no_sparse_grad,
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
        args.optimizer, learning_rate=args.lr,
        clip_gradient=args.elementwise_clip_gradient
        if args.elementwise_clip_gradient > 0 else None)
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
                subwords, subwords_mask = \
                    indices_to_subwordindices_mask(center, idx_to_subwordidxs)
            elif args.model.lower() == 'cbow':
                subwords, subwords_mask = \
                    indices_to_subwordindices_mask(word_context,
                                                   idx_to_subwordidxs)
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

            with mx.autograd.record():
                # Combine subword level embeddings with word embeddings
                if args.model.lower() == 'skipgram':
                    emb_in = embedding(center, center_mask, subwords,
                                       subwords_mask)

                    word_context_negatives = mx.nd.concat(
                        word_context, negatives, dim=1)
                    word_context_negatives_mask = mx.nd.concat(
                        word_context_mask, mx.nd.ones_like(negatives), dim=1)

                    emb_out = embedding_out(word_context_negatives,
                                            word_context_negatives_mask)

                    # Compute loss
                    pred = mx.nd.batch_dot(
                        emb_in.expand_dims(1), emb_out.swapaxes(1, 2))
                    pred = pred.squeeze() * word_context_negatives_mask
                    label = mx.nd.concat(word_context_mask,
                                         mx.nd.zeros_like(negatives), dim=1)

                elif args.model.lower() == 'cbow':
                    emb_in = embedding(word_context, word_context_mask,
                                       subwords, subwords_mask).sum(axis=-2)

                    center_negatives = mx.nd.concat(center, negatives, dim=1)
                    center_negatives_mask = mx.nd.concat(
                        center_mask, mx.nd.ones_like(negatives), dim=1)

                    emb_out = embedding_out(center_negatives,
                                            center_negatives_mask)

                    # Compute loss
                    pred = mx.nd.batch_dot(
                        emb_in.expand_dims(1), emb_out.expand_dims(2))
                    pred = pred.reshape((-1, 1 + args.negative))
                    label = mx.nd.concat(
                        mx.nd.ones_like(center), mx.nd.zeros_like(negatives),
                        dim=1)

                loss = loss_function(pred, label)

            loss.backward()

            # Normalize gradients
            if args.groupwise_clip_gradient > 0:
                clip_embeddings_gradients(trainer._params,
                                          args.groupwise_clip_gradient)

            if args.optimizer != 'adagrad':
                trainer.set_learning_rate(args.lr * (1 - progress))
            trainer.step(batch_size=1)

            # Logging
            if i % args.eval_interval == 0:
                with print_time('mx.nd.waitall()'):
                    mx.nd.waitall()

                log(args, sw, embedding, embedding_out, loss, num_update,
                    vocab, subword_function)

        # Log at the end of every epoch
        with print_time('mx.nd.waitall()'):
            mx.nd.waitall()
        log(args, sw, embedding, embedding_out, loss, num_update, vocab,
            subword_function)

        # Save params at end of epoch
        save_params(args, embedding, embedding_out)

    sw.close()


def log(args, sw, embedding, embedding_out, loss, num_update, vocab,
        subword_function):
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

    results = evaluate(args, embedding, vocab, subword_function)
    for result in results:
        tag = result['dataset_name'] + '_' + str(result['dataset_kwargs'])
        if result['task'] == 'analogy':
            sw.add_scalar(tag=tag, value=float(result['accuracy']),
                          global_step=num_update)
        if result['task'] == 'similarity':
            sw.add_scalar(tag=tag, value=float(result['spearmanr']),
                          global_step=num_update)

    sw.flush()


def evaluate(args, embedding, vocab, subword_function):
    """Evaluation helper"""
    if 'eval_tokens' not in globals():
        global eval_tokens

        eval_tokens_set = evaluation.get_tokens_in_evaluation_datasets(args)
        if args.eval_analogy:
            eval_tokens_set.update(vocab.idx_to_token)
        eval_tokens = list(eval_tokens_set)

    # Compute their word vectors
    context = get_context(args)
    idx_to_token = eval_tokens
    mx.nd.waitall()
    token_embedding = embedding.to_token_embedding(
        idx_to_token, vocab.token_to_idx, subword_function, ctx=context[0])

    results = evaluation.evaluate_similarity(
        args, token_embedding, context[0], logfile=os.path.join(
            args.logdir, 'similarity.tsv'))
    if args.eval_analogy:
        results += evaluation.evaluate_analogy(
            args, token_embedding, context[0], logfile=os.path.join(
                args.logdir, 'analogy.tsv'))

    return results


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    args_ = parse_args()
    train(args_)
