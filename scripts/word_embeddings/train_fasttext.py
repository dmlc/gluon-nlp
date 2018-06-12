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
import argparse
import functools
import itertools
import logging
import os
import random
import sys
import tempfile

import mxnet as mx
import numpy as np
import tqdm

import evaluation
import gluonnlp as nlp
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
        '--ngram-buckets', type=int, default=500000,
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
    group.add_argument('--lr', type=float, default=0.05)
    group.add_argument('--optimizer-subwords', type=str, default='adagrad')
    group.add_argument('--lr-subwords', type=float, default=0.01)

    # Logging
    group = parser.add_argument_group('Logging arguments')
    group.add_argument('--logdir', type=str, default='logs',
                       help='Directory to store logs.')
    group.add_argument('--eval-interval', type=int, default=50000,
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

    idx_to_counts = mx.nd.array([counter[w] for w in vocab.idx_to_token])
    negatives_weights = idx_to_counts**0.75
    negatives_sampler = nlp.data.UnigramCandidateSampler(
        weights=negatives_weights)

    # Skip "unknown" tokens
    with print_time('code dataset'):
        coded_dataset = [[
            vocab[token] for token in sentence if token in vocab
        ] for sentence in dataset]

    with print_time('prune frequent words from sentences'):
        frequent_tokens_subsampling_constant = 1e-3
        f = idx_to_counts / mx.nd.sum(idx_to_counts)
        idx_to_pdiscard = (
            mx.nd.sqrt(frequent_tokens_subsampling_constant / f) +
            frequent_tokens_subsampling_constant / f).asnumpy()

        prune_sentences_ = functools.partial(prune_sentences,
                                             idx_to_pdiscard=idx_to_pdiscard)
        coded_dataset = list(map(prune_sentences_, coded_dataset))

    with print_time('prepare subwords'):
        subword_function = nlp.vocab.create_subword_function(
            'NGramHashes', ngrams=args.ngrams, num_subwords=args.ngram_buckets)

        # Precompute a idx to subwordidxs mapping to support fast lookup
        idx_to_subwordidxs = list(subword_function(vocab.idx_to_token))
        max_subwordidxs_len = max(len(s) for s in idx_to_subwordidxs)

        # Padded max_subwordidxs_len + 1 so each row contains at least one -1
        # element which can be found by np.argmax below.
        idx_to_subwordidxs = np.stack(
            np.pad(b.asnumpy(), (0, max_subwordidxs_len - len(b) + 1), \
                   constant_values=-1, mode='constant')
            for b in idx_to_subwordidxs).astype(np.float32)
        idx_to_subwordidxs = mx.nd.array(idx_to_subwordidxs)

        logging.info('Using %s to obtain subwords. '
                     'The word with largest number of subwords '
                     'has %s subwords.', subword_function, max_subwordidxs_len)

    return (coded_dataset, negatives_sampler, vocab, subword_function,
            idx_to_subwordidxs)


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
    if not isinstance(indices, mx.nd.NDArray):
        indices = mx.nd.array(indices)
    subwords = idx_to_subwordidxs[indices]
    mask = mx.nd.zeros_like(subwords)
    mask += subwords != -1
    lengths = mx.nd.argmax(subwords == -1, axis=1)
    subwords += subwords == -1

    new_length = int(max(mx.nd.max(lengths).asscalar(), 1))
    subwords = subwords[:, :new_length]
    mask = mask[:, :new_length]

    return subwords, mask


###############################################################################
# Training code
###############################################################################
def train(args):
    """Training helper."""
    coded_dataset, negatives_sampler, vocab, subword_function, \
        idx_to_subwordidxs = get_train_data(args)
    embedding = nlp.model.train.FasttextEmbeddingModel(
        token_to_idx=vocab.token_to_idx,
        subword_function=subword_function,
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

    optimizer_subwords_kwargs = dict(learning_rate=args.lr_subwords)
    params_subwords = list(
        embedding.subword_embedding.collect_params().values())
    trainer_subwords = mx.gluon.Trainer(
        params_subwords, args.optimizer_subwords, optimizer_subwords_kwargs)

    num_update = 0
    for epoch in range(args.epochs):
        random.shuffle(coded_dataset)
        context_sampler = nlp.data.ContextSampler(coded=coded_dataset,
                                                  batch_size=args.batch_size,
                                                  window=args.window)
        num_batches = len(context_sampler)

        for i, batch in tqdm.tqdm(
                enumerate(context_sampler), total=num_batches, ascii=True,
                smoothing=1, dynamic_ncols=True):
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
            mx.nd.waitall()  # waitall() until mxnet #11041 is merged
            center = center.as_in_context(context[0])
            center_mask = mx.nd.ones((center.shape[0], ), ctx=center.context)
            subwords = subwords.as_in_context(context[0])
            subwords_mask = subwords_mask.astype(np.float32).as_in_context(
                context[0])
            word_context = word_context.as_in_context(context[0])
            word_context_mask = word_context_mask.as_in_context(context[0])
            negatives = negatives_sampler(word_context.shape + (args.negative, )) \
                .reshape((word_context.shape[0],
                          word_context.shape[1] * args.negative)) \
                .as_in_context(context[0])

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

                    center_negatives = mx.nd.concat(
                        center.expand_dims(1), negatives, dim=1)
                    center_negatives_mask = mx.nd.concat(
                        center_mask.expand_dims(1), mx.nd.ones_like(negatives),
                        dim=1)

                    emb_out = embedding_out(center_negatives,
                                            center_negatives_mask)

                    # Compute loss
                    pred = mx.nd.batch_dot(
                        emb_in.expand_dims(1), emb_out.swapaxes(1, 2))
                    pred = pred.reshape((-1, 1 + args.negative))
                    label = mx.nd.concat(
                        mx.nd.ones_like(center).expand_dims(1),
                        mx.nd.zeros_like(negatives), dim=1)

                loss = loss_function(pred, label)

            loss.backward()

            if args.optimizer.lower() not in ['adagrad', 'adam']:
                trainer.set_learning_rate(args.lr * (1 - progress))
            if args.optimizer_subwords.lower() not in ['adagrad', 'adam']:
                trainer_subwords.set_learning_rate(args.lr * (1 - progress))
            trainer.step(batch_size=1)
            trainer_subwords.step(batch_size=1)

            # Logging
            if i % args.eval_interval == 0:
                with print_time('mx.nd.waitall()'):
                    mx.nd.waitall()

                evaluate(args, embedding, vocab, num_update)

        # Log at the end of every epoch
        with print_time('mx.nd.waitall()'):
            mx.nd.waitall()
        evaluate(args, embedding, vocab, num_update,
                 eval_analogy=(epoch == args.epochs - 1
                               and not args.no_eval_analogy))

        # Save params at end of epoch
        save_params(args, embedding, embedding_out)


def evaluate(args, embedding, vocab, global_step, eval_analogy=False):
    """Evaluation helper"""
    if 'eval_tokens' not in globals():
        global eval_tokens

        eval_tokens_set = evaluation.get_tokens_in_evaluation_datasets(args)
        if not args.no_eval_analogy:
            eval_tokens_set.update(vocab.idx_to_token)
        eval_tokens = list(eval_tokens_set)

    os.makedirs(args.logdir, exist_ok=True)

    # Compute their word vectors
    context = get_context(args)
    idx_to_token = eval_tokens
    mx.nd.waitall()
    token_embedding = embedding.to_token_embedding(idx_to_token,
                                                   ctx=context[0])

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
