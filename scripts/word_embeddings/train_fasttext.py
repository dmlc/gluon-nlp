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
import argparse
import itertools
import logging
import math
import os
import random
import sys
import time
import warnings

import mxnet as mx
import numpy as np

import gluonnlp as nlp
from gluonnlp.base import numba_jitclass, numba_types
import evaluation
from data import WikiDumpStream
from utils import get_context, print_time

os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'


# * Utils
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
    group.add_argument('--max-vocab-size', type=int,
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
    group.add_argument('--eval-interval', type=int,
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


def get_train_data(args):
    """Helper function to get training data."""

    def text8():
        """Text8 dataset helper."""
        data = nlp.data.Text8(segment='train')
        counter = nlp.data.count_tokens(itertools.chain.from_iterable(data))
        vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                          bos_token=None, eos_token=None, min_freq=5,
                          max_size=args.max_vocab_size)
        idx_to_counts = [counter[w] for w in vocab.idx_to_token]
        data = nlp.data.SimpleDataStream([data])
        return data, vocab, idx_to_counts

    def wiki():
        """Wikipedia dump helper."""
        data = WikiDumpStream(
            root=os.path.expanduser(args.wiki_root),
            language=args.wiki_language, date=args.wiki_date)
        vocab = data.vocab
        if args.max_vocab_size:
            for token in vocab.idx_to_token[args.max_vocab_size:]:
                vocab.token_to_idx.pop(token)
            vocab.idx_to_token = vocab.idx_to_token[:args.max_vocab_size]
        idx_to_counts = data.idx_to_counts
        return data, vocab, idx_to_counts

    with print_time('load training data'):
        f_data = text8 if args.data == 'text8' else wiki
        data, vocab, idx_to_counts = f_data()

    # Apply transforms
    def code(shard):
        return [[vocab[token] for token in sentence if token in vocab]
                for sentence in shard]

    data = data.transform(code)

    context = get_context(args)
    negatives_sampler = {
        ctx: nlp.data.UnigramCandidateSampler(
            weights=mx.nd.array(idx_to_counts, ctx=ctx)**0.75)
        for ctx in context}

    sum_counts = float(sum(idx_to_counts))
    idx_to_pdiscard = [
        1 - math.sqrt(args.frequent_token_subsampling / (count / sum_counts))
        for count in idx_to_counts]

    def subsample(shard):
        return [[
            t for t, r in zip(sentence,
                              np.random.uniform(0, 1, size=len(sentence)))
            if r > idx_to_pdiscard[t]] for sentence in shard]

    data = data.transform(subsample)

    if args.ngram_buckets:
        with print_time('prepare subwords'):
            subword_function = nlp.vocab.create_subword_function(
                'NGramHashes', ngrams=args.ngrams,
                num_subwords=args.ngram_buckets)

            # Store subword indices for all words in vocabulary
            idx_to_subwordidxs = list(subword_function(vocab.idx_to_token))
            subword_lookup = SubwordLookup(len(vocab))
            for i, subwords in enumerate(idx_to_subwordidxs):
                subword_lookup.set(i, np.array(subwords, dtype=np.int64))
            max_subwordidxs_len = max(len(s) for s in idx_to_subwordidxs)
            if max_subwordidxs_len > 500:
                warnings.warn(
                    'The word with largest number of subwords '
                    'has {} subwords, suggesting there are '
                    'some noisy words in your vocabulary. '
                    'You should filter out very long words '
                    'to avoid memory issues.'.format(max_subwordidxs_len))

        return (data, negatives_sampler, vocab, subword_function,
                subword_lookup, sum_counts)
    else:
        return data, negatives_sampler, vocab, sum_counts


@numba_jitclass([('idx_to_subwordidxs',
                  numba_types.List(numba_types.int_[::1])),
                 ('offset', numba_types.int_)])
class SubwordLookup(object):
    """Just-in-time compiled helper class for fast subword lookup.

    Parameters
    ----------
    num_words : int
         Number of tokens for which to hold subword arrays.

    """

    def __init__(self, num_words):
        self.idx_to_subwordidxs = [
            np.arange(1).astype(np.int64) for _ in range(num_words)]
        self.offset = num_words

    def set(self, i, subwords):
        """Set the subword array of the i-th token."""
        self.idx_to_subwordidxs[i] = subwords

    def skipgram(self, indices):
        """Get a sparse COO array of words and subwords."""
        row = []
        col = []
        data = []
        for i, idx in enumerate(indices):
            row.append(i)
            col.append(idx)
            data.append(1 / (1 + len(self.idx_to_subwordidxs[idx])))
            for subword in self.idx_to_subwordidxs[idx]:
                row.append(i)
                col.append(subword + self.offset)
                data.append(1 / (1 + len(self.idx_to_subwordidxs[idx])))
        return np.array(data), np.array(row), np.array(col)

    def cbow(self, context_row, context_col):
        """Get a sparse COO array of words and subwords."""
        row = []
        col = []
        data = []

        num_rows = np.max(context_row) + 1
        row_to_numwords = np.zeros(num_rows)

        for i, idx in enumerate(context_col):
            row_ = context_row[i]
            row_to_numwords[row_] += 1

            row.append(row_)
            col.append(idx)
            data.append(1 / (1 + len(self.idx_to_subwordidxs[idx])))
            for subword in self.idx_to_subwordidxs[idx]:
                row.append(row_)
                col.append(subword + self.offset)
                data.append(1 / (1 + len(self.idx_to_subwordidxs[idx])))

        # Normalize by number of words
        for i, row_ in enumerate(row):
            assert 0 <= row_ <= num_rows
            data[i] /= row_to_numwords[row_]

        return np.array(data), np.array(row), np.array(col)


# * Training code
class Net(mx.gluon.HybridBlock):
    """Base class for SkipGram and CBOW networks"""

    # pylint: disable=abstract-method
    def __init__(self, output_dim, vocab, negatives, subword_function=None,
                 sparse_grad=True, dtype='float32'):
        super().__init__()

        self._emsize = output_dim
        self._negatives = negatives
        self._dtype = dtype

        with self.name_scope():
            if subword_function is not None:
                self.embedding = nlp.model.train.FasttextEmbeddingModel(
                    token_to_idx=vocab.token_to_idx,
                    subword_function=subword_function, output_dim=output_dim,
                    weight_initializer=mx.init.Uniform(scale=1 / output_dim),
                    sparse_grad=sparse_grad)
            else:
                self.embedding = nlp.model.train.CSREmbeddingModel(
                    token_to_idx=vocab.token_to_idx, output_dim=output_dim,
                    weight_initializer=mx.init.Uniform(scale=1 / output_dim),
                    sparse_grad=sparse_grad)
            self.embedding_out = mx.gluon.nn.Embedding(
                len(vocab.token_to_idx), output_dim=output_dim,
                weight_initializer=mx.init.Zero(), sparse_grad=sparse_grad,
                dtype=dtype)

    def __getitem__(self, tokens):
        return self.embedding[tokens]


class SG(Net):
    """SkipGram network"""

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, center, context, negatives, mask):
        """SkipGram forward pass.

        Parameters
        ----------
        center : mxnet.nd.NDArray or mxnet.sym.Symbol
            Sparse CSR array of word / subword indices of shape (batch_size,
            len(vocab) + num_subwords). Embedding for center words are computed
            via F.sparse.dot between the CSR center array and the weight
            matrix.
        context : mxnet.nd.NDArray or mxnet.sym.Symbol
            Dense array of context words of shape (batch_size, ).
        negatives : mxnet.nd.NDArray or mxnet.sym.Symbol
            Dense array of negative words of shape (batch_size * negatives, ).
        mask : mxnet.nd.NDArray or mxnet.sym.Symbol
            Dense array containing mask for negatives of shape (batch_size *
            negatives, ).

        """
        emb_center = self.embedding(center).expand_dims(1)
        emb_context = self.embedding_out(context).expand_dims(2)
        pred_pos = F.batch_dot(emb_center, emb_context).squeeze()
        loss_pos = (F.relu(pred_pos) - pred_pos + F.Activation(
            -F.abs(pred_pos), act_type='softrelu')) / (mask.sum(axis=1) + 1)

        emb_negatives = self.embedding_out(negatives).reshape(
            (-1, self._negatives, self._emsize)).swapaxes(1, 2)
        pred_neg = F.batch_dot(emb_center, emb_negatives).squeeze()
        mask = mask.reshape((-1, self._negatives))
        loss_neg = (F.relu(pred_neg) + F.Activation(
            -F.abs(pred_neg), act_type='softrelu')) * mask
        loss_neg = loss_neg.sum(axis=1) / (mask.sum(axis=1) + 1)

        return loss_pos + loss_neg


class CBOW(Net):
    """CBOW network"""

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, center, context, negatives, mask):
        """CBOW forward pass.

        Parameters
        ----------
        center : mxnet.nd.NDArray or mxnet.sym.Symbol
            Dense array of center words of shape (batch_size, ).
        context : mxnet.nd.NDArray or mxnet.sym.Symbol
            Sparse CSR array of word / subword indices of shape (batch_size,
            len(vocab) + num_subwords). Embedding for context words are
            computed via F.sparse.dot between the CSR center array and the
            weight matrix.
        negatives : mxnet.nd.NDArray or mxnet.sym.Symbol
            Dense array of negative words of shape (batch_size * negatives, ).
        mask : mxnet.nd.NDArray or mxnet.sym.Symbol
            Dense array containing mask for negatives of shape (batch_size *
            negatives, ).

        """
        emb_context = self.embedding(context).expand_dims(1)
        emb_center = self.embedding_out(center).expand_dims(2)
        pred_pos = F.batch_dot(emb_context, emb_center).squeeze()
        loss_pos = (F.relu(pred_pos) - pred_pos + F.Activation(
            -F.abs(pred_pos), act_type='softrelu')) / (mask.sum(axis=1) + 1)

        emb_negatives = self.embedding_out(negatives).reshape(
            (-1, self._negatives, self._emsize)).swapaxes(1, 2)
        pred_neg = F.batch_dot(emb_context, emb_negatives).squeeze()
        mask = mask.reshape((-1, self._negatives))
        loss_neg = (F.relu(pred_neg) + F.Activation(
            -F.abs(pred_neg), act_type='softrelu')) * mask
        loss_neg = loss_neg.sum(axis=1) / (mask.sum(axis=1) + 1)

        return loss_pos + loss_neg


def train(args):
    """Training helper."""
    if args.ngram_buckets:
        data, negatives_sampler, vocab, subword_function, \
            subword_lookup, num_tokens = get_train_data(args)
    else:
        data, negatives_sampler, vocab, num_tokens = get_train_data(args)
        subword_function = None

    if args.model.lower() == 'cbow':
        embedding = CBOW(args.emsize, vocab, args.negative, subword_function)
    elif args.model.lower() == 'skipgram':
        embedding = SG(args.emsize, vocab, args.negative, subword_function)
    else:
        logging.error('Unsupported model %s.', args.model)
        sys.exit(1)

    context = get_context(args)
    embedding.initialize(ctx=context)
    if not args.no_hybridize:
        embedding.hybridize(static_alloc=not args.no_static_alloc)

    optimizer_kwargs = dict(learning_rate=args.lr)
    trainer = mx.gluon.Trainer(embedding.collect_params(), args.optimizer,
                               optimizer_kwargs)

    def construct_batch(data, ctx):
        """Create a batch for Skipgram training objective."""
        centers_cpu, contexts_cpu = data
        (contexts_data_cpu, contexts_row_cpu, contexts_col_cpu) = contexts_cpu

        negatives_shape = (len(centers_cpu), args.negative)
        negatives = negatives_sampler[ctx](negatives_shape).astype(np.int64)

        # Remove accidental hits
        centers = centers_cpu.as_in_context(ctx)
        contexts_col = contexts_col_cpu.as_in_context(ctx)
        negatives_mask = (negatives != centers.expand_dims(1))
        if args.model.lower() != 'cbow':
            negatives_mask = mx.nd.stack(
                negatives_mask, (negatives != contexts_col.expand_dims(1)))
            negatives_mask = negatives_mask.min(axis=0)
        negatives_mask = negatives_mask.astype(np.float32)

        if args.ngram_buckets and args.model.lower() == 'cbow':
            data, row, col = subword_lookup.cbow(contexts_row_cpu.asnumpy(),
                                                 contexts_col_cpu.asnumpy())
            contexts = mx.nd.sparse.csr_matrix(
                (data, (row, col)), dtype=np.float32, ctx=ctx,
                shape=(len(centers), embedding.embedding.weight.shape[0]))
        elif args.model.lower() == 'cbow':
            contexts = mx.nd.sparse.csr_matrix(
                (contexts_data_cpu, (contexts_row_cpu, contexts_col_cpu)),
                shape=(len(centers), embedding.embedding.weight.shape[0]),
                dtype=np.float32, ctx=ctx)
        elif args.ngram_buckets and args.model.lower() == 'skipgram':
            contexts = contexts_col
            data, row, col = subword_lookup.skipgram(centers_cpu.asnumpy())
            centers = mx.nd.sparse.csr_matrix(
                (data, (row, col)), dtype=np.float32, ctx=ctx,
                shape=(len(centers), embedding.embedding.weight.shape[0]))
        elif args.model.lower() == 'skipgram':
            contexts = contexts_col
            centers = mx.nd.sparse.csr_matrix(
                (mx.nd.ones(centers.shape, ctx=ctx), centers,
                 mx.nd.arange(len(centers) + 1, ctx=ctx)),
                shape=(len(centers), embedding.embedding.weight.shape[0]),
                dtype=np.float32, ctx=ctx)
        else:
            logging.error('Unsupported model %s.', args.model)
            sys.exit(1)

        return centers, contexts, negatives, negatives_mask

    batchify = nlp.data.batchify.EmbeddingCenterContextBatchify(
        batch_size=args.batch_size, window_size=args.window,
        cbow=args.model.lower() == 'cbow')
    data = data.transform(batchify)

    num_update = 0
    for epoch in range(args.epochs):
        # Logging variables
        log_wc = 0
        log_start_time = time.time()
        log_avg_loss = 0

        samples = itertools.chain.from_iterable(data)

        for i, sample in enumerate(samples):
            ctx = context[i % len(context)]
            batch = construct_batch(sample, ctx)
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
