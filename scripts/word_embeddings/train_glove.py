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
"""GloVe embedding model
===========================

This example shows how to train a GloVe embedding model based on the vocabulary
and co-occurrence matrix constructed by the vocab_count and cooccur tool. The
tools are located in the same ./tools folder next to this script.

The GloVe model was introduced by

- Pennington, J., Socher, R., & Manning, C. D. (2014). Glove: global vectors
  for word representation. In A. Moschitti, B. Pang, & W. Daelemans,
  Proceedings of the 2014 Conference on Empirical Methods in Natural Language
  Processing, {EMNLP} 2014, October 25-29, 2014, Doha, Qatar, {A} meeting of
  SIGDAT, a Special Interest Group of the {ACL (pp. 1532–1543). : ACL.

"""
# * Imports
import argparse
import io
import logging
import os
import random
import sys
import tempfile
import time

import mxnet as mx
import numpy as np

import evaluation
import gluonnlp as nlp
from gluonnlp.base import _str_types
from utils import get_context, print_time

os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'


# * Utils
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='GloVe with GluonNLP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data options
    group = parser.add_argument_group('Data arguments')
    group.add_argument(
        'cooccurrences', type=str,
        help='Path to cooccurrences.npz containing a sparse (COO) '
        'representation of the co-occurrence matrix in numpy archive format. '
        'Output of ./cooccur')
    group.add_argument('vocab', type=str,
                       help='Vocabulary indices. Output of vocab_count tool.')

    # Computation options
    group = parser.add_argument_group('Computation arguments')
    group.add_argument('--batch-size', type=int, default=512,
                       help='Batch size for training.')
    group.add_argument('--epochs', type=int, default=50, help='Epoch limit')
    group.add_argument(
        '--gpu', type=int, nargs='+',
        help='Number (index) of GPU to run on, e.g. 0. '
        'If not specified, uses CPU.')
    group.add_argument('--no-hybridize', action='store_true',
                       help='Disable hybridization of gluon HybridBlocks.')
    group.add_argument(
        '--no-static-alloc', action='store_true',
        help='Disable static memory allocation for HybridBlocks.')

    # Model
    group = parser.add_argument_group('Model arguments')
    group.add_argument('--emsize', type=int, default=300,
                       help='Size of embedding vectors.')
    group.add_argument('--x-max', type=int, default=100)
    group.add_argument('--alpha', type=float, default=0.75)

    # Optimization options
    group = parser.add_argument_group('Optimization arguments')
    group.add_argument('--adagrad-eps', type=float, default=1,
                       help='Initial AdaGrad state value.')
    group.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    group.add_argument('--seed', type=int, default=1, help='Random seed')
    group.add_argument('--dropout', type=float, default=0.15)

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


def get_train_data(args):
    """Helper function to get training data."""
    counter = dict()
    with io.open(args.vocab, 'r', encoding='utf-8') as f:
        for line in f:
            token, count = line.split('\t')
            counter[token] = int(count)
    vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                      bos_token=None, eos_token=None, min_freq=1)

    npz = np.load(args.cooccurrences)
    row, col, counts = npz['row'], npz['col'], npz['data']

    rank_dtype = 'int32'
    if row.max() >= np.iinfo(np.int32).max:
        rank_dtype = 'int64'
        # MXNet has no support for uint32, so we must fall back to int64
        logging.info('More words than could be counted using int32. '
                     'Using int64 to represent word indices.')
    row = mx.nd.array(row, dtype=rank_dtype)
    col = mx.nd.array(col, dtype=rank_dtype)
    # row is always used as 'source' and col as 'context' word. Therefore
    # duplicate the entries.

    assert row.shape == col.shape
    row = mx.nd.concatenate([row, col])
    col = mx.nd.concatenate([col, row[:len(row) // 2]])

    counts = mx.nd.array(counts, dtype='float32')
    counts = mx.nd.concatenate([counts, counts])

    return vocab, row, col, counts


# * Gluon Block definition
class GloVe(nlp.model.train.EmbeddingModel, mx.gluon.HybridBlock):
    """GloVe EmbeddingModel"""

    def __init__(self, token_to_idx, output_dim, x_max, alpha, dropout=0,
                 weight_initializer=None,
                 bias_initializer=mx.initializer.Zero(), sparse_grad=True,
                 dtype='float32', **kwargs):
        assert isinstance(token_to_idx, dict)

        super(GloVe, self).__init__(**kwargs)
        self.token_to_idx = token_to_idx
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.sparse_grad = sparse_grad
        self.dtype = dtype

        self._x_max = x_max
        self._alpha = alpha
        self._dropout = dropout

        with self.name_scope():
            self.source_embedding = mx.gluon.nn.Embedding(
                len(token_to_idx), output_dim,
                weight_initializer=weight_initializer, sparse_grad=sparse_grad,
                dtype=dtype)
            self.context_embedding = mx.gluon.nn.Embedding(
                len(token_to_idx), output_dim,
                weight_initializer=weight_initializer, sparse_grad=sparse_grad,
                dtype=dtype)
            self.source_bias = mx.gluon.nn.Embedding(
                len(token_to_idx), 1, weight_initializer=bias_initializer,
                sparse_grad=sparse_grad, dtype=dtype)
            self.context_bias = mx.gluon.nn.Embedding(
                len(token_to_idx), 1, weight_initializer=bias_initializer,
                sparse_grad=sparse_grad, dtype=dtype)

    def hybrid_forward(self, F, row, col, counts):
        """Compute embedding of words in batch.

        Parameters
        ----------
        row : mxnet.nd.NDArray or mxnet.sym.Symbol
            Array of token indices for source words. Shape (batch_size, ).
        row : mxnet.nd.NDArray or mxnet.sym.Symbol
            Array of token indices for context words. Shape (batch_size, ).
        counts : mxnet.nd.NDArray or mxnet.sym.Symbol
            Their co-occurrence counts. Shape (batch_size, ).

        Returns
        -------
        mxnet.nd.NDArray or mxnet.sym.Symbol
            Loss. Shape (batch_size, ).

        """

        emb_in = self.source_embedding(row)
        emb_out = self.context_embedding(col)

        if self._dropout:
            emb_in = F.Dropout(emb_in, p=self._dropout)
            emb_out = F.Dropout(emb_out, p=self._dropout)

        bias_in = self.source_bias(row).squeeze()
        bias_out = self.context_bias(col).squeeze()
        dot = F.batch_dot(emb_in.expand_dims(1),
                          emb_out.expand_dims(2)).squeeze()
        tmp = dot + bias_in + bias_out - F.log(counts).squeeze()
        weight = F.clip(((counts / self._x_max)**self._alpha), a_min=0,
                        a_max=1).squeeze()
        loss = weight * F.square(tmp)
        return loss

    def __contains__(self, token):
        return token in self.idx_to_token

    def __getitem__(self, tokens):
        """Looks up embedding vectors of text tokens.

        Parameters
        ----------
        tokens : str or list of strs
            A token or a list of tokens.

        Returns
        -------
        mxnet.ndarray.NDArray:
            The embedding vector(s) of the token(s). According to numpy
            conventions, if `tokens` is a string, returns a 1-D NDArray
            (vector); if `tokens` is a list of strings, returns a 2-D NDArray
            (matrix) of shape=(len(tokens), vec_len).
        """
        squeeze = False
        if isinstance(tokens, _str_types):
            tokens = [tokens]
            squeeze = True

        indices = mx.nd.array([self.token_to_idx[t] for t in tokens],
                              ctx=self.source_embedding.weight.list_ctx()[0])
        vecs = self.source_embedding(indices) + self.context_embedding(indices)

        if squeeze:
            assert len(vecs) == 1
            return vecs[0].squeeze()
        else:
            return vecs


# * Training code
def train(args):
    """Training helper."""
    vocab, row, col, counts = get_train_data(args)
    model = GloVe(token_to_idx=vocab.token_to_idx, output_dim=args.emsize,
                  dropout=args.dropout, x_max=args.x_max, alpha=args.alpha,
                  weight_initializer=mx.init.Uniform(scale=1 / args.emsize))
    context = get_context(args)
    model.initialize(ctx=context)
    if not args.no_hybridize:
        model.hybridize(static_alloc=not args.no_static_alloc)

    optimizer_kwargs = dict(learning_rate=args.lr, eps=args.adagrad_eps)
    params = list(model.collect_params().values())
    try:
        trainer = mx.gluon.Trainer(params, 'groupadagrad', optimizer_kwargs)
    except ValueError:
        logging.warning('MXNet <= v1.3 does not contain '
                        'GroupAdaGrad support. Falling back to AdaGrad')
        trainer = mx.gluon.Trainer(params, 'adagrad', optimizer_kwargs)

    index_dtype = 'int32'
    if counts.shape[0] >= np.iinfo(np.int32).max:
        index_dtype = 'int64'
        logging.info('Co-occurrence matrix is large. '
                     'Using int64 to represent sample indices.')
    indices = mx.nd.arange(counts.shape[0], dtype=index_dtype)
    for epoch in range(args.epochs):
        # Logging variables
        log_wc = 0
        log_start_time = time.time()
        log_avg_loss = 0

        mx.nd.shuffle(indices, indices)  # inplace shuffle
        bs = args.batch_size
        num_batches = indices.shape[0] // bs
        for i in range(num_batches):
            batch_indices = indices[bs * i:bs * (i + 1)]
            ctx = context[i % len(context)]
            batch_row = row[batch_indices].as_in_context(ctx)
            batch_col = col[batch_indices].as_in_context(ctx)
            batch_counts = counts[batch_indices].as_in_context(ctx)
            with mx.autograd.record():
                loss = model(batch_row, batch_col, batch_counts)
                loss.backward()

            if len(context) == 1 or (i + 1) % len(context) == 0:
                trainer.step(batch_size=1)

            # Logging
            log_wc += loss.shape[0]
            log_avg_loss += loss.mean().as_in_context(context[0])
            if (i + 1) % args.log_interval == 0:
                # Forces waiting for computation by computing loss value
                log_avg_loss = log_avg_loss.asscalar() / args.log_interval
                wps = log_wc / (time.time() - log_start_time)
                logging.info('[Epoch {} Batch {}/{}] loss={:.4f}, '
                             'throughput={:.2f}K wps, wc={:.2f}K'.format(
                                 epoch, i + 1, num_batches, log_avg_loss,
                                 wps / 1000, log_wc / 1000))
                log_dict = dict(
                    global_step=epoch * len(indices) + i * args.batch_size,
                    epoch=epoch, batch=i + 1, loss=log_avg_loss,
                    wps=wps / 1000)
                log(args, log_dict)

                log_start_time = time.time()
                log_avg_loss = 0
                log_wc = 0

            if args.eval_interval and (i + 1) % args.eval_interval == 0:
                with print_time('mx.nd.waitall()'):
                    mx.nd.waitall()
                with print_time('evaluate'):
                    evaluate(args, model, vocab, i + num_batches * epoch)

    # Evaluate
    with print_time('mx.nd.waitall()'):
        mx.nd.waitall()
    with print_time('evaluate'):
        evaluate(args, model, vocab, num_batches * args.epochs,
                 eval_analogy=not args.no_eval_analogy)

    # Save params
    with print_time('save parameters'):
        model.save_parameters(os.path.join(args.logdir, 'glove.params'))


# * Evaluation
def evaluate(args, model, vocab, global_step, eval_analogy=False):
    """Evaluation helper"""
    if 'eval_tokens' not in globals():
        global eval_tokens

        eval_tokens_set = evaluation.get_tokens_in_evaluation_datasets(args)
        if not args.no_eval_analogy:
            eval_tokens_set.update(vocab.idx_to_token)

        # GloVe does not support computing vectors for OOV words
        eval_tokens_set = filter(lambda t: t in vocab, eval_tokens_set)

        eval_tokens = list(eval_tokens_set)

    # Compute their word vectors
    context = get_context(args)
    mx.nd.waitall()

    token_embedding = nlp.embedding.TokenEmbedding(unknown_token=None,
                                                   allow_extend=True)
    token_embedding[eval_tokens] = model[eval_tokens]

    results = evaluation.evaluate_similarity(
        args, token_embedding, context[0], logfile=os.path.join(
            args.logdir, 'similarity.tsv'), global_step=global_step)
    if eval_analogy:
        assert not args.no_eval_analogy
        results += evaluation.evaluate_analogy(
            args, token_embedding, context[0], logfile=os.path.join(
                args.logdir, 'analogy.tsv'))

    return results


# * Logging
def log(args, kwargs):
    """Log to a file."""
    logfile = os.path.join(args.logdir, 'log.tsv')

    if 'log_created' not in globals():
        if os.path.exists(logfile):
            logging.error('Logfile %s already exists.', logfile)
            sys.exit(1)

        global log_created

        log_created = sorted(kwargs.keys())
        header = '\t'.join((str(k) for k in log_created)) + '\n'
        with open(logfile, 'w') as f:
            f.write(header)

    # Log variables shouldn't change during training
    assert log_created == sorted(kwargs.keys())

    with open(logfile, 'a') as f:
        f.write('\t'.join((str(kwargs[k]) for k in log_created)) + '\n')


# * Main
if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    args_ = parse_args()

    if os.path.exists(args_.logdir):
        newlogdir = tempfile.mkdtemp(dir=args_.logdir)
        logging.warning('%s exists. Using %s', args_.logdir, newlogdir)
        args_.logdir = newlogdir
    if not os.path.isdir(args_.logdir):
        os.makedirs(args_.logdir)

    train(args_)
