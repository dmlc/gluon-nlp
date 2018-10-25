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
"""Utils for embedding example.

This is based on train_fasttext.py in the scripts/word_embedding folder.

"""
import itertools
import math
import os

import mxnet as mx
import numpy as np

import gluonnlp as nlp
from gluonnlp.base import numba_jitclass, numba_types

os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'


def prepare_batches(data, ngrams, num_subwords, num_negatives, batch_size,
                    window):
    """Helper function to get training data."""

    counter = nlp.data.count_tokens(itertools.chain.from_iterable(data))
    vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                      bos_token=None, eos_token=None, min_freq=5)
    idx_to_counts = [counter[w] for w in vocab.idx_to_token]

    def code(sentence):
        return [vocab[token] for token in sentence if token in vocab]

    data = data.transform(code)

    negatives_sampler = nlp.data.UnigramCandidateSampler(
        weights=mx.nd.array(idx_to_counts)**0.75)

    sum_counts = float(sum(idx_to_counts))
    idx_to_pdiscard = [
        1 - math.sqrt(1e-5 / (count / sum_counts)) for count in idx_to_counts]

    def subsample(sentence):
        return [
            t for t, r in zip(sentence,
                              np.random.uniform(0, 1, size=len(sentence)))
            if r > idx_to_pdiscard[t]]

    data = data.transform(subsample)

    subword_function = nlp.vocab.create_subword_function(
        'NGramHashes', ngrams=ngrams, num_subwords=num_subwords)

    # Store subword indices for all words in vocabulary
    idx_to_subwordidxs = list(subword_function(vocab.idx_to_token))
    subword_lookup = SubwordLookup(len(vocab))
    for i, subwords in enumerate(idx_to_subwordidxs):
        subword_lookup.set(i, np.array(subwords, dtype=np.int64))

    def construct_batch(centers, contexts):
        """Create a batch for Skipgram training objective."""
        contexts_data, contexts_row, contexts_col = contexts

        negatives_shape = (len(centers), num_negatives)
        negatives = negatives_sampler(negatives_shape).astype(np.int64)

        # Remove accidental hits
        negatives_mask = (negatives != centers.expand_dims(1))
        negatives_mask = mx.nd.stack(
            negatives_mask, (negatives != contexts_col.expand_dims(1)))
        negatives_mask = negatives_mask.min(axis=0)
        negatives_mask = negatives_mask.astype(np.float32)

        contexts = contexts_col
        data, row, col = subword_lookup.skipgram(centers.asnumpy())
        centers = mx.nd.sparse.csr_matrix(
            (data, (row, col)),
            shape=(len(centers), len(vocab) + num_subwords), dtype=np.float32)

        return centers, contexts, negatives, negatives_mask

    batchify = nlp.data.batchify.EmbeddingCenterContextBatchify(
        batch_size=batch_size, window_size=window, cbow=False)
    centers_contexts = batchify(data)
    batches = centers_contexts.transform(construct_batch)

    return (batches, vocab, subword_function)


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


class SkipGramNet(mx.gluon.HybridBlock):
    """Base class for SkipGram and CBOW networks"""

    # pylint: disable=abstract-method
    def __init__(self, output_dim, vocab, negatives, subword_function,
                 sparse_grad=True, dtype='float32'):
        super().__init__()

        self._emsize = output_dim
        self._negatives = negatives
        self._dtype = dtype

        with self.name_scope():
            self.embedding = nlp.model.train.FasttextEmbeddingModel(
                token_to_idx=vocab.token_to_idx,
                subword_function=subword_function,
                output_dim=output_dim,
                weight_initializer=mx.init.Uniform(scale=1 / output_dim),
                sparse_grad=sparse_grad,
            )
            self.embedding_out = mx.gluon.nn.Embedding(
                len(vocab.token_to_idx), output_dim=output_dim,
                weight_initializer=mx.init.Zero(), sparse_grad=sparse_grad,
                dtype=dtype)

    def __getitem__(self, tokens):
        return self.embedding[tokens]

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
