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

# pylint: disable=abstract-method
"""Trainable embedding models."""

__all__ = ['EmbeddingModel', 'SimpleEmbeddingModel', 'FasttextEmbeddingModel']

import numpy as np
from mxnet import cpu, nd
from mxnet.gluon import Block, HybridBlock, nn

from ... import embedding as emb
from ...data.batchify import Pad


class EmbeddingModel(Block):
    """A trainable embedding model.

    This class is defines some common methods but is not usable standalone.
    Subclasses implement the main functionality.

    Parameters
    ----------
    embedding_size : int
        Dimension of embeddings.

    """

    def __init__(self, embedding_size, **kwargs):
        super(EmbeddingModel, self).__init__(**kwargs)
        self.embedding_size = embedding_size

    def _to_token_embedding_batch(self, batch, token_to_idx,
                                  model_subwordfunction, unknown_behavior,
                                  ctx):
        # Handle subwords
        subword_padding = Pad(pad_val=-1)
        if model_subwordfunction is not None:
            subwords = model_subwordfunction(batch)
            subwords = subword_padding(subwords)
            subwords_mask = subwords != -1
            subwords += subwords == -1  # -1 is invalid. Change to 0

            if unknown_behavior == 'impute_raise':
                # Check that subwords are present  for all tokens
                without_subwords_idxs = np.where(
                    (subwords_mask.max(axis=1) == 0).asnumpy())[0].tolist()
                if len(without_subwords_idxs):
                    without_subwords = [
                        batch[idx] for idx in without_subwords_idxs
                    ]
                    raise ValueError('No subwords were found for: ' +
                                     ', '.join(without_subwords))

            subwords = nd.array(subwords, ctx=ctx)
            subwords_mask = nd.array(subwords_mask,
                                     dtype=np.float32).as_in_context(ctx)

        # Handle words
        if unknown_behavior == 'raise':
            if any(token not in token_to_idx for token in batch):
                raise ValueError
            words = [token_to_idx[token] for token in batch]
            mask = nd.ones(shape=(len(words), ), ctx=ctx)
        if unknown_behavior == 'impute_raise':
            words = [
                token_to_idx[token] if token in token_to_idx else 0
                for token in batch
            ]
            mask = nd.array(
                [1 if token in token_to_idx else 0 for token in batch],
                ctx=ctx)

        words = nd.array(words, ctx=ctx)

        # Compute embeddings
        if model_subwordfunction is not None:
            return self(words, mask, subwords, subwords_mask)
        else:
            return self(words, mask)

    def to_token_embedding(
            self, tokens, model_token_to_idx, subword_function=None,
            unknown_behavior='impute_raise', batch_size=1024, ctx=cpu()):
        """Computes a TokenEmbedding from the trained embedding model.

        Parameters
        ----------
        tokens : list of str
            The tokens for which to add vectors to the resulting
            TokenEmbedding.
        model_token_to_idx : :class:`dict` instance
            The token_to_idx mapping used when training the `embedding_model`.
            It contains all token-index pairs observed during training.
        subword_function : :class:`gluonnlp.vocab.SubwordFunction`, optional
            The subword vocabulary of the `EmbeddingModel`. Only needed if the
            `EmbeddingModel` makes use of subword information.
        unknown_behavior : ['impute_raise', 'raise'], default 'impute_raise'
            How to handle tokens that are not in the `model_token_to_idx`.
              - 'impute_raise' tries to impute an embedding based on the
                subwords of the token as computed from `model_subwordfunction`.
                If no subwords are associated with the respective token or
                `model_subwordfunction` is None, a ValueError is raised. -
                'raise' raises a ValueError if any token is not in
                `model_token_to_idx`.
        batch_size : int, default 1024
            Use batches of `batch_size` to compute the embeddings from the
            `embedding_model`.
        ctx
            Context to perform computation on.
        """

        new_embedding = emb.TokenEmbedding(unknown_token=None)
        new_embedding._token_to_idx.update(
            (token, idx) for idx, token in enumerate(tokens))
        new_embedding._idx_to_token = tokens
        new_idx_to_vec = []

        # Compute embeddings in batches for all idx
        start_pointer = 0
        end_pointer = batch_size
        while True:
            batch = tokens[start_pointer:end_pointer]
            batch_embedding = self._to_token_embedding_batch(
                batch, model_token_to_idx, subword_function, unknown_behavior,
                ctx)
            new_idx_to_vec.append(batch_embedding.as_in_context(cpu()))

            if end_pointer >= len(tokens):
                break
            start_pointer += batch_size
            end_pointer += batch_size

        new_idx_to_vec = nd.concat(*new_idx_to_vec, dim=0)
        new_embedding._idx_to_vec = new_idx_to_vec
        return new_embedding


class SimpleEmbeddingModel(EmbeddingModel, HybridBlock):
    """A trainable embedding model.

    This class is a simple wrapper around the mxnet.gluon.nn.Embedding.

    Parameters
    ----------
    num_tokens : int
        Number of tokens in the vocabulary.
    embedding_size : int
        Dimension of embeddings.
    weight_initializer : mxnet.initializer.Initializer, optional
        Initializer for the embeddings matrix.
    sparse_grad : bool, default True
        Specifies mxnet.gluon.nn.Embedding sparse_grad argument.

    """

    def __init__(self, num_tokens, embedding_size, weight_initializer=None,
                 sparse_grad=True, **kwargs):
        super(SimpleEmbeddingModel,
              self).__init__(embedding_size=embedding_size, **kwargs)
        self.num_tokens = num_tokens
        self.weight_initializer = weight_initializer
        self.sparse_grad = sparse_grad

        with self.name_scope():
            self.embedding = nn.Embedding(
                num_tokens, embedding_size,
                weight_initializer=weight_initializer, sparse_grad=sparse_grad)

    def hybrid_forward(self, F, words, wordsmask):
        """Compute embedding of words in batch.

        Parameters
        ----------
        words : mx.nd.NDArray
            Array of token indices.
        wordsmask : mx.nd.NDArray
            Mask for embeddings returend by the word level embedding operator.

        """
        #pylint: disable=arguments-differ
        wordsmask = F.expand_dims(wordsmask, axis=-1)
        return F.broadcast_mul(self.embedding(words), wordsmask)


class _MaskedSumEmbedding(HybridBlock):
    def __init__(self, num_tokens, embedding_size, weight_initializer=None,
                 sparse_grad=True, **kwargs):
        super(_MaskedSumEmbedding, self).__init__(**kwargs)
        self.num_tokens = num_tokens
        self.embedding_size = embedding_size
        self.weight_initializer = weight_initializer
        self.sparse_grad = sparse_grad

        with self.name_scope():
            self.embedding = nn.Embedding(
                num_tokens,
                embedding_size,
                weight_initializer=weight_initializer,
                sparse_grad=sparse_grad,
            )

    def hybrid_forward(self, F, x, mask):
        #pylint: disable=arguments-differ
        mask = F.expand_dims(mask, axis=-1)
        masked_embeddings = F.broadcast_mul(self.embedding(x), mask)
        return F.sum(masked_embeddings, axis=-2)


class FasttextEmbeddingModel(EmbeddingModel):
    """FastText embedding model.

    A FasttextEmbeddingModel combines a word level embedding matrix and a
    subword level embedding matrix.

    Parameters
    ----------
    num_tokens : int
        Number of tokens in the vocabulary.
    num_subwords : int
        Number subwords.
    embedding_size : int
        Dimension of embeddings.
    weight_initializer : mxnet.initializer.Initializer, optional
        Initializer for the embeddings and subword embeddings matrix.
    sparse_grad : bool, default True
        Specifies mxnet.gluon.nn.Embedding sparse_grad argument.

    """

    def __init__(self, num_tokens, num_subwords, embedding_size,
                 weight_initializer=None, sparse_grad=True, **kwargs):
        super(FasttextEmbeddingModel,
              self).__init__(embedding_size=embedding_size, **kwargs)
        self.num_tokens = num_tokens
        self.weight_initializer = weight_initializer
        self.sparse_grad = sparse_grad

        with self.name_scope():
            self.embedding = nn.Embedding(
                num_tokens,
                embedding_size,
                weight_initializer=weight_initializer,
                sparse_grad=sparse_grad,
            )
            self.subword_embedding = _MaskedSumEmbedding(
                num_subwords,
                embedding_size,
                weight_initializer=weight_initializer,
                sparse_grad=sparse_grad,
            )

    def forward(self, words, wordsmask, subwords, subwordsmask, F=nd):
        """Compute embedding of words in batch.

        Parameters
        ----------
        words : mx.nd.NDArray
            Array of token indices.
        wordsmask : mx.nd.NDArray
            Mask for embeddings returend by the word level embedding operator.
        subwords : mx.nd.NDArray
            The subwords associated with the tokens in `words`.
        subwordsmask : mx.nd.NDArray
            A mask for the subword embeddings looked up from `subwords`.
            Applied before sum reducing the subword embeddings.

        """
        #pylint: disable=arguments-differ
        wordsmask = F.expand_dims(wordsmask, axis=-1)
        embeddings = F.broadcast_mul(self.embedding(words), wordsmask)
        subword_embeddings = self.subword_embedding(subwords, subwordsmask)
        return embeddings + subword_embeddings
