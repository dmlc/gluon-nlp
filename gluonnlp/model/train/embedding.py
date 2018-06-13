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
    """Abstract base class for trainable embedding models

    Subclasses implement concrete network architectures. All EmbeddingModels
    support conversion to TokenEmbedding by calling the to_token_embedding()
    method.

    Parameters
    ----------
    embedding_size : int
        Dimension of embeddings.

    """

    def __init__(self, embedding_size, **kwargs):
        super(EmbeddingModel, self).__init__(**kwargs)
        self.embedding_size = embedding_size

    def to_token_embedding(self, tokens, unknown_behavior='impute_raise',
                           batch_size=1024, ctx=cpu()):
        """Computes a TokenEmbedding from the trained embedding model.

        Parameters
        ----------
        tokens : list of str
            The tokens for which to add vectors to the resulting
            TokenEmbedding.
        unknown_behavior : str, default 'impute_raise'
          - 'impute_raise' Try to impute an embedding for out of vocabulary
            words. If imputation is impossible (e.g. no known subwords are
            associated with the out of vocabulary word) , a ValueError is
            raised.
          - 'raise' raises a ValueError if any token is out of vocaublary.
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
                batch, unknown_behavior, ctx)
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

    This class is a simple wrapper around the mxnet.gluon.nn.Embedding. It
    trains independent embedding vectors for every token. It implements the
    `gluonnlp.model.train.EmbeddingModel` interface which provides convenient
    functions.

    Parameters
    ----------
    token_to_idx : dict
        token_to_idx mapping of the vocabulary that this model is to be trained
        with. token_to_idx is used when to_token_embedding is called. For
        initialization len(token_to_idx) is used to specify the size of the
        subword embedding matrix..
    embedding_size : int
        Dimension of embeddings.
    weight_initializer : mxnet.initializer.Initializer, optional
        Initializer for the embeddings matrix.
    sparse_grad : bool, default True
        Specifies mxnet.gluon.nn.Embedding sparse_grad argument.

    """

    def __init__(self, token_to_idx, embedding_size, weight_initializer=None,
                 sparse_grad=True, **kwargs):
        super(SimpleEmbeddingModel,
              self).__init__(embedding_size=embedding_size, **kwargs)
        self.token_to_idx = token_to_idx
        self.weight_initializer = weight_initializer
        self.sparse_grad = sparse_grad

        with self.name_scope():
            self.embedding = nn.Embedding(
                len(token_to_idx), embedding_size,
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

    def _to_token_embedding_batch(self, batch, unknown_behavior, ctx):
        # SimpleEmbeddingModel does not support imputation
        if unknown_behavior == 'raise' or unknown_behavior == 'impute_raise':
            if any(token not in self.token_to_idx for token in batch):
                raise ValueError
            words = [self.token_to_idx[token] for token in batch]
            mask = nd.ones(shape=(len(words), ), ctx=ctx)
        else:
            raise RuntimeError(
                'Unsupported unknown_behavior {}'.format(unknown_behavior))

        words = nd.array(words, ctx=ctx)

        return self(words, mask)


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


class FasttextEmbeddingModel(EmbeddingModel, HybridBlock):
    """FastText embedding model.

    The FasttextEmbeddingModel combines a word level embedding matrix and a
    subword level embedding matrix. It implements the
    `gluonnlp.model.train.EmbeddingModel` interface which provides convenient
    functions.

    Parameters
    ----------
    token_to_idx : dict
        token_to_idx mapping of the vocabulary that this model is to be trained
        with. token_to_idx is used when to_token_embedding is called. For
        initialization len(token_to_idx) is used to specify the size of the
        subword embedding matrix..
    subword_function : gluonnlp.vocab.SubwordFunction
        The subword function used to obtain the subword indices during training
        this model. The subword_function is used when to_token_embedding is
        called. For initialization len(subword_function) is used to specify the
        size of the subword embedding matrix..
    embedding_size : int
        Dimension of embeddings.
    weight_initializer : mxnet.initializer.Initializer, optional
        Initializer for the embeddings and subword embeddings matrix.
    sparse_grad : bool, default True
        Specifies mxnet.gluon.nn.Embedding sparse_grad argument.

    """

    def __init__(self, token_to_idx, subword_function, embedding_size,
                 weight_initializer=None, sparse_grad=True, **kwargs):
        super(FasttextEmbeddingModel,
              self).__init__(embedding_size=embedding_size, **kwargs)
        self.token_to_idx = token_to_idx
        self.subword_function = subword_function
        self.weight_initializer = weight_initializer
        self.sparse_grad = sparse_grad

        with self.name_scope():
            self.embedding = nn.Embedding(
                len(token_to_idx),
                embedding_size,
                weight_initializer=weight_initializer,
                sparse_grad=sparse_grad,
            )
            self.subword_embedding = _MaskedSumEmbedding(
                len(subword_function),
                embedding_size,
                weight_initializer=weight_initializer,
                sparse_grad=sparse_grad,
            )

    def hybrid_forward(self, F, words, wordsmask, subwords, subwordsmask):
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

    def _to_token_embedding_batch(self, batch, unknown_behavior, ctx):
        # Handle subwords
        subword_padding = Pad(pad_val=-1)
        subwords = self.subword_function(batch)
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
            if any(token not in self.token_to_idx for token in batch):
                raise ValueError
            words = [self.token_to_idx[token] for token in batch]
            mask = nd.ones(shape=(len(words), ), ctx=ctx)
        elif unknown_behavior == 'impute_raise':
            words = [
                self.token_to_idx[token] if token in self.token_to_idx else 0
                for token in batch
            ]
            mask = nd.array(
                [1 if token in self.token_to_idx else 0 for token in batch],
                ctx=ctx)
        else:
            raise RuntimeError(
                'Unsupported unknown_behavior {}'.format(unknown_behavior))

        words = nd.array(words, ctx=ctx)

        # Compute embeddings
        return self(words, mask, subwords, subwords_mask)
