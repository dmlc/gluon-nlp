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

__all__ = [
    'EmbeddingModel', 'SimpleEmbeddingModel', 'FasttextEmbeddingModel',
    'DeduplicatedFasttextEmbeddingModel'
]

import logging
import struct
import sys

import numpy as np
from mxnet import cpu, nd
from mxnet.gluon import Block, HybridBlock, nn

from ... import embedding as emb
from ...data.batchify import Pad
from ...vocab import create_subword_function

if sys.version_info[0] == 3:
    _str_types = (str, )
else:
    _str_types = (str, unicode)


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

    def __contains__(self, token):
        """Checks if a vector for token could be computed.

        Parameters
        ----------
        token : str
            A token.

        Returns
        -------
        bool:
            True if a vector for token can be computed.
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        assert isinstance(token_to_idx, dict)

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


class _FasttextEmbeddingModel(EmbeddingModel):
    """Base class for FastText embedding models."""
    FASTTEXT_FILEFORMAT_MAGIC = 793712314

    def __init__(self, token_to_idx, subword_function, embedding_size,
                 weight_initializer=None, sparse_grad=True, **kwargs):
        super(_FasttextEmbeddingModel,
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

    @classmethod
    def load_fasttext_format(cls, path, ctx=cpu(), **kwargs):
        """Create an instance of the class and load weights.

        Load the weights from the fastText binary format created by
        https://github.com/facebookresearch/fastText

        Parameters
        ----------
        path : str
            Path to the .bin model file.
        ctx : mx.Context, default mx.cpu()
            Context to initialize the weights on.
        kwargs : dict
            Keyword arguments are passed to the class initializer.

        """
        with open(path, 'rb') as f:
            new_format, dim, bucket, minn, maxn, = cls._read_model_params(f)
            idx_to_token = cls._read_vocab(f, new_format)
            dim, matrix = cls._read_vectors(f, new_format, bucket,
                                            len(idx_to_token))

        token_to_idx = {token: idx for idx, token in enumerate(idx_to_token)}
        word_idx_to_vec = nd.array(matrix[:len(idx_to_token)])
        subword_idx_to_vec = nd.array(matrix[len(idx_to_token):])
        subword_function = create_subword_function(
            'NGramHashes', num_subwords=subword_idx_to_vec.shape[0],
            ngrams=list(range(minn, maxn + 1)), special_tokens='</s>')

        self = cls(token_to_idx, subword_function, embedding_size=dim,
                   **kwargs)

        self.initialize(ctx=ctx)
        self.embedding.weight.set_data(word_idx_to_vec)
        self.subword_embedding.embedding.weight.set_data(subword_idx_to_vec)

        return self

    @classmethod
    def _read_model_params(cls, file_handle):
        magic, _ = cls._struct_unpack(file_handle, '@2i')
        if magic == cls.FASTTEXT_FILEFORMAT_MAGIC:  # newer format
            new_format = True
            dim, _, _, _, _, _, _, _, bucket, minn, maxn, _, _ = \
                cls._struct_unpack(file_handle, '@12i1d')
        else:  # older format
            new_format = False
            dim = magic
            _, _, _, _, _, _, bucket, minn, maxn, _, _ = \
                cls._struct_unpack(file_handle, '@10i1d')

        return new_format, dim, bucket, minn, maxn,

    @classmethod
    def _read_vocab(cls, file_handle, new_format, encoding='utf8'):
        vocab_size, nwords, nlabels = cls._struct_unpack(file_handle, '@3i')
        if nlabels > 0:
            raise NotImplementedError(
                'Provided model is not a word embeddings model.')
        logging.info('Loading %s words from fastText model.', vocab_size)

        cls._struct_unpack(file_handle, '@1q')  # number of tokens
        if new_format:
            pruneidx_size, = cls._struct_unpack(file_handle, '@q')

        idx_to_token = []
        for _ in range(vocab_size):
            word_bytes = b''
            char_byte = file_handle.read(1)
            # Read vocab word
            while char_byte != b'\x00':
                word_bytes += char_byte
                char_byte = file_handle.read(1)
            word = word_bytes.decode(encoding)
            _, _ = cls._struct_unpack(file_handle, '@qb')

            idx_to_token.append(word)

        assert len(idx_to_token) == nwords, \
            'Mismatch between words in pretrained model file ({} words), ' \
            'and expected number of words ({} words)'.format(len(idx_to_token), nwords)

        if new_format:
            for _ in range(pruneidx_size):
                cls._struct_unpack(file_handle, '@2i')

        return idx_to_token

    @classmethod
    def _read_vectors(cls, file_handle, new_format, bucket, vocab_len):
        if new_format:
            # bool quant_input in fasttext.cc
            cls._struct_unpack(file_handle, '@?')
        num_vectors, dim = cls._struct_unpack(file_handle, '@2q')
        assert num_vectors == bucket + vocab_len

        # Vectors stored by Matrix::save
        float_size = struct.calcsize('@f')
        if float_size == 4:
            dtype = np.dtype(np.float32)
        elif float_size == 8:
            dtype = np.dtype(np.float64)

        vectors_ngrams = np.fromfile(file_handle, dtype=dtype,
                                     count=num_vectors * dim) \
                           .reshape((num_vectors, dim))

        return dim, vectors_ngrams

    @classmethod
    def _struct_unpack(cls, file_handle, fmt):
        num_bytes = struct.calcsize(fmt)
        return struct.unpack(fmt, file_handle.read(num_bytes))

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
            without_subwords_and_words = [
                batch[idx] for idx in without_subwords_idxs
                if batch[idx] not in self.token_to_idx
            ]
            if len(without_subwords_and_words):
                raise ValueError('No subwords were found for: ' +
                                 ', '.join(without_subwords_and_words))

        subwords = subwords.as_in_context(ctx)
        subwords_mask = subwords_mask.as_in_context(ctx)

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

    def __contains__(self, token):
        return True  # supports computing vector for any str

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

        vecs = []

        for token in tokens:
            if token in self.token_to_idx:
                # Word is part of fastText model
                word = nd.array([self.token_to_idx[token]])
                wordmask = nd.ones_like(word)
            else:
                word = nd.array([0])
                wordmask = nd.zeros_like(word)
            subwords = self.subword_function([token])[0]
            if subwords.shape[0]:
                vec = self(word, wordmask, subwords, nd.ones_like(subwords))
            else:
                # token is a special_token and subwords are not taken into account
                vec = self(word, wordmask, nd.zeros((1, 1)), nd.zeros((1, 1)))
                assert token in self.token_to_idx

            vecs.append(vec)

        if squeeze:
            assert len(vecs) == 1
            return vecs[0].squeeze()
        else:
            return nd.concat(*vecs, dim=0)


class FasttextEmbeddingModel(_FasttextEmbeddingModel, HybridBlock):
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
        num_embeddings = \
            F.sum(subwordsmask, axis=-1, keepdims=True) + wordsmask

        embeddings = F.broadcast_mul(self.embedding(words), wordsmask)
        subword_embeddings = self.subword_embedding(subwords, subwordsmask)

        return F.broadcast_div(embeddings + subword_embeddings, num_embeddings)


class DeduplicatedFasttextEmbeddingModel(_FasttextEmbeddingModel):
    """FastText embedding model with word deduplication.

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

    def forward(self, words, wordsmask, unique_subwords, unique_subwordsmask,
                words_to_unique_subwords_indices=None):
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
        wordsmask = nd.expand_dims(wordsmask, axis=-1)
        embeddings = nd.broadcast_mul(self.embedding(words), wordsmask)
        subword_embedding_weights = self.subword_embedding(
            unique_subwords, unique_subwordsmask)

        if words_to_unique_subwords_indices is None:
            assert words.shape[0] == unique_subwords.shape[0]
            words_to_unique_subwords_indices = nd.arange(
                words.shape[0], ctx=words.context)
        words_to_unique_subwords_indices = \
            words_to_unique_subwords_indices.reshape(words.shape)

        subword_embeddings = nd.Embedding(
            data=words_to_unique_subwords_indices,
            weight=subword_embedding_weights,
            input_dim=subword_embedding_weights.shape[0],
            output_dim=self.embedding_size)

        num_embeddings = nd.Embedding(
            data=words_to_unique_subwords_indices, weight=nd.sum(
                unique_subwordsmask, axis=-1,
                keepdims=True), input_dim=subword_embedding_weights.shape[0],
            output_dim=1).reshape(wordsmask.shape) + wordsmask

        return nd.broadcast_div(embeddings + subword_embeddings,
                                num_embeddings)
