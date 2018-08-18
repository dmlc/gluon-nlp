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

import logging
import struct
import warnings

import numpy as np
from mxnet import cpu, nd
from mxnet.gluon import Block, HybridBlock, nn

from ...base import _str_types
from ...vocab.subwords import create_subword_function


class EmbeddingModel(Block):
    """Abstract base class for embedding models for training.

    An embedding model is a Gluon block with helper methods to directly work
    with the textual token representation.

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


class SimpleEmbeddingModel(EmbeddingModel):
    """A trainable embedding model.

    This class is a simple wrapper around the mxnet.gluon.nn.Embedding. It
    trains independent embedding vectors for every token. It implements the
    `gluonnlp.model.train.EmbeddingModel` interface which provides convenient
    helper methods.

    Parameters
    ----------
    token_to_idx : dict
        token_to_idx mapping of the vocabulary that this model is to be trained
        with. token_to_idx is used for __getitem__ and __contains__. For
        initialization len(token_to_idx) is used to specify the size of the
        subword embedding matrix.
    embedding_size : int
        Dimension of embeddings.
    weight_initializer : mxnet.initializer.Initializer, optional
        Initializer for the embeddings matrix.
    sparse_grad : bool, default True
        Specifies mxnet.gluon.nn.Embedding sparse_grad argument.
    dtype : str, default 'float32'
        dtype argument passed to gluon.nn.Embedding

    """

    def __init__(self, token_to_idx, embedding_size, weight_initializer=None,
                 sparse_grad=True, dtype='float32', **kwargs):
        assert isinstance(token_to_idx, dict)

        super(SimpleEmbeddingModel,
              self).__init__(embedding_size=embedding_size, **kwargs)
        self.token_to_idx = token_to_idx
        self.weight_initializer = weight_initializer
        self.sparse_grad = sparse_grad
        self.dtype = dtype

        with self.name_scope():
            self.embedding = nn.Embedding(
                len(token_to_idx), embedding_size,
                weight_initializer=weight_initializer, sparse_grad=sparse_grad,
                dtype=dtype)

    def __call__(self, words, wordsmask=None):
        return super(SimpleEmbeddingModel, self).__call__(words, wordsmask)

    def forward(self, words, wordsmask=None):
        """Compute embedding of words in batch.

        Parameters
        ----------
        words : mx.nd.NDArray
            Array of token indices.
        wordsmask : mx.nd.NDArray
            Mask for embeddings returned by the word level embedding operator.

        """
        #pylint: disable=arguments-differ
        if wordsmask is not None:
            wordsmask = nd.expand_dims(wordsmask, axis=-1)
            return nd.broadcast_mul(self.embedding(words), wordsmask)
        else:
            return self.embedding(words)

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

        indices = nd.array([self.token_to_idx[t] for t in tokens],
                           ctx=self.embedding.weight.list_ctx()[0])
        vecs = self(indices)

        if squeeze:
            assert len(vecs) == 1
            return vecs[0].squeeze()
        else:
            return vecs


class _MaskedSumEmbedding(HybridBlock):
    def __init__(self, num_tokens, embedding_size, weight_initializer=None,
                 sparse_grad=True, dtype='float32', **kwargs):
        super(_MaskedSumEmbedding, self).__init__(**kwargs)
        self.num_tokens = num_tokens
        self.embedding_size = embedding_size
        self.weight_initializer = weight_initializer
        self.sparse_grad = sparse_grad
        self.dtype = dtype

        with self.name_scope():
            self.embedding = nn.Embedding(
                num_tokens, embedding_size,
                weight_initializer=weight_initializer, sparse_grad=sparse_grad,
                dtype=dtype)

    def hybrid_forward(self, F, x, mask):
        #pylint: disable=arguments-differ
        mask = F.expand_dims(mask, axis=-1)
        masked_embeddings = F.broadcast_mul(self.embedding(x), mask)
        return F.sum(masked_embeddings, axis=-2)


class FasttextEmbeddingModel(EmbeddingModel):
    """FastText embedding model.

    The FasttextEmbeddingModel combines a word level embedding matrix and a
    subword level embedding matrix. It implements the
    `gluonnlp.model.train.EmbeddingModel` interface which provides convenient
    functions.

    Parameters
    ----------
    token_to_idx : dict
        token_to_idx mapping of the vocabulary that this model is to be trained
        with. token_to_idx is used for __getitem__ and __contains__. For
        initialization len(token_to_idx) is used to specify the size of the
        subword embedding matrix..
    subword_function : gluonnlp.vocab.SubwordFunction
        The subword function used to obtain the subword indices during training
        this model. The subword_function is used for __getitem__ and
        __contains__. For initialization len(subword_function) is used to
        specify the size of the subword embedding matrix..
    embedding_size : int
        Dimension of embeddings.
    weight_initializer : mxnet.initializer.Initializer, optional
        Initializer for the embeddings and subword embeddings matrix.
    sparse_grad : bool, default True
        Specifies mxnet.gluon.nn.Embedding sparse_grad argument.
    dtype : str, default 'float32'
        dtype argument passed to gluon.nn.Embedding

    """
    FASTTEXT_FILEFORMAT_MAGIC = 793712314

    def __init__(self, token_to_idx, subword_function, embedding_size,
                 weight_initializer=None, sparse_grad=True, dtype='float32',
                 **kwargs):
        super(FasttextEmbeddingModel,
              self).__init__(embedding_size=embedding_size, **kwargs)
        self.token_to_idx = token_to_idx
        self.subword_function = subword_function
        self.weight_initializer = weight_initializer
        self.sparse_grad = sparse_grad
        self.dtype = dtype

        with self.name_scope():
            self.embedding = nn.Embedding(
                len(token_to_idx), embedding_size,
                weight_initializer=weight_initializer, sparse_grad=sparse_grad,
                dtype=dtype)
            self.subword_embedding = _MaskedSumEmbedding(
                len(subword_function), embedding_size,
                weight_initializer=weight_initializer, sparse_grad=sparse_grad,
                dtype=dtype)

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
        if len(token_to_idx) != len(idx_to_token):
            # If multiple tokens with invalid encoding were collapsed in a
            # single token due to replacement of invalid bytes with Unicode
            # replacement character
            warnings.warn(
                'There are duplicate tokens in the embedding file. '
                'This is likely due to decoding errors for some tokens, '
                'where invalid bytes were replaced by '
                'the Unicode replacement character. '
                'This affects {} tokens.'.format(
                    len(idx_to_token) - len(token_to_idx)))
            for _ in range(len(token_to_idx), len(idx_to_token)):
                # Add pseudo tokens to make sure length is the same
                token_to_idx[object()] = -1
        assert len(token_to_idx) == len(idx_to_token)

        word_idx_to_vec = nd.array(matrix[:len(idx_to_token)])
        subword_idx_to_vec = nd.array(matrix[len(idx_to_token):])
        subword_function = create_subword_function(
            'NGramHashes', num_subwords=subword_idx_to_vec.shape[0],
            ngrams=list(range(minn, maxn + 1)), special_tokens={'</s>'})

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
            warnings.warn((
                'Provided model contains labels (nlabels={})'
                'This indicates you are either not using a word embedding model '
                'or that the model was created with a buggy version of fasttext. '
                'Ignoring all labels.').format(nlabels))
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
            # 'surrogateescape' would be better but only available in Py3
            word = word_bytes.decode(encoding, errors='replace')
            _, entry_type = cls._struct_unpack(file_handle, '@qb')
            if entry_type:
                # Skip incorrectly included labels (affects wiki.fr)
                assert nlabels > 0
                continue
            idx_to_token.append(word)

        assert len(idx_to_token) == nwords, \
            'Mismatch between words in pre-trained model file ({} words), ' \
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

    def __contains__(self, token):
        # supports computing vector for any str that is at least either in the
        # word level vocabulary or contains subwords
        return (token in self.token_to_idx
                or self.subword_function([token])[0])

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

        ctx = self.embedding.weight.list_ctx()[0]
        for token in tokens:
            if token in self.token_to_idx:
                # Word is part of fastText model
                word = nd.array([self.token_to_idx[token]], ctx=ctx)
                wordmask = nd.ones_like(word)
            else:
                word = nd.array([0], ctx=ctx)
                wordmask = nd.zeros_like(word)
            subwords = nd.array(self.subword_function([token]), ctx=ctx)
            if subwords.shape[1]:
                vec = self(word, subwords, wordsmask=wordmask)
            elif token not in self.token_to_idx:
                assert token not in self  # Assert consistency with __contains__
                raise KeyError
            else:
                # Known tokens (eg. special token such as EOS) without subwords
                vec = self.embedding(word)

            vecs.append(vec)

        if squeeze:
            assert len(vecs) == 1
            return vecs[0].squeeze()
        else:
            return nd.concat(*vecs, dim=0)

    def __call__(self, words, subwords, wordsmask=None, subwordsmask=None,
                 words_to_unique_subwords_indices=None):
        return super(FasttextEmbeddingModel, self).__call__(
            words, subwords, wordsmask, subwordsmask,
            words_to_unique_subwords_indices)

    def forward(self, words, subwords, wordsmask=None, subwordsmask=None,
                words_to_unique_subwords_indices=None):
        """Compute embedding of words in batch.

        Parameters
        ----------
        words : mx.nd.NDArray
            Array of token indices.
        subwords : mx.nd.NDArray
            The subwords associated with the tokens in `words`. If
            words_to_unique_subwords_indices is specified may contain the
            subwords of the unique tokens in `words` with
            `words_to_unique_subwords_indices` containing the reverse mapping.
        wordsmask : mx.nd.NDArray, optional
            Mask for embeddings returned by the word level embedding operator.
        subwordsmask : mx.nd.NDArray, optional
            A mask for the subword embeddings looked up from `subwords`.
            Applied before sum reducing the subword embeddings.
        words_to_unique_subwords_indices : mx.nd.NDArray, optional
            Mapping from the position in the `words` array to the position in
            the words_to_unique_subwords_indices` array.

        """
        #pylint: disable=arguments-differ
        embeddings = self.embedding(words)
        if wordsmask is not None:
            wordsmask = nd.expand_dims(wordsmask, axis=-1)
            embeddings = nd.broadcast_mul(embeddings, wordsmask)
        else:
            wordsmask = 1

        if words_to_unique_subwords_indices is None:
            assert words.shape[0] == subwords.shape[0]

            if subwordsmask is None:
                subwordsmask = nd.ones_like(subwords)

            num_embeddings = \
                nd.sum(subwordsmask, axis=-1, keepdims=True) + wordsmask

            subword_embeddings = self.subword_embedding(subwords, subwordsmask)
            return nd.broadcast_div(embeddings + subword_embeddings,
                                    num_embeddings)

        else:
            if subwordsmask is None:
                subwordsmask = nd.ones_like(subwords)

            subword_embedding_weights = self.subword_embedding(
                subwords, subwordsmask)
            words_to_unique_subwords_indices = \
                words_to_unique_subwords_indices.reshape(words.shape)

            subword_embeddings = nd.Embedding(
                data=words_to_unique_subwords_indices,
                weight=subword_embedding_weights,
                input_dim=subword_embedding_weights.shape[0],
                output_dim=self.embedding_size)

            num_embeddings = nd.Embedding(
                data=words_to_unique_subwords_indices, weight=nd.sum(
                    subwordsmask, axis=-1, keepdims=True),
                input_dim=subword_embedding_weights.shape[0],
                output_dim=1).reshape(words.shape).expand_dims(-1) + wordsmask

            return nd.broadcast_div(embeddings + subword_embeddings,
                                    num_embeddings)
