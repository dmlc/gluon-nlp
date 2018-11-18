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

__all__ = ['EmbeddingModel', 'CSREmbeddingModel', 'FasttextEmbeddingModel']

import logging
import struct
import warnings

import numpy as np
from mxnet import cpu, nd
from mxnet.gluon import Block, HybridBlock

from ...base import _str_types
from ...vocab.subwords import create_subword_function


class EmbeddingModel(Block):
    """Abstract base class for embedding models for training.

    An embedding model is a Gluon block with additional __contains__ and
    __getitem__ support for computing embeddings given a string or list of
    strings. See the documentation of __contains__ and __getitem__ for details.

    """

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


class CSREmbeddingModel(EmbeddingModel, HybridBlock):
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
    output_dim : int
        Dimension of the dense embedding.
    weight_initializer : mxnet.initializer.Initializer, optional
        Initializer for the embeddings matrix.
    sparse_grad : bool, default True
        Specifies mxnet.gluon.nn.Embedding sparse_grad argument.
    dtype : str, default 'float32'
        dtype argument passed to gluon.nn.Embedding

    """

    def __init__(self, token_to_idx, output_dim, weight_initializer=None,
                 sparse_grad=True, dtype='float32', **kwargs):
        super(CSREmbeddingModel, self).__init__(**kwargs)
        assert isinstance(token_to_idx, dict)
        self._token_to_idx = token_to_idx
        self._kwargs = {
            'input_dim': len(token_to_idx), 'output_dim': output_dim,
            'dtype': dtype, 'sparse_grad': sparse_grad}
        grad_stype = 'row_sparse' if sparse_grad else 'default'
        self.weight = self.params.get(
            'weight', shape=(len(token_to_idx), output_dim),
            init=weight_initializer, dtype=dtype,
            allow_deferred_init=True, grad_stype=grad_stype)  # yapf: disable

    def hybrid_forward(self, F, words, weight):
        """Compute embedding of words in batch.

        Parameters
        ----------
        words : mx.nd.NDArray
            Array of token indices.

        """
        #pylint: disable=arguments-differ
        embeddings = F.sparse.dot(words, weight)
        return embeddings

    def __repr__(self):
        s = '{block_name}({input_dim} -> {output_dim}, {dtype})'
        return s.format(block_name=self.__class__.__name__, **self._kwargs)

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

        row = np.arange(len(tokens))
        col = np.array([self._token_to_idx[t] for t in tokens])
        x = nd.sparse.csr_matrix(
            (np.ones(len(row)), (row, col)),
            dtype=self._kwargs['dtype'],
            ctx=self.weight.list_ctx()[0],
            shape=(len(tokens), self.weight.shape[0]),
        )
        vecs = self(x)

        if squeeze:
            assert len(vecs) == 1
            return vecs[0].squeeze()
        else:
            return vecs


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
        with. token_to_idx is used for __getitem__ and __contains__. For
        initialization len(token_to_idx) is used to specify the size of the
        subword embedding matrix..
    subword_function : gluonnlp.vocab.SubwordFunction
        The subword function used to obtain the subword indices during training
        this model. The subword_function is used for __getitem__ and
        __contains__. For initialization len(subword_function) is used to
        specify the size of the subword embedding matrix..
    output_dim : int
        Dimension of embeddings.
    weight_initializer : mxnet.initializer.Initializer, optional
        Initializer for the embeddings and subword embeddings matrix.
    sparse_grad : bool, default True
        Specifies mxnet.gluon.nn.Embedding sparse_grad argument.
    dtype : str, default 'float32'
        dtype argument passed to gluon.nn.Embedding

    """
    FASTTEXT_FILEFORMAT_MAGIC = 793712314

    def __init__(self, token_to_idx, subword_function, output_dim,
                 weight_initializer=None, sparse_grad=True, dtype='float32',
                 **kwargs):
        super(FasttextEmbeddingModel, self).__init__(**kwargs)
        self._token_to_idx = token_to_idx
        self._subword_function = subword_function

        self._kwargs = {
            'num_words': len(token_to_idx),
            'num_subwords': len(subword_function), 'output_dim': output_dim,
            'dtype': dtype, 'sparse_grad': sparse_grad}
        self.weight_initializer = weight_initializer
        self.sparse_grad = sparse_grad
        self.dtype = dtype

        grad_stype = 'row_sparse' if sparse_grad else 'default'
        self.weight = self.params.get(
            'weight', shape=(len(token_to_idx) + len(subword_function), output_dim),
            init=weight_initializer, dtype=dtype,
            allow_deferred_init=True, grad_stype=grad_stype)  # yapf: disable

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

        subword_function = create_subword_function(
            'NGramHashes', num_subwords=matrix.shape[0] - len(idx_to_token),
            ngrams=list(range(minn, maxn + 1)), special_tokens={'</s>'})

        self = cls(token_to_idx, subword_function, output_dim=dim, **kwargs)

        self.initialize(ctx=ctx)
        self.weight.set_data(nd.array(matrix))

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

        return new_format, dim, bucket, minn, maxn

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

    def __repr__(self):
        s = '{block_name}({num_words} + {num_subwords} -> {output_dim}, {dtype})'
        return s.format(block_name=self.__class__.__name__, **self._kwargs)

    def __contains__(self, token):
        # supports computing vector for any str that is at least either in the
        # word level vocabulary or contains subwords
        return (token in self._token_to_idx
                or self._subword_function([token])[0])

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

        data = []
        row = []
        col = []
        subwords = self._subword_function(tokens)
        offset = len(self._token_to_idx)
        for i, (token, token_subwords) in enumerate(zip(tokens, subwords)):
            if token not in self:
                raise KeyError

            if token in self._token_to_idx:
                col.append(self._token_to_idx[token])
                num = 1 + len(token_subwords)
            else:
                num = len(token_subwords)
            data += [1.0 / num] * num
            row += [i] * num
            col += [s + offset for s in token_subwords]

        x = nd.sparse.csr_matrix(
            (data, (row, col)), shape=(len(tokens), self.weight.shape[0]),
            dtype=self.dtype, ctx=self.weight.list_ctx()[0])
        emb = self(x)

        if squeeze:
            return emb.squeeze()
        else:
            return emb

    def hybrid_forward(self, F, words, weight):
        """Compute embedding of words in batch.

        Parameters
        ----------
        words : mxnet.ndarray.sparse.CSRNDArray
            Sparse array containing weights for every word and subword index.
            Output is the weighted sum of word and subword embeddings.
        """
        #pylint: disable=arguments-differ
        embeddings = F.sparse.dot(words, weight)
        return embeddings
