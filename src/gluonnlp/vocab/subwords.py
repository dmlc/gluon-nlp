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

# pylint: disable=consider-iterating-dictionary
"""Subword functions."""
from __future__ import absolute_import, print_function
import sys

import numpy as np
from mxnet import registry

from ..base import numba_njit

__all__ = [
    'SubwordFunction', 'ByteSubwords', 'NGramHashes',
    'register_subword_function', 'create_subword_function',
    'list_subword_functions'
]


def register_subword_function(subword_cls):
    """Registers a new subword function."""
    register_text_embedding = registry.get_register_func(SubwordFunction, 'subword function')
    return register_text_embedding(subword_cls)


def create_subword_function(subword_function_name, **kwargs):
    """Creates an instance of a subword function."""

    create_ = registry.get_create_func(SubwordFunction, 'token embedding')
    return create_(subword_function_name, **kwargs)


def list_subword_functions():
    """Get valid subword function names."""
    reg = registry.get_registry(SubwordFunction)
    return list(reg.keys())


class SubwordFunction(object):
    """A SubwordFunction maps words to lists of subword indices.

    This class is abstract and to be subclassed. Use
    gluonnlp.vocab.list_subword_functions to list all available subword
    functions.

    A SubwordFunction object is callable and returns a list of ndarrays of
    subwordindices for the given words in a call.

    """

    def __call__(self, words):
        """Return a list of ndarrays of subwordindices for the given words."""
        raise NotImplementedError

    def __len__(self):
        """Return the number of subwords modeled."""
        raise NotImplementedError

    def indices_to_subwords(self, indices):
        """Return list of subwords associated with subword indices.

        This may raise RuntimeError if the subword function is not invertible.

        Parameters
        ----------
        subwordindices : iterable of int
            Subword indices to look up.

        Returns
        -------
        Iterable of str.

        """
        raise NotImplementedError

    def subwords_to_indices(self, subwords):
        """Return list of subwordindices associated with subwords.

        Parameters
        ----------
        subwords : iterable of str
            Subwords to replace by indices.

        Returns
        -------
        Iterable of int.

        """
        raise NotImplementedError


@register_subword_function
class ByteSubwords(SubwordFunction):
    """Map words to a list of bytes.

    Parameters
    ----------
    encoding : str, default 'utf-8
        Encoding to use for obtaining bytes.

    """

    def __init__(self, encoding='utf-8'):
        self.encoding = encoding

    def __call__(self, words):
        if sys.version_info[0] == 3:
            return [list(word.encode(self.encoding)) for word in words]
        else:
            return [
                list((ord(c) for c in word.encode(self.encoding)))
                for word in words
            ]

    def __len__(self):
        return 256

    def __repr__(self):
        return 'ByteSubwords(encoding={})'.format(self.encoding)

    def indices_to_subwords(self, indices):
        """Return list of subwords associated with subword indices.

        This may raise RuntimeError if the subword function is not invertible.

        Parameters
        ----------
        subwordindices : iterable of int
            Subword indices to look up.

        Returns
        -------
        Iterable of str.

        """
        return indices

    def subwords_to_indices(self, subwords):
        """Return list of subwordindices associated with subwords.

        Parameters
        ----------
        subwords : iterable of str
            Subwords to replace by indices.

        Returns
        -------
        Iterable of int.

        """
        return subwords


@numba_njit
def _byte_to_int(b):
    return b


if sys.version_info[0] == 2:
    # Python 2 requires an extra `ord()` when operating on memoryview outside of numba

    try:
        import numba
        if numba.config.DISABLE_JIT:
            # JIT disabled is equivalent to numba not being present
            raise ImportError
    except ImportError:

        # pylint: disable=function-redefined
        @numba_njit
        def _byte_to_int(b):
            return ord(b)


@numba_njit
def _fasttext_ngram_hashes(word, ns, bucket_size):
    hashes = []
    max_n = np.max(ns)
    for i in range(len(word)):  # pylint: disable=consider-using-enumerate
        if (_byte_to_int(word[i]) & 0xC0) == 0x80:
            # Byte is continuation byte
            continue
        n = 0
        for j in range(i, len(word)):
            if (j + 1 < len(word)
                    and _byte_to_int(word[j + 1]) & 0xC0) == 0x80:
                # Next byte is continuation byte
                continue
            n += 1
            if (np.sum(n == ns)  # n in ns
                    and not (n == 1 and (i == 0 or j == len(word)))):
                ngram = word[i:j + 1]
                h = _fasttext_hash(ngram)
                hashes.append(h % bucket_size)
            if n >= max_n:
                break
    return hashes


@numba_njit
def _fasttext_hash(ngram):
    h = np.uint32(2166136261)
    for c in ngram:
        # Extra np.uint32 casts due to https://github.com/numba/numba/issues/3112
        h = np.uint32(h ^ np.uint32(np.int8(_byte_to_int(c))))
        h = np.uint32(h * np.uint32(16777619))
    return h


@register_subword_function
class NGramHashes(SubwordFunction):
    """Map words to a list of hashes in a restricted domain.

    The hash function is the same as in
    https://github.com/facebookresearch/fastText

    Parameters
    ----------
    num_subwords : int
        Size of target set for the hash function.
    ngrams : list of int, default [3, 4, 5, 6]
        n-s for which to hash the ngrams
    special_tokens : set of str, default None
        Set of words for which not to look up subwords.

    """

    def __init__(self, num_subwords, ngrams=(3, 4, 5, 6), special_tokens=None):
        self.num_subwords = num_subwords
        self.ngrams = ngrams
        self._ngrams = np.asarray(ngrams)

        assert not isinstance(special_tokens, str)
        if special_tokens is None:
            special_tokens = set()

        self.special_tokens = special_tokens

        # Information for __repr__
        self.ngrams = ngrams

    @staticmethod
    def fasttext_hash_asbytes(ngram, encoding='utf-8'):
        ngram_enc = memoryview(ngram.encode(encoding))
        return _fasttext_hash(ngram_enc)

    def _word_to_hashes(self, word):
        if word not in self.special_tokens:
            word_enc = bytearray((u'<' + word + u'>').encode('utf-8'))
            hashes = _fasttext_ngram_hashes(
                memoryview(word_enc), ns=self._ngrams,
                bucket_size=self.num_subwords)
        else:
            hashes = []
        return hashes

    def __call__(self, words):
        return [self._word_to_hashes(word) for word in words]

    def __len__(self):
        return self.num_subwords

    def __repr__(self):
        return ('NGramHashes(num_subwords={}, ngrams={})'.format(self.num_subwords, self.ngrams))

    def indices_to_subwords(self, indices):
        """Return list of subwords associated with subword indices.

        This may raise RuntimeError if the subword function is not invertible.

        Parameters
        ----------
        subwordindices : iterable of int
            Subword indices to look up.

        Returns
        -------
        Iterable of str.

        """
        raise RuntimeError('ngram hash function is not invertible.')

    def subwords_to_indices(self, subwords):
        """Return list of subwordindices associated with subwords.

        Parameters
        ----------
        subwords : iterable of str
            Subwords to replace by indices.

        Returns
        -------
        Iterable of int.

        """
        return [self.fasttext_hash_asbytes(sw) % self.num_subwords for sw in subwords]
