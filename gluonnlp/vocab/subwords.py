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

import numpy as np
from mxnet import nd, registry

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
        return [
            nd.array(np.frombuffer(word.encode(self.encoding), dtype=np.uint8).astype(np.int_))
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
    cache : bool, default True
        Cache the word to hash computation.

    """

    def __init__(self, num_subwords, ngrams=(3, 4, 5, 6), special_tokens=None, cache=True):
        self.num_subwords = num_subwords
        self.ngrams = ngrams

        if special_tokens is None:
            special_tokens = set()

        self.special_tokens = special_tokens

        self.cache = cache
        self._cache = {}

        # Information for __repr__
        self.ngrams = ngrams

    @staticmethod
    def fasttext_hash_asbytes(s, encoding='utf-8'):
        h = np.uint32(2166136261)
        s = s.encode(encoding)
        old_settings = np.seterr(all='ignore')
        for c in bytearray(s):
            h = h ^ np.uint32(np.int8(c))
            h = h * np.uint32(16777619)
        np.seterr(**old_settings)
        return h

    def _word_to_hashes(self, word):
        if word not in self._cache:
            if word not in self.special_tokens:
                hashes = nd.array([
                    self.fasttext_hash_asbytes((u'<' + word + u'>')[i:i + N]) % self.num_subwords
                    for N in self.ngrams for i in range((len(word) + 2) - N + 1)
                ])
            else:
                hashes = nd.zeros(shape=0)
            if self.cache:
                self._cache[word] = hashes
            return hashes
        return self._cache[word]

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
