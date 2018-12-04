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

"""Vocabulary."""
from __future__ import absolute_import
from __future__ import print_function

__all__ = ['Vocab']

import json
import warnings

from mxnet import nd

from ..data.utils import DefaultLookupDict
from .. import _constants as C
from .. import embedding as emb


class Vocab(object):
    """Indexing and embedding attachment for text tokens.

    Parameters
    ----------
    counter : Counter or None, default None
        Counts text token frequencies in the text data. Its keys will be indexed according to
        frequency thresholds such as `max_size` and `min_freq`. Keys of `counter`,
        `unknown_token`, and values of `reserved_tokens` must be of the same hashable type.
        Examples: str, int, and tuple.
    max_size : None or int, default None
        The maximum possible number of the most frequent tokens in the keys of `counter` that can be
        indexed. Note that this argument does not count any token from `reserved_tokens`. Suppose
        that there are different keys of `counter` whose frequency are the same, if indexing all of
        them will exceed this argument value, such keys will be indexed one by one according to
        their __cmp__() order until the frequency threshold is met. If this argument is None or
        larger than its largest possible value restricted by `counter` and `reserved_tokens`, this
        argument has no effect.
    min_freq : int, default 1
        The minimum frequency required for a token in the keys of `counter` to be indexed.
    unknown_token : hashable object or None, default '<unk>'
        The representation for any unknown token. In other words, any unknown token will be indexed
        as the same representation. If None, looking up an unknown token will result in KeyError.
    padding_token : hashable object or None, default '<pad>'
        The representation for the special token of padding token.
    bos_token : hashable object or None, default '<bos>'
        The representation for the special token of beginning-of-sequence token.
    eos_token : hashable object or None, default '<eos>'
        The representation for the special token of end-of-sequence token.
    reserved_tokens : list of hashable objects or None, default None
        A list of reserved tokens (excluding `unknown_token`) that will always be indexed, such as
        special symbols representing padding, beginning of sentence, and end of sentence. It cannot
        contain `unknown_token` or duplicate reserved tokens. Keys of `counter`, `unknown_token`,
        and values of `reserved_tokens` must be of the same hashable type. Examples: str, int, and
        tuple.

    Attributes
    ----------
    embedding : instance of :class:`gluonnlp.embedding.TokenEmbedding`
        The embedding of the indexed tokens.
    idx_to_token : list of strs
        A list of indexed tokens where the list indices and the token indices are aligned.
    reserved_tokens : list of strs or None
        A list of reserved tokens that will always be indexed.
    token_to_idx : dict mapping str to int
        A dict mapping each token to its index integer.
    unknown_token : hashable object or None
        The representation for any unknown token. In other words, any unknown token will be indexed
        as the same representation.
    padding_token : hashable object or None
        The representation for padding token.
    bos_token : hashable object or None
        The representation for beginning-of-sentence token.
    eos_token : hashable object or None
        The representation for end-of-sentence token.


    Examples
    --------

    >>> text_data = ['hello', 'world', 'hello', 'nice', 'world', 'hi', 'world']
    >>> counter = gluonnlp.data.count_tokens(text_data)
    >>> my_vocab = gluonnlp.Vocab(counter)
    >>> fasttext = gluonnlp.embedding.create('fasttext', source='wiki.simple')
    >>> my_vocab.set_embedding(fasttext)
    >>> my_vocab.embedding[['hello', 'world']][:, :5]
    <BLANKLINE>
    [[ 0.39567   0.21454  -0.035389 -0.24299  -0.095645]
     [ 0.10444  -0.10858   0.27212   0.13299  -0.33165 ]]
    <NDArray 2x5 @cpu(0)>
    >>> my_vocab[['hello', 'world']]
    [5, 4]

    >>> input_dim, output_dim = my_vocab.embedding.idx_to_vec.shape
    >>> layer = gluon.nn.Embedding(input_dim, output_dim)
    >>> layer.initialize()
    >>> layer.weight.set_data(my_vocab.embedding.idx_to_vec)
    >>> layer(mx.nd.array([5, 4]))[:, :5]
    <BLANKLINE>
    [[ 0.39567   0.21454  -0.035389 -0.24299  -0.095645]
     [ 0.10444  -0.10858   0.27212   0.13299  -0.33165 ]]
    <NDArray 2x5 @cpu(0)>
    >>> glove = gluonnlp.embedding.create('glove', source='glove.6B.50d')
    >>> my_vocab.set_embedding(glove)
    >>> my_vocab.embedding[['hello', 'world']][:, :5]
    <BLANKLINE>
    [[-0.38497   0.80092   0.064106 -0.28355  -0.026759]
     [-0.41486   0.71848  -0.3045    0.87445   0.22441 ]]
    <NDArray 2x5 @cpu(0)>
    """

    def __init__(self, counter=None, max_size=None, min_freq=1, unknown_token=C.UNK_TOKEN,
                 padding_token=C.PAD_TOKEN, bos_token=C.BOS_TOKEN, eos_token=C.EOS_TOKEN,
                 reserved_tokens=None):

        # Sanity checks.
        assert min_freq > 0, '`min_freq` must be set to a positive value.'

        self._unknown_token = unknown_token
        special_tokens = []
        self._padding_token = padding_token
        if padding_token:
            special_tokens.append(padding_token)
        self._bos_token = bos_token
        if bos_token:
            special_tokens.append(bos_token)
        self._eos_token = eos_token
        if eos_token:
            special_tokens.append(eos_token)
        if reserved_tokens:
            special_tokens.extend(reserved_tokens)
            special_token_set = set(special_tokens)
            if unknown_token:
                assert unknown_token not in special_token_set, \
                    '`reserved_token` cannot contain `unknown_token`.'
            assert len(special_token_set) == len(special_tokens), \
                '`reserved_tokens` cannot contain duplicate reserved tokens or ' \
                'other special tokens.'
        self._index_special_tokens(unknown_token, special_tokens)

        if counter:
            self._index_counter_keys(counter, unknown_token, special_tokens, max_size, min_freq)

        self._embedding = None

    def _index_special_tokens(self, unknown_token, special_tokens):
        """Indexes unknown and reserved tokens."""
        self._idx_to_token = [unknown_token] if unknown_token else []

        if not special_tokens:
            self._reserved_tokens = None
        else:
            self._reserved_tokens = special_tokens[:]
            self._idx_to_token.extend(special_tokens)

        if unknown_token:
            self._token_to_idx = DefaultLookupDict(C.UNK_IDX)
        else:
            self._token_to_idx = {}
        self._token_to_idx.update((token, idx) for idx, token in enumerate(self._idx_to_token))

    def _index_counter_keys(self, counter, unknown_token, special_tokens, max_size,
                            min_freq):
        """Indexes keys of `counter`.


        Indexes keys of `counter` according to frequency thresholds such as `max_size` and
        `min_freq`.
        """

        unknown_and_special_tokens = set(special_tokens) if special_tokens else set()

        if unknown_token:
            unknown_and_special_tokens.add(unknown_token)

        token_freqs = sorted(counter.items(), key=lambda x: x[0])
        token_freqs.sort(key=lambda x: x[1], reverse=True)

        token_cap = len(unknown_and_special_tokens) + (
            len(counter) if not max_size else max_size)

        for token, freq in token_freqs:
            if freq < min_freq or len(self._idx_to_token) == token_cap:
                break
            if token not in unknown_and_special_tokens:
                self._idx_to_token.append(token)
                self._token_to_idx[token] = len(self._idx_to_token) - 1

    @property
    def embedding(self):
        return self._embedding

    @property
    def idx_to_token(self):
        return self._idx_to_token

    @property
    def reserved_tokens(self):
        return self._reserved_tokens

    @property
    def token_to_idx(self):
        return self._token_to_idx

    @property
    def unknown_token(self):
        return self._unknown_token

    @property
    def padding_token(self):
        return self._padding_token

    @property
    def bos_token(self):
        return self._bos_token

    @property
    def eos_token(self):
        return self._eos_token

    def __contains__(self, token):
        """Checks whether a text token exists in the vocabulary.


        Parameters
        ----------
        token : str
            A text token.


        Returns
        -------
        bool
            Whether the text token exists in the vocabulary (including `unknown_token`).
        """

        return token in self._token_to_idx

    def __getitem__(self, tokens):
        """Looks up indices of text tokens according to the vocabulary.

        If `unknown_token` of the vocabulary is None, looking up unknown tokens results in KeyError.

        Parameters
        ----------
        tokens : str or list of strs
            A source token or tokens to be converted.


        Returns
        -------
        int or list of ints
            A token index or a list of token indices according to the vocabulary.
        """

        if not isinstance(tokens, (list, tuple)):
            return self._token_to_idx[tokens]
        else:
            return [self._token_to_idx[token] for token in tokens]

    def __len__(self):
        return len(self._idx_to_token)

    def set_embedding(self, *embeddings):
        """Attaches one or more embeddings to the indexed text tokens.


        Parameters
        ----------
        embeddings : None or tuple of :class:`gluonnlp.embedding.TokenEmbedding` instances
            The embedding to be attached to the indexed tokens. If a tuple of multiple embeddings
            are provided, their embedding vectors will be concatenated for the same token.
        """

        if len(embeddings) == 1 and embeddings[0] is None:
            self._embedding = None
            return

        for embs in embeddings:
            assert isinstance(embs, emb.TokenEmbedding), \
                'The argument `embeddings` must be an instance or a list of instances of ' \
                '`gluonnlp.embedding.TokenEmbedding`.'

        assert all([embs.unknown_token for embs in embeddings]) or \
            all([not embs.unknown_token for embs in embeddings]), \
            'Either all or none of the TokenEmbeddings must have an ' \
            'unknown_token set.'

        new_embedding = emb.TokenEmbedding(self.unknown_token, allow_extend=False)
        new_embedding._token_to_idx = self.token_to_idx
        new_embedding._idx_to_token = self.idx_to_token

        new_vec_len = sum(embs.idx_to_vec.shape[1] for embs in embeddings
                          if embs and embs.idx_to_vec is not None)
        new_idx_to_vec = nd.zeros(shape=(len(self), new_vec_len))

        col_start = 0
        # Concatenate all the embedding vectors in embedding.
        for embs in embeddings:
            if embs and embs.idx_to_vec is not None:
                col_end = col_start + embs.idx_to_vec.shape[1]
                # Cancatenate vectors of the unknown token.
                new_idx_to_vec[0, col_start:col_end] = embs.idx_to_vec[0]
                new_idx_to_vec[1:, col_start:col_end] = embs[self._idx_to_token[1:]]
                col_start = col_end

        new_embedding._idx_to_vec = new_idx_to_vec
        self._embedding = new_embedding

    def to_tokens(self, indices):
        """Converts token indices to tokens according to the vocabulary.


        Parameters
        ----------
        indices : int or list of ints
            A source token index or token indices to be converted.


        Returns
        -------
        str or list of strs
            A token or a list of tokens according to the vocabulary.
        """

        to_reduce = False
        if not isinstance(indices, (list, tuple)):
            indices = [indices]
            to_reduce = True

        max_idx = len(self._idx_to_token) - 1

        tokens = []
        for idx in indices:
            if not isinstance(idx, int) or idx > max_idx:
                raise ValueError('Token index {} in the provided `indices` is invalid.'.format(idx))
            else:
                tokens.append(self._idx_to_token[idx])

        return tokens[0] if to_reduce else tokens

    def to_indices(self, tokens):
        """Looks up indices of text tokens according to the vocabulary.


        Parameters
        ----------
        tokens : str or list of strs
            A source token or tokens to be converted.


        Returns
        -------
        int or list of ints
            A token index or a list of token indices according to the vocabulary.
        """

        return self[tokens]

    def __call__(self, tokens):
        """Looks up indices of text tokens according to the vocabulary.


        Parameters
        ----------
        tokens : str or list of strs
            A source token or tokens to be converted.


        Returns
        -------
        int or list of ints
            A token index or a list of token indices according to the vocabulary.
        """

        return self[tokens]

    def __repr__(self):
        return 'Vocab(size={}, unk="{}", reserved="{}")'.format(len(self), self._unknown_token,
                                                                self._reserved_tokens)

    def to_json(self):
        """Serialize Vocab object to json string.

        This method does not serialize the underlying embedding.
        """
        if self._embedding:
            warnings.warn('Serialization of attached embedding '
                          'to json is not supported. '
                          'You may serialize the embedding to a binary format '
                          'separately using vocab.embedding.serialize')
        vocab_dict = {}
        vocab_dict['idx_to_token'] = self._idx_to_token
        vocab_dict['token_to_idx'] = dict(self._token_to_idx)
        vocab_dict['reserved_tokens'] = self._reserved_tokens
        vocab_dict['unknown_token'] = self._unknown_token
        vocab_dict['padding_token'] = self._padding_token
        vocab_dict['bos_token'] = self._bos_token
        vocab_dict['eos_token'] = self._eos_token
        return json.dumps(vocab_dict)

    @classmethod
    def from_json(cls, json_str):
        """Deserialize Vocab object from json string.

        Parameters
        ----------
        json_str : str
            Serialized json string of a Vocab object.


        Returns
        -------
        Vocab
        """
        vocab_dict = json.loads(json_str)

        unknown_token = vocab_dict.get('unknown_token')
        vocab = cls(unknown_token=unknown_token)
        vocab._idx_to_token = vocab_dict.get('idx_to_token')
        vocab._token_to_idx = vocab_dict.get('token_to_idx')
        if unknown_token:
            vocab._token_to_idx = DefaultLookupDict(vocab._token_to_idx[unknown_token],
                                                    vocab._token_to_idx)
        vocab._reserved_tokens = vocab_dict.get('reserved_tokens')
        vocab._padding_token = vocab_dict.get('padding_token')
        vocab._bos_token = vocab_dict.get('bos_token')
        vocab._eos_token = vocab_dict.get('eos_token')
        return vocab
