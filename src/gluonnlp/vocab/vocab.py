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

import collections
import json
import uuid
import warnings

from mxnet import nd

from ..data.utils import DefaultLookupDict, count_tokens
from .. import _constants as C
from .. import embedding as emb

UNK_IDX = 0


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
        The representation for any unknown token. If `unknown_token` is not
        `None`, looking up any token that is not part of the vocabulary and
        thus considered unknown will return the index of `unknown_token`. If
        None, looking up an unknown token will result in `KeyError`.
    padding_token : hashable object or None, default '<pad>'
        The representation for the special token of padding token.
    bos_token : hashable object or None, default '<bos>'
        The representation for the special token of beginning-of-sequence token.
    eos_token : hashable object or None, default '<eos>'
        The representation for the special token of end-of-sequence token.
    reserved_tokens : list of hashable objects or None, default None
        A list specifying additional tokens to be added to the vocabulary.
        `reserved_tokens` must not contain the value of `unknown_token` or
        duplicate tokens. It must neither contain special tokens specified via
        keyword arguments.
    token_to_idx : dict mapping tokens (hashable objects) to int or None, default None
        If not `None`, specifies the indices of tokens to be used by the
        vocabulary. Each token in `token_to_index` must be part of the Vocab
        and each index can only be associated with a single token.
        `token_to_idx` is not required to contain a mapping for all tokens. For
        example, it is valid to only set the `unknown_token` index to 10
        (instead of the default of 0) with `token_to_idx = {'<unk>': 10}`,
        assuming that there are at least 10 tokens in the vocabulary.
    **kwargs
        Keyword arguments of the format `xxx_token` can be used to specify
        further special tokens that will be exposed as attribute of the
        vocabulary and associated with an index.
        For example, specifying `mask_token='<mask>` as additional keyword
        argument when constructing a vocabulary `v` leads to `v.mask_token`
        exposing the value of the special token: `<mask>`.
        If the specified token is not part of the Vocabulary, it will be added,
        just as if it was listed in the `reserved_tokens` argument. The
        specified tokens are listed together with reserved tokens in the
        `reserved_tokens` attribute of the vocabulary object.

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
    -etc-
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
    -etc-
    >>> my_vocab.set_embedding(glove)
    >>> my_vocab.embedding[['hello', 'world']][:, :5]
    <BLANKLINE>
    [[-0.38497   0.80092   0.064106 -0.28355  -0.026759]
     [-0.41486   0.71848  -0.3045    0.87445   0.22441 ]]
    <NDArray 2x5 @cpu(0)>

    Extra keyword arguments of the format `xxx_token` are used to expose
    specified tokens as attributes.

    >>> my_vocab2 = gluonnlp.Vocab(counter, special_token='hi')
    >>> my_vocab2.special_token
    'hi'

    With the `token_to_idx` argument the order of the `Vocab`'s index can be
    adapted. For example, `Vocab` assigns the index `0` to the `unknown_token`
    by default. With the `token_to_idx` argument, the default can be
    overwritten. Here we assign index `3` to the unknown token representation
    `<unk>`.

    >>> tok2idx = {'<unk>': 3}
    >>> my_vocab3 = gluonnlp.Vocab(counter, token_to_idx=tok2idx)
    >>> my_vocab3.unknown_token
    '<unk>'
    >>> my_vocab3[my_vocab3.unknown_token]
    3
    >>> my_vocab[my_vocab.unknown_token]
    0

    """

    def __init__(self, counter=None, max_size=None, min_freq=1, unknown_token=C.UNK_TOKEN,
                 padding_token=C.PAD_TOKEN, bos_token=C.BOS_TOKEN, eos_token=C.EOS_TOKEN,
                 reserved_tokens=None, token_to_idx=None, **kwargs):

        # Sanity checks.
        assert min_freq > 0, '`min_freq` must be set to a positive value.'

        # Set up idx_to_token and token_to_idx based on presence of unknown token
        self._unknown_token = unknown_token
        self._idx_to_token = [unknown_token] if unknown_token else []
        if unknown_token:
            self._token_to_idx = DefaultLookupDict(UNK_IDX)
        else:
            self._token_to_idx = {}

        kwargs['padding_token'] = padding_token
        kwargs['bos_token'] = bos_token
        kwargs['eos_token'] = eos_token

        # Handle special tokens
        special_tokens = []
        for special_token_name, special_token in kwargs.items():
            # Test if kwarg specifies a special token
            if not special_token_name.endswith('_token'):
                raise ValueError('{} is invalid. Only keyword arguments '
                                 'that end in \'_token\' are supported '
                                 'to declare special tokens.'.format(special_token_name))

            if special_token is not None and special_token not in special_tokens:
                special_tokens.append(special_token)

        if reserved_tokens is not None:
            special_tokens.extend(reserved_tokens)
            special_token_set = set(special_tokens)
            if unknown_token:
                assert unknown_token not in special_token_set, \
                    '`reserved_token` cannot contain `unknown_token`.'
            assert len(special_token_set) == len(special_tokens), \
                '`reserved_tokens` cannot contain duplicate reserved tokens or ' \
                'other special tokens.'

        if not special_tokens:
            self._reserved_tokens = None
        else:
            self._reserved_tokens = special_tokens
            self._idx_to_token.extend(special_tokens)

        self._token_to_idx.update((token, idx) for idx, token in enumerate(self._idx_to_token))
        self._embedding = None

        if counter:
            self._index_counter_keys(counter, unknown_token, special_tokens, max_size, min_freq)

        self._identifiers_to_tokens = kwargs
        if kwargs:
            self._expose_tokens_as_attributes(kwargs)

        if token_to_idx:
            self._sort_index_according_to_user_specification(token_to_idx)


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

    def _expose_tokens_as_attributes(self, identifiers_to_tokens):
        # This method must not be called before internal attributes accessed by
        # @properties getters are set. Otherwise the @properties may raise
        # during the hasattr(self, identifier) check

        for identifier, token in identifiers_to_tokens.items():
            # Special tokens are automatically added to the vocab; assert, just to be sure
            assert token is None or token in self
            if identifier.startswith('_'):
                raise ValueError('It is not allowed to use identifiers starting with '
                                 'underscore. In Python identifier names beginning with '
                                 'underscore are internal.')
            if hasattr(self, identifier):
                raise ValueError('vocab.{} already exists. '
                                 'Please choose a different identifier for token {}'
                                 .format(identifier, token))
            setattr(self, identifier, token)

    def _sort_index_according_to_user_specification(self, token_to_idx):
        # Sanity checks
        if not set(token_to_idx.keys()).issubset(self.token_to_idx.keys()):
            raise ValueError('User-specified token_to_idx mapping can only contain '
                             'tokens that will be part of the vocabulary.')
        if len(set(token_to_idx.values())) != len(token_to_idx):
            raise ValueError('User-specified indices must not contain duplicates.')
        if min(token_to_idx.values()) < 0 or max(token_to_idx.values()) >= len(self.token_to_idx):
            raise ValueError('User-specified indices must not be < 0 or >= the number of tokens '
                             'that will be in the vocabulary. The current vocab contains {}'
                             'tokens.'.format(len(self.token_to_idx)))

        # Update index ordering
        for token, new_idx in token_to_idx.items():
            old_idx = self.token_to_idx[token]
            ousted_token = self.idx_to_token[new_idx]

            self.token_to_idx[token] = new_idx
            self.token_to_idx[ousted_token] = old_idx
            self.idx_to_token[old_idx] = ousted_token
            self.idx_to_token[new_idx] = token

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
            assert embs.idx_to_vec is not None, \
                'For all specified `embeddings`, `embeddings.idx_to_vec` must be initialized. ' \
                'Use eg. `emb[emb.unknown_token] = nd.zeros(emsize)` to initialize, ' \
                'where `emsize` is the desired embedding dimensionality.'

        assert all([embs.unknown_token for embs in embeddings]) or \
            all([not embs.unknown_token for embs in embeddings]), \
            'Either all or none of the TokenEmbeddings must have an ' \
            'unknown_token set.'

        new_vec_len = sum(embs.idx_to_vec.shape[1] for embs in embeddings)
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

        self._embedding = emb.TokenEmbedding(self.unknown_token,
                                             init_unknown_vec=None,
                                             allow_extend=False,
                                             idx_to_token=self.idx_to_token,
                                             idx_to_vec=new_idx_to_vec)

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
        vocab_dict['identifiers_to_tokens'] = self._identifiers_to_tokens
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
        token_to_idx = vocab_dict.get('token_to_idx')
        unknown_token = vocab_dict.get('unknown_token')
        reserved_tokens = vocab_dict.get('reserved_tokens')
        identifiers_to_tokens = vocab_dict.get('identifiers_to_tokens', dict())

        special_tokens = {unknown_token}

        # Backward compatibility for explicit serialization of padding_token,
        # bos_token, eos_token handling in the json string as done in older
        # versions of GluonNLP.
        deprecated_arguments = ['padding_token', 'bos_token', 'eos_token']
        for token_name in deprecated_arguments:
            if token_name in vocab_dict:
                token = vocab_dict[token_name]
                assert token_name not in identifiers_to_tokens, 'Invalid json string. ' \
                    '{} was serialized twice.'.format(token_name)
                identifiers_to_tokens[token_name] = token

        # Separate reserved from special tokens
        special_tokens.update(identifiers_to_tokens.values())
        if reserved_tokens is not None:
            reserved_tokens = [
                t for t in reserved_tokens if t not in special_tokens
            ]

        # Backward compatiblity code to deserialize corrupted vocabularies
        # created without bugfix https://github.com/dmlc/gluon-nlp/pull/749
        corrected_token_to_idx = collections.defaultdict(list)
        idx_to_token = vocab_dict.get('idx_to_token')
        if len(idx_to_token) > len(token_to_idx):  # Index is corrupt
            warnings.warn(
                'Detected a corrupted index in the deserialize vocabulary. '
                'For versions before GluonNLP v0.7 the index is corrupted '
                'by specifying the same token for different special purposes, '
                'for example eos_token == padding_token. '
                'Deserializing the vocabulary nevertheless.'
            )
            for token, count in collections.Counter(idx_to_token).items():
                if count == 1:
                    continue
                # Introduce new tokens to avoid invalid duplicates
                idx = -1
                while count > 0:
                    count -= 1
                    idx = idx_to_token.index(token, idx + 1)
                    if idx == token_to_idx[token]:
                        # Valid idx
                        continue
                    else:
                        # Introduce temporary token
                        token_to_idx.update({str(uuid.uuid4()): idx})
                        corrected_token_to_idx[token].append(idx)

        vocab = cls(
            counter=count_tokens(token_to_idx.keys()),
            unknown_token=unknown_token,
            reserved_tokens=reserved_tokens,
            token_to_idx=token_to_idx,
            **identifiers_to_tokens)

        # Backward compatiblity code to deserialize corrupted vocabularies
        # created without bugfix https://github.com/dmlc/gluon-nlp/pull/749
        for token, corrected_idxs in corrected_token_to_idx.items():
            for idx in corrected_idxs:
                # delete temporary tokens
                del vocab._token_to_idx[vocab._idx_to_token[idx]]
                vocab._idx_to_token[idx] = token

        return vocab
