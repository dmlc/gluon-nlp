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

__all__ = ['Vocab', 'SubwordVocab']

import json
import logging
import warnings

from mxnet import nd, registry
import numpy as np

from .data.utils import DefaultLookupDict
from . import _constants as C
from . import embedding as emb


###############################################################################
# Token level vocabulary
###############################################################################
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
    idx_to_counts : numpy.ndarray
        A list of the counts of tokens that were passed during Vocab construction.
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

    >>> text_data = " hello world \\\\n hello nice world \\\\n hi world \\\\n"
    >>> counter = gluonnlp.data.count_tokens(text_data)
    >>> my_vocab = gluonnlp.Vocab(counter)
    >>> fasttext = gluonnlp.embedding.create('fasttext', source='wiki.simple.vec')
    >>> my_vocab.set_embedding(fasttext)
    >>> my_vocab.embedding[['hello', 'world']]
    [[  3.95669997e-01   2.14540005e-01  -3.53889987e-02  -2.42990002e-01
        ...
       -7.54180014e-01  -3.14429998e-01   2.40180008e-02  -7.61009976e-02]
     [  1.04440004e-01  -1.08580001e-01   2.72119999e-01   1.32990003e-01
        ...
       -3.73499990e-01   5.67310005e-02   5.60180008e-01   2.90190000e-02]]
    <NDArray 2x300 @cpu(0)>

    >>> my_vocab[['hello', 'world']]
    [5, 4]

    >>> input_dim, output_dim = my_vocab.embedding.idx_to_vec.shape
    >>> layer = gluon.nn.Embedding(input_dim, output_dim)
    >>> layer.initialize()
    >>> layer.weight.set_data(my_vocab.embedding.idx_to_vec)
    >>> layer(nd.array([5, 4]))
    [[  3.95669997e-01   2.14540005e-01  -3.53889987e-02  -2.42990002e-01
        ...
       -7.54180014e-01  -3.14429998e-01   2.40180008e-02  -7.61009976e-02]
     [  1.04440004e-01  -1.08580001e-01   2.72119999e-01   1.32990003e-01
        ...
       -3.73499990e-01   5.67310005e-02   5.60180008e-01   2.90190000e-02]]
    <NDArray 2x300 @cpu(0)>

    >>> glove = gluonnlp.embedding.create('glove', source='glove.6B.50d.txt')
    >>> my_vocab.set_embedding(glove)
    >>> my_vocab.embedding[['hello', 'world']]
    [[  -0.38497001  0.80092001
        ...
        0.048833    0.67203999]
     [  -0.41486001  0.71847999
        ...
       -0.37639001 -0.67541999]]
    <NDArray 2x50 @cpu(0)>

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
        self._idx_to_counts = [0] if unknown_token else []

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
                self._idx_to_counts.append(freq)
                self._token_to_idx[token] = len(self._idx_to_token) - 1

    @property
    def embedding(self):
        return self._embedding

    @property
    def idx_to_token(self):
        return self._idx_to_token

    @property
    def idx_to_counts(self):
        return self._idx_to_counts

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

        new_embedding = emb.TokenEmbedding(self.unknown_token)
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
        vocab_dict['idx_to_counts'] = self._idx_to_counts
        vocab_dict['token_to_idx'] = dict(self._token_to_idx)
        vocab_dict['reserved_tokens'] = self._reserved_tokens
        vocab_dict['unknown_token'] = self._unknown_token
        vocab_dict['padding_token'] = self._padding_token
        vocab_dict['bos_token'] = self._bos_token
        vocab_dict['eos_token'] = self._eos_token
        return json.dumps(vocab_dict)

    @staticmethod
    def from_json(json_str):
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
        vocab = Vocab(unknown_token=unknown_token)
        vocab._idx_to_token = vocab_dict.get('idx_to_token')
        vocab._idx_to_counts = vocab_dict.get('idx_to_counts')
        vocab._token_to_idx = vocab_dict.get('token_to_idx')
        if unknown_token:
            vocab._token_to_idx = DefaultLookupDict(vocab._token_to_idx[unknown_token],
                                                    vocab._token_to_idx)
        vocab._reserved_tokens = vocab_dict.get('reserved_tokens')
        vocab._padding_token = vocab_dict.get('padding_token')
        vocab._bos_token = vocab_dict.get('bos_token')
        vocab._eos_token = vocab_dict.get('eos_token')
        return vocab


###############################################################################
# Subword level vocabulary
###############################################################################
class SubwordVocab(object):
    """Precomputed token index to subword unit mapping.

    Parameters
    ----------
    idx_to_token : list of str
        Known tokens for which the subword units should be precomputed.
    subword_function
        Callable to map tokens to lists of subword indices. Can be created with
        nlp.vocab.create. List available subword functions with
        nlp.vocab.list_sources.

    """

    def __init__(self, idx_to_token, subword_function):
        self.idx_to_token = idx_to_token
        self.subword_function = subword_function

        self._precompute_idx_to_subwordidxs()

    def _precompute_idx_to_subwordidxs(self):
        # Precompute a idx to subwordidxs mapping to support fast lookup
        idx_to_subwordidxs = list(self.subword_function(self.idx_to_token))
        max_subwordidxs_len = max(len(s) for s in idx_to_subwordidxs)
        self.idx_to_subwordidxs_list = idx_to_subwordidxs

        # Padded max_subwordidxs_len + 1 so each row contains at least one -1
        # element which can be found by np.argmax below.
        self.idx_to_subwordidxs = np.stack(
            np.pad(b, (0, max_subwordidxs_len - len(b) + 1), \
                   constant_values=-1, mode='constant')
            for b in idx_to_subwordidxs).astype(np.float32)

        logging.info('Constructing subword vocabulary with %s. '
                     'The word with largest number of subwords '
                     'has %s subwords.', self.subword_function,
                     max_subwordidxs_len)

    def indices_to_subwordindices(self, indices):
        """Return list of lists of subwordindices for indices.

        Parameters
        ----------
        indices : iterable of int
            Token indices that should be mapped to subword indices.

        Returns
        -------
        List of variable length lists of subword indices.

        """

        return [self.idx_to_subwordidxs_list[i] for i in indices]

    def indices_to_subwordindices_mask(self, indices):
        """Return array of subwordindices for indices.

        A padded numpy array and a mask is returned. The mask is used as
        indices map to varying length subwords.

        Parameters
        ----------
        indices : list of int, numpy array or mxnet NDArray
            Token indices that should be mapped to subword indices.

        Returns
        -------
        Array of subword indices.

        """
        if isinstance(indices, nd.NDArray):
            indices = indices.asnumpy().astype(np.int)
        else:
            indices = np.array(indices, dtype=np.int)
        subwords = self.idx_to_subwordidxs[indices]
        mask = np.zeros_like(subwords)
        mask += subwords != -1
        subwords += subwords == -1
        lengths = np.argmax(subwords == -1, axis=1)

        new_length = max(np.max(lengths), 1)  # Return at least 1
        subwords = subwords[:, :new_length]
        mask = mask[:, :new_length]

        return subwords, mask

    def words_to_subwordindices(self, words):
        """Return list of lists of subwordindices for words.

        Parameters
        ----------
        indices : iterable of str
            Words that should be mapped to subword indices.

        Returns
        -------
        List of variable length lists of subword indices.

        """

        subwordindices = list(self.subword_function(words))
        return subwordindices

    def subwordindices_to_subwords(self, subwordindices):
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

        subwordindices = subwordindices
        return self.subword_function.indices_to_subwords(subwordindices)

    def subwords_to_subwordindices(self, subwords):
        """Return list of subwordindices associated with subwords.

        Parameters
        ----------
        subwords : iterable of str
            Subwords to replace by indices.

        Returns
        -------
        Iterable of int.

        """
        return self.subword_function.subwords_to_subwordindices(subwords)

    def __len__(self):
        return len(self.subword_function)

    def to_json(self):
        """Serialize subword vocab object to json string."""
        dict_ = {}
        try:
            dict_['subwordidx_to_subword'] = \
                self.subword_function.indices_to_subwords(
                    list(range(len(self.subword_function))))
        except RuntimeError:
            # Not all subword functions are invertible
            pass
        return json.dumps(dict_)


###############################################################################
# Subword functions and registry
###############################################################################
def register(subword_cls):
    """Registers a new subword function."""
    register_text_embedding = registry.get_register_func(
        _SubwordFunction, 'subword function')
    return register_text_embedding(subword_cls)


def create(subword_function_name, **kwargs):
    """Creates an instance of a subword function."""

    create_ = registry.get_create_func(_SubwordFunction, 'token embedding')
    return create_(subword_function_name, **kwargs)


def list_sources():
    """Get valid subword function names."""
    reg = registry.get_registry(_SubwordFunction)
    return list(reg.keys())


class _SubwordFunction(object):
    def __call__(self, words):
        """Return a generator over subwords in the given word."""
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


@register
class ByteSubwords(_SubwordFunction):
    """Map words to a list of bytes.

    Parameters
    ----------
    encoding : str, default 'utf-8
        Encoding to use for obtaining bytes.

    """
    def __init__(self, encoding='utf-8'):
        self.encoding = encoding

    def __call__(self, words):
        generator = (np.frombuffer(word.encode(self.encoding),
                                   dtype=np.uint8).astype(np.int_)
                     for word in words)
        return generator

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


@register
class NGramHashes(_SubwordFunction):
    """Map words to a list of hashes in a restricted domain.

    The hash function is the same as in
    https://github.com/facebookresearch/fastText

    Parameters
    ----------
    num_subwords : int
        Size of target set for the hash function.
    ngrams : list of int, default [3, 4, 5, 6]
        n-s for which to hash the ngrams

    """
    def __init__(self, num_subwords, ngrams=(3, 4, 5, 6)):
        self.num_subwords = num_subwords
        self.ngrams = ngrams

        # Information for __repr__
        self.ngrams = ngrams

    @staticmethod
    def fasttext_hash_asbytes(s, encoding='utf-8'):
        h = np.uint32(2166136261)
        s = s.encode(encoding)
        old_settings = np.seterr(all='ignore')
        for c in s:
            h = h ^ np.uint32(c)
            h = h * np.uint32(16777619)
        np.seterr(**old_settings)
        return h

    @staticmethod
    def _get_all_ngram_generator(words, ngrams):
        return ((('<' + word + '>')[i:i + N] for N in ngrams
                 for i in range((len(word) + 2) - N + 1)) for word in words)

    def __call__(self, words):
        generator = (np.array([
            self.fasttext_hash_asbytes(
                ('<' + word + '>')[i:i + N]) % self.num_subwords
            for N in self.ngrams for i in range((len(word) + 2) - N + 1)
        ]) for word in words)
        return generator

    def __len__(self):
        return self.num_subwords

    def __repr__(self):
        return ('NGramHashes(num_subwords={}, ngrams={})'.format(
            self.num_subwords, self.ngrams))

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
        return [
            self.fasttext_hash_asbytes(sw) % self.num_subwords
            for sw in subwords
        ]
