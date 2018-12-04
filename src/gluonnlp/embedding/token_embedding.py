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

# pylint: disable=consider-iterating-dictionary, too-many-lines

"""Text token embedding."""
from __future__ import absolute_import
from __future__ import print_function

__all__ = [
    'register', 'create', 'list_sources', 'TokenEmbedding', 'GloVe',
    'FastText', 'Word2Vec'
]

import io
import logging
import os
import warnings

import numpy as np
from mxnet import nd, registry, cpu
from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url

from .. import _constants as C
from ..data.utils import DefaultLookupDict, _get_home_dir
from ..model.train import FasttextEmbeddingModel


def register(embedding_cls):
    """Registers a new token embedding.


    Once an embedding is registered, we can create an instance of this embedding with
    :func:`~gluonnlp.embedding.create`.


    Examples
    --------
    >>> @gluonnlp.embedding.register
    ... class MyTextEmbed(gluonnlp.embedding.TokenEmbedding):
    ...     def __init__(self, source='my_pretrain_file'):
    ...         pass
    >>> embed = gluonnlp.embedding.create('MyTextEmbed')
    >>> print(type(embed))
    <class 'MyTextEmbed'>
    """

    register_text_embedding = registry.get_register_func(TokenEmbedding, 'token embedding')
    return register_text_embedding(embedding_cls)


def create(embedding_name, **kwargs):
    """Creates an instance of token embedding.


    Creates a token embedding instance by loading embedding vectors from an externally hosted
    pre-trained token embedding file, such as those of GloVe and FastText. To get all the valid
    `embedding_name` and `source`, use :func:`gluonnlp.embedding.list_sources`.


    Parameters
    ----------
    embedding_name : str
        The token embedding name (case-insensitive).
    kwargs : dict
        All other keyword arguments are passed to the initializer of token
        embedding class. For example `create(embedding_name='fasttext',
        source='wiki.simple', load_ngrams=True)` will return
        `FastText(source='wiki.simple', load_ngrams=True)`.


    Returns
    -------
    An instance of :class:`gluonnlp.embedding.TokenEmbedding`:
        A token embedding instance that loads embedding vectors from an externally hosted
        pre-trained token embedding file.
    """

    create_text_embedding = registry.get_create_func(TokenEmbedding, 'token embedding')
    return create_text_embedding(embedding_name, **kwargs)


def list_sources(embedding_name=None):
    """Get valid token embedding names and their pre-trained file names.


    To load token embedding vectors from an externally hosted pre-trained token embedding file,
    such as those of GloVe and FastText, one should use
    `gluonnlp.embedding.create(embedding_name, source)`. This method returns all the
    valid names of `source` for the specified `embedding_name`. If `embedding_name` is set to
    None, this method returns all the valid names of `embedding_name` with their associated
    `source`.


    Parameters
    ----------
    embedding_name : str or None, default None
        The pre-trained token embedding name.


    Returns
    -------
    dict or list:
        A list of all the valid pre-trained token embedding file names (`source`) for the
        specified token embedding name (`embedding_name`). If the text embeding name is set to None,
        returns a dict mapping each valid token embedding name to a list of valid pre-trained files
        (`source`). They can be plugged into
        `gluonnlp.embedding.create(embedding_name, source)`.
    """

    text_embedding_reg = registry.get_registry(TokenEmbedding)

    if embedding_name is not None:
        embedding_name = embedding_name.lower()
        if embedding_name not in text_embedding_reg:
            raise KeyError('Cannot find `embedding_name` {}. Use '
                           '`list_sources(embedding_name=None).keys()` to get all the valid'
                           'embedding names.'.format(embedding_name))
        return list(text_embedding_reg[embedding_name].source_file_hash.keys())
    else:
        return {embedding_name: list(embedding_cls.source_file_hash.keys())
                for embedding_name, embedding_cls in registry.get_registry(TokenEmbedding).items()}


class TokenEmbedding(object):
    """Token embedding base class.

    To load token embedding from an externally hosted pre-trained token embedding file, such as
    those of GloVe and FastText, use :func:`gluonnlp.embedding.create`.
    To get all the available `embedding_name` and `source`, use
    :func:`gluonnlp.embedding.list_sources`.

    Alternatively, to load embedding vectors from a custom pre-trained token embedding file, use
    :func:`gluonnlp.embedding.from_file`.

    If `unknown_token` is None, looking up unknown tokens results in KeyError.
    Otherwise, for every unknown token, if its representation `self.unknown_token` is encountered
    in the pre-trained token embedding file, index 0 of `self.idx_to_vec` maps to the pre-trained
    token embedding vector loaded from the file; otherwise, index 0 of `self.idx_to_vec` maps to
    the token embedding vector initialized by `init_unknown_vec`.

    If a token is encountered multiple times in the pre-trained token embedding file, only the
    first-encountered token embedding vector will be loaded and the rest will be skipped.

    Parameters
    ----------
    unknown_token : hashable object or None, default '<unk>'
        Any unknown token will be replaced by unknown_token and consequently
        will be indexed as the same representation. Only used if oov_imputer is
        not specified.
    init_unknown_vec : callback
        The callback used to initialize the embedding vector for the unknown
        token. Only used if `unknown_token` is not None.
    allow_extend : bool, default False
        If True, embedding vectors for previously unknown words can be added
        via token_embedding[tokens] = vecs. If False, only vectors for known
        tokens can be updated.
    unknown_lookup : object subscriptable with list of tokens returning nd.NDarray, default None
        If not None, unknown_lookup[tokens] is called for any unknown tokens.
        The result is cached if unknown_autoextend is True.
    unknown_autoextend : bool, default True
        If True, any unknown token for which a vector was looked up in
        unknown_lookup together with the resulting vector will be added to
        token_to_idx, idx_to_token and idx_to_vec, adding a new index. This
        option is ignored if allow_extend is False.

    """

    def __init__(self, unknown_token='<unk>', init_unknown_vec=nd.zeros,
                 allow_extend=False, unknown_lookup=None,
                 unknown_autoextend=True):
        self._unknown_token = unknown_token
        self._init_unknown_vec = init_unknown_vec
        self._allow_extend = allow_extend
        self._unknown_lookup = unknown_lookup
        self._unknown_autoextend = unknown_autoextend
        self._idx_to_token = [unknown_token] if unknown_token else []
        if unknown_token:
            self._token_to_idx = DefaultLookupDict(C.UNK_IDX)
        else:
            self._token_to_idx = {}
        self._token_to_idx.update((token, idx) for idx, token in enumerate(self._idx_to_token))
        self._idx_to_vec = None

    @staticmethod
    def _get_file_url(cls_name, source_file_hash, source):
        namespace = 'gluon/embeddings/{}'.format(cls_name)
        return _get_repo_file_url(namespace, source_file_hash[source][0])

    @classmethod
    def _get_file_path(cls, source_file_hash, embedding_root, source):
        cls_name = cls.__name__.lower()
        embedding_root = os.path.expanduser(embedding_root)
        url = cls._get_file_url(cls_name, source_file_hash, source)

        embedding_dir = os.path.join(embedding_root, cls_name)

        pretrained_file_name, expected_file_hash = source_file_hash[source]
        pretrained_file_path = os.path.join(embedding_dir, pretrained_file_name)

        if not os.path.exists(pretrained_file_path) \
           or not check_sha1(pretrained_file_path, expected_file_hash):
            print('Embedding file {} is not found. Downloading from Gluon Repository. '
                  'This may take some time.'.format(pretrained_file_name))
            download(url, pretrained_file_path, sha1_hash=expected_file_hash)

        return pretrained_file_path

    def _load_embedding(self, pretrained_file_path, elem_delim,
                        encoding='utf8'):
        """Load embedding vectors from a pre-trained token embedding file.

        Both text files and TokenEmbedding serialization files are supported.
        elem_delim and encoding are ignored for non-text files.

        For every unknown token, if its representation `self.unknown_token` is encountered in the
        pre-trained token embedding file, index 0 of `self.idx_to_vec` maps to the pre-trained token
        embedding vector loaded from the file; otherwise, index 0 of `self.idx_to_vec` maps to the
        text embedding vector initialized by `self._init_unknown_vec`.

        If a token is encountered multiple times in the pre-trained text embedding file, only the
        first-encountered token embedding vector will be loaded and the rest will be skipped.

        """

        pretrained_file_path = os.path.expanduser(pretrained_file_path)

        if not os.path.isfile(pretrained_file_path):
            raise ValueError('`pretrained_file_path` must be a valid path '
                             'to the pre-trained token embedding file.')

        logging.info('Loading pre-trained token embedding vectors from %s',
                     pretrained_file_path)

        if pretrained_file_path.endswith('.npz'):
            self._load_embedding_serialized(
                pretrained_file_path=pretrained_file_path)
        else:
            self._load_embedding_txt(
                pretrained_file_path=pretrained_file_path,
                elem_delim=elem_delim, encoding=encoding)

    def _load_embedding_txt(self, pretrained_file_path, elem_delim, encoding='utf8'):
        """Load embedding vectors from a pre-trained token embedding file.

        For every unknown token, if its representation `self.unknown_token` is encountered in the
        pre-trained token embedding file, index 0 of `self.idx_to_vec` maps to the pre-trained token
        embedding vector loaded from the file; otherwise, index 0 of `self.idx_to_vec` maps to the
        text embedding vector initialized by `self._init_unknown_vec`.

        If a token is encountered multiple times in the pre-trained text embedding file, only the
        first-encountered token embedding vector will be loaded and the rest will be skipped.
        """

        vec_len = None
        all_elems = []
        tokens = set()
        loaded_unknown_vec = None
        with io.open(pretrained_file_path, 'rb') as f:
            for line_num, line in enumerate(f):
                try:
                    line = line.decode(encoding)
                except ValueError:
                    warnings.warn('line {} in {}: failed to decode. Skipping.'
                                  .format(line_num, pretrained_file_path))
                    continue

                elems = line.rstrip().split(elem_delim)

                assert len(elems) > 1, 'line {} in {}: unexpected data format.'.format(
                    line_num, pretrained_file_path)

                token, elems = elems[0], [float(i) for i in elems[1:]]

                if token == self.unknown_token and loaded_unknown_vec is None:
                    loaded_unknown_vec = elems
                    tokens.add(self.unknown_token)
                elif token in tokens:
                    warnings.warn('line {} in {}: duplicate embedding found for '
                                  'token "{}". Skipped.'.format(line_num, pretrained_file_path,
                                                                token))
                elif len(elems) == 1 and line_num == 0:
                    warnings.warn('line {} in {}: skipped likely header line.'
                                  .format(line_num, pretrained_file_path))
                else:
                    if not vec_len:
                        vec_len = len(elems)
                        if self.unknown_token:
                            # Reserve a vector slot for the unknown token at the very beggining
                            # because the unknown token index is 0.
                            all_elems.extend([0] * vec_len)
                    else:
                        assert len(elems) == vec_len, \
                            'line {} in {}: found vector of inconsistent dimension for token ' \
                            '"{}". expected dim: {}, found: {}'.format(line_num,
                                                                       pretrained_file_path,
                                                                       token, vec_len, len(elems))
                    all_elems.extend(elems)
                    self._idx_to_token.append(token)
                    self._token_to_idx[token] = len(self._idx_to_token) - 1
                    tokens.add(token)

        self._idx_to_vec = nd.array(all_elems).reshape((-1, vec_len))

        if self.unknown_token:
            if loaded_unknown_vec is None:
                self._idx_to_vec[C.UNK_IDX] = self._init_unknown_vec(shape=vec_len)
            else:
                self._idx_to_vec[C.UNK_IDX] = nd.array(loaded_unknown_vec)

    def _load_embedding_serialized(self, pretrained_file_path):
        """Load embedding vectors from a pre-trained token embedding file.

        For every unknown token, if its representation `self.unknown_token` is encountered in the
        pre-trained token embedding file, index 0 of `self.idx_to_vec` maps to the pre-trained token
        embedding vector loaded from the file; otherwise, index 0 of `self.idx_to_vec` maps to the
        text embedding vector initialized by `self._init_unknown_vec`.

        ValueError is raised if a token occurs multiple times.
        """

        deserialized_embedding = TokenEmbedding.deserialize(pretrained_file_path)
        if deserialized_embedding.unknown_token:
            # Some .npz files on S3 may contain an unknown token and its
            # respective embedding. As a workaround, we assume that C.UNK_IDX
            # is the same now as it was when the .npz was generated. Under this
            # assumption we can safely overwrite the respective token and
            # vector from the npz.
            if deserialized_embedding.unknown_token:
                idx_to_token = deserialized_embedding.idx_to_token
                idx_to_vec = deserialized_embedding.idx_to_vec
                idx_to_token[C.UNK_IDX] = self.unknown_token
                if self._init_unknown_vec:
                    vec_len = idx_to_vec.shape[1]
                    idx_to_vec[C.UNK_IDX] = self._init_unknown_vec(shape=vec_len)
            else:
                # If the TokenEmbedding shall not have an unknown token, we
                # just delete the one in the npz.
                assert C.UNK_IDX == 0
                idx_to_token = deserialized_embedding.idx_to_token[C.UNK_IDX + 1:]
                idx_to_vec = deserialized_embedding.idx_to_vec[C.UNK_IDX + 1:]
        else:
            idx_to_token = deserialized_embedding.idx_to_token
            idx_to_vec = deserialized_embedding.idx_to_vec

        if not len(set(idx_to_token)) == len(idx_to_token):
            raise ValueError('Serialized embedding invalid. '
                             'It contains duplicate tokens.')

        if self.unknown_token:
            try:
                unknown_token_idx = deserialized_embedding.idx_to_token.index(
                    self.unknown_token)
                idx_to_token[C.UNK_IDX], idx_to_token[
                    unknown_token_idx] = idx_to_token[
                        unknown_token_idx], idx_to_token[C.UNK_IDX]
                idxs = [C.UNK_IDX, unknown_token_idx]
                idx_to_vec[idxs] = idx_to_vec[idxs[::-1]]
            except ValueError:
                vec_len = idx_to_vec.shape[1]
                idx_to_token.insert(0, self.unknown_token)
                idx_to_vec = nd.concat(
                    self._init_unknown_vec(shape=vec_len).reshape((1, -1)),
                    idx_to_vec, dim=0)

        self._idx_to_token = idx_to_token
        self._idx_to_vec = idx_to_vec
        self._token_to_idx.update((token, idx) for idx, token in enumerate(self._idx_to_token))

    @property
    def idx_to_token(self):
        """Index to token mapping.

        Returns
        -------
        list of str:
             A list of indexed tokens where the list indices and the token
             indices are aligned.

        """
        return self._idx_to_token

    @property
    def token_to_idx(self):
        """Token to index mapping.

        Returns
        -------
        dict of int to strs:
             A dictionary of tokens with their corresponding index numbers;
             inverse vocab.
        """
        return self._token_to_idx

    @property
    def idx_to_vec(self):
        """Index to vector mapping.

        Returns
        -------
        mxnet.ndarray.NDArray:
            For all the indexed tokens in this embedding, this NDArray maps
            each token's index to an embedding vector.

        """
        return self._idx_to_vec

    @property
    def unknown_token(self):
        """Unknown token representation.

        Any token that is unknown will be indexed using the representation of
        unknown_token.

        Returns
        -------
        hashable object or None:
            Unknown token representation

        """
        return self._unknown_token

    @property
    def allow_extend(self):
        """Allow extension of the TokenEmbedding with new tokens.

        If True, `TokenEmbedding[tokens] = vec` can introduce new tokens that
        were previously unknown. New indices will be assigned to the newly
        introduced tokens. If False, only known tokens can be updated.

        Returns
        -------
        bool:
            Extension of the TokenEmbedding is allowed.

        """
        return self._allow_extend


    @property
    def unknown_lookup(self):
        """Vector lookup for unknown tokens.

        If not None, unknown_lookup[tokens] is called for any unknown tokens.
        The result is cached if unknown_autoextend is True.

        Returns
        -------
        Mapping[List[str], nd.NDarray]
            Vector lookup mapping from tokens to vectors.

        """
        return self._unknown_lookup

    @unknown_lookup.setter
    def unknown_lookup(self, unknown_lookup):
        """Vector lookup for unknown tokens.

        If not None, unknown_lookup[tokens] is called for any unknown tokens.
        The result is cached if unknown_autoextend is True.

        Parameters
        ----------
        unknown_lookup : Mapping[List[str], nd.NDarray]
            Vector lookup mapping from tokens to vectors.

        """
        self._unknown_lookup = unknown_lookup

    @property
    def unknown_autoextend(self):
        """Autoextension behavior for unknown token lookup.

        If True, any unknown token for which a vector was looked up in
        unknown_lookup together with the resulting vector will be added to
        token_to_idx, idx_to_token and idx_to_vec, adding a new index. Applies
        only if unknown_lookup is not None.

        Returns
        -------
        bool
            Autoextension behavior
        """

        return self._unknown_autoextend

    def __contains__(self, token):
        """Check if token is known.

        Parameters
        ----------
        token : str
            A token.

        Returns
        -------
        bool:
            Return True if the token is known. A token is known if it has been
            assigned an index and vector.
        """
        return token in self._token_to_idx

    def __eq__(self, other):
        if isinstance(other, TokenEmbedding):
            return self.unknown_token == other.unknown_token \
                and self.idx_to_token == other.idx_to_token and \
                ((self.idx_to_vec == other.idx_to_vec).min().asscalar() == 1) \
                and (self._token_to_idx == other._token_to_idx)
        else:
            return NotImplemented

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        else:
            return not result

    def __getitem__(self, tokens):
        """Looks up embedding vectors of text tokens.

        Parameters
        ----------
        tokens : str or list of strs
            A token or a list of tokens.

        Returns
        -------
        mxnet.ndarray.NDArray:
            The embedding vector(s) of the token(s). According to numpy conventions, if `tokens` is
            a string, returns a 1-D NDArray (vector); if `tokens` is a list of
            strings, returns a 2-D NDArray (matrix) of shape=(len(tokens), vec_len).
        """

        to_reduce = not isinstance(tokens, (list, tuple))
        if to_reduce:
            tokens = [tokens]

        if self.unknown_lookup is not None and (not self.allow_extend
                                                or not self.unknown_autoextend):
            vecs = [
                self.idx_to_vec[self.token_to_idx[token]]
                if token in self.token_to_idx else self.unknown_lookup[token]
                for token in tokens
            ]
            vecs = nd.stack(*vecs, axis=0)
        else:
            if self.unknown_lookup is not None and self.allow_extend and self.unknown_autoextend:
                new_tokens = [t for t in tokens if t not in self.token_to_idx]
                self[new_tokens] = self.unknown_lookup[new_tokens]

            indices = [self._token_to_idx[token] for token in tokens]
            vecs = nd.Embedding(
                nd.array(indices), self.idx_to_vec, self.idx_to_vec.shape[0],
                self.idx_to_vec.shape[1])

        return vecs[0] if to_reduce else vecs

    def _check_vector_update(self, tokens, new_embedding):
        """Check that tokens and embedding are in the format for __setitem__."""
        assert self._idx_to_vec is not None, '`idx_to_vec` has not been initialized.'

        if not isinstance(tokens, (list, tuple)) or len(tokens) == 1:
            assert isinstance(new_embedding, nd.NDArray) and len(new_embedding.shape) in [1, 2], \
                '`new_embedding` must be a 1-D or 2-D NDArray if `tokens` is a single token.'
            if not isinstance(tokens, (list, tuple)):
                tokens = [tokens]
            if len(new_embedding.shape) == 1:
                new_embedding = new_embedding.expand_dims(0)

        else:
            assert isinstance(new_embedding, nd.NDArray) and len(new_embedding.shape) == 2, \
                '`new_embedding` must be a 2-D NDArray if `tokens` is a list of multiple strings.'
        if self._idx_to_vec is not None:
            assert new_embedding.shape == (len(tokens), self._idx_to_vec.shape[1]), \
                'The length of `new_embedding` must be equal to the number ' \
                'of tokens and the width of new_embedding must be equal ' \
                'to the dimension of embedding of the glossary.'
        else:
            assert new_embedding.shape[0] == len(tokens), \
                'The length of `new_embedding` must be equal to the number of tokens'
        return tokens

    def __setitem__(self, tokens, new_embedding):
        """Updates embedding vectors for tokens.

        If self.allow_extend is True, vectors for previously unknown tokens can be introduced.

        Parameters
        ----------
        tokens : hashable object or a list or tuple of hashable objects
            A token or a list of tokens whose embedding vector are to be updated.
        new_embedding : mxnet.ndarray.NDArray
            An NDArray to be assigned to the embedding vectors of `tokens`. Its length must be equal
            to the number of `tokens` and its width must be equal to the dimension of embedding of
            the glossary. If `tokens` is a singleton, it must be 1-D or 2-D. If `tokens` is a list
            of multiple strings, it must be 2-D.
        """
        if self.allow_extend and self._idx_to_vec is None:
            # Initialize self._idx_to_vec
            assert C.UNK_IDX == 0
            self._idx_to_vec = self._init_unknown_vec(shape=(1, new_embedding.shape[-1]))

        tokens = self._check_vector_update(tokens, new_embedding)

        if self.allow_extend:
            # Add new / previously unknown tokens
            for token in filter(lambda t: t not in self._token_to_idx, tokens):
                idx = len(self._token_to_idx)
                self._token_to_idx[token] = idx
                self._idx_to_token.append(token)

            num_extended = len(self._token_to_idx) - self.idx_to_vec.shape[0]
            if num_extended == 1:
                warnings.warn(
                    'When adding new tokens via TokenEmbedding.__setitem__ '
                    'the internal embedding matrix needs to be reallocated. '
                    'Users are therefore encouraged to batch their updates '
                    '(i.e. add multiple new tokens at a time).')

            # Extend shape of idx_to_vec
            idx_to_vec = nd.zeros(shape=(len(self._token_to_idx),
                                         self.idx_to_vec.shape[1]))
            idx_to_vec[:self.idx_to_vec.shape[0]] = self._idx_to_vec
            self._idx_to_vec = idx_to_vec

        indices = []
        for token in tokens:
            if token in self._token_to_idx:
                indices.append(self._token_to_idx[token])
            else:
                if self.unknown_token:
                    raise KeyError(('Token "{}" is unknown. To update the embedding vector for an'
                                    ' unknown token, please explicitly include "{}" as the '
                                    '`unknown_token` in `tokens`. This is to avoid unintended '
                                    'updates.').format(token, self._idx_to_token[C.UNK_IDX]))
                else:
                    raise KeyError(('Token "{}" is unknown. Updating the embedding vector for an '
                                    'unknown token is not allowed because `unknown_token` is not '
                                    'specified.').format(token))

        self._idx_to_vec[nd.array(indices)] = new_embedding

    @classmethod
    def _check_source(cls, source_file_hash, source):
        """Checks if a pre-trained token embedding source name is valid.


        Parameters
        ----------
        source : str
            The pre-trained token embedding source.
        """
        embedding_name = cls.__name__.lower()
        if source not in source_file_hash:
            raise KeyError('Cannot find pre-trained source {} for token embedding {}. '
                           'Valid pre-trained file names for embedding {}: {}'.format(
                               source, embedding_name, embedding_name,
                               ', '.join(source_file_hash.keys())))

    @staticmethod
    def from_file(file_path, elem_delim=' ', encoding='utf8', **kwargs):
        """Creates a user-defined token embedding from a pre-trained embedding file.


        This is to load embedding vectors from a user-defined pre-trained token embedding file.
        For example, if `elem_delim` = ' ', the expected format of a custom pre-trained token
        embedding file may look like:

        'hello 0.1 0.2 0.3 0.4 0.5\\\\nworld 1.1 1.2 1.3 1.4 1.5\\\\n'

        where embedding vectors of words `hello` and `world` are [0.1, 0.2, 0.3, 0.4, 0.5] and
        [1.1, 1.2, 1.3, 1.4, 1.5] respectively.


        Parameters
        ----------
        file_path : str
            The path to the user-defined pre-trained token embedding file.
        elem_delim : str, default ' '
            The delimiter for splitting a token and every embedding vector element value on the same
            line of the custom pre-trained token embedding file.
        encoding : str, default 'utf8'
            The encoding scheme for reading the custom pre-trained token embedding file.
        kwargs : dict
            All other keyword arguments are passed to the TokenEmbedding initializer.


        Returns
        -------
        instance of :class:`gluonnlp.embedding.TokenEmbedding`
            The user-defined token embedding instance.
        """
        embedding = TokenEmbedding(**kwargs)
        embedding._load_embedding(file_path, elem_delim=elem_delim, encoding=encoding)
        return embedding

    def serialize(self, file_path, compress=True):
        """Serializes the TokenEmbedding to a file specified by file_path.

        TokenEmbedding is serialized by converting the list of tokens, the
        array of word embeddings and other metadata to numpy arrays, saving all
        in a single (optionally compressed) Zipfile. See
        https://docs.scipy.org/doc/numpy-1.14.2/neps/npy-format.html for more
        information on the format.


        Parameters
        ----------
        file_path : str or file
            The path at which to create the file holding the serialized
            TokenEmbedding. If file is a string or a Path, the .npz extension
            will be appended to the file name if it is not already there.
        compress : bool, default True
            Compress the Zipfile or leave it uncompressed.

        """
        if self.unknown_lookup is not None:
            warnings.warn(
                'Serialization of `unknown_lookup` is not supported. '
                'Save it manually and pass the loaded lookup object '
                'during deserialization.')

        unknown_token = np.array(self.unknown_token)
        idx_to_token = np.array(self.idx_to_token, dtype='O')
        idx_to_vec = self.idx_to_vec.asnumpy()

        if not unknown_token:  # Store empty string instead of None
            unknown_token = ''
        else:
            assert unknown_token == idx_to_token[C.UNK_IDX]

        if not compress:
            np.savez(file=file_path, unknown_token=unknown_token,
                     idx_to_token=idx_to_token, idx_to_vec=idx_to_vec)
        else:
            np.savez_compressed(file=file_path, unknown_token=unknown_token,
                                idx_to_token=idx_to_token,
                                idx_to_vec=idx_to_vec)

    @classmethod
    def deserialize(cls, file_path, **kwargs):
        """Create a new TokenEmbedding from a serialized one.

        TokenEmbedding is serialized by converting the list of tokens, the
        array of word embeddings and other metadata to numpy arrays, saving all
        in a single (optionally compressed) Zipfile. See
        https://docs.scipy.org/doc/numpy-1.14.2/neps/npy-format.html for more
        information on the format.


        Parameters
        ----------
        file_path : str or file
            The path to a file that holds the serialized TokenEmbedding.
        kwargs : dict
            Keyword arguments are passed to the TokenEmbedding initializer.
            Useful for attaching unknown_lookup.
        """
        # idx_to_token is of dtype 'O' so we need to allow pickle
        npz_dict = np.load(file_path, allow_pickle=True)

        unknown_token = npz_dict['unknown_token']
        if not unknown_token:
            unknown_token = None
        else:
            if isinstance(unknown_token, np.ndarray):
                if unknown_token.dtype.kind == 'S':
                    unknown_token = unknown_token.tobytes().decode()
                else:
                    unknown_token = str(unknown_token)
        idx_to_token = npz_dict['idx_to_token'].tolist()
        idx_to_vec = nd.array(npz_dict['idx_to_vec'])

        embedding = cls(unknown_token=unknown_token, **kwargs)
        if unknown_token:
            assert unknown_token == idx_to_token[C.UNK_IDX]
            embedding._token_to_idx = DefaultLookupDict(C.UNK_IDX)
        else:
            embedding._token_to_idx = {}

        embedding._idx_to_token = idx_to_token
        embedding._idx_to_vec = idx_to_vec
        embedding._token_to_idx.update((token, idx) for idx, token in enumerate(idx_to_token))

        return embedding


@register
class GloVe(TokenEmbedding):
    """The GloVe word embedding.

    GloVe is an unsupervised learning algorithm for obtaining vector representations for words.
    Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and
    the resulting representations showcase interesting linear substructures of the word vector
    space. (Source from https://nlp.stanford.edu/projects/glove/)

    Reference:

    GloVe: Global Vectors for Word Representation.
    Jeffrey Pennington, Richard Socher, and Christopher D. Manning.
    https://nlp.stanford.edu/pubs/glove.pdf

    Website: https://nlp.stanford.edu/projects/glove/

    To get the updated URLs to the externally hosted pre-trained token embedding
    files, visit https://nlp.stanford.edu/projects/glove/

    License for pre-trained embedding: https://opendatacommons.org/licenses/pddl/

    Available sources

    .. runblock:: pycon

        >>> import warnings; warnings.filterwarnings('ignore');
        >>> import gluonnlp as nlp
        >>> nlp.embedding.list_sources('GloVe')

    Parameters
    ----------
    source : str, default 'glove.6B.50d'
        The name of the pre-trained token embedding file.
    embedding_root : str, default '$MXNET_HOME/embedding'
        The root directory for storing embedding-related files.
        MXNET_HOME defaults to '~/.mxnet'.
    kwargs
        All other keyword arguments are passed to
        `gluonnlp.embedding.TokenEmbedding`.

    Attributes
    ----------
    idx_to_vec : mxnet.ndarray.NDArray
        For all the indexed tokens in this embedding, this NDArray maps each token's index to an
        embedding vector.
    unknown_token : hashable object
        The representation for any unknown token. In other words, any unknown token will be indexed
        as the same representation.
    """

    # Map a pre-trained token embedding file and its SHA-1 hash.
    source_file_hash = C.GLOVE_NPZ_SHA1

    def __init__(self, source='glove.6B.50d',
                 embedding_root=os.path.join(_get_home_dir(), 'embedding'), **kwargs):
        self._check_source(self.source_file_hash, source)

        super(GloVe, self).__init__(**kwargs)
        pretrained_file_path = GloVe._get_file_path(self.source_file_hash, embedding_root, source)

        self._load_embedding(pretrained_file_path, elem_delim=' ')


@register
class FastText(TokenEmbedding):
    """The fastText word embedding.


    FastText is an open-source, free, lightweight library that allows users to learn text
    representations and text classifiers. It works on standard, generic hardware. Models can later
    be reduced in size to even fit on mobile devices. (Source from https://fasttext.cc/)


    References:

    Enriching Word Vectors with Subword Information.
    Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov.
    https://arxiv.org/abs/1607.04606

    Bag of Tricks for Efficient Text Classification.
    Armand Joulin, Edouard Grave, Piotr Bojanowski, and Tomas Mikolov.
    https://arxiv.org/abs/1607.01759

    FastText.zip: Compressing text classification models.
    Armand Joulin, Edouard Grave, Piotr Bojanowski, Matthijs Douze, Herve Jegou, and Tomas Mikolov.
    https://arxiv.org/abs/1612.03651

    For 'wiki.multi' embedding:
    Word Translation Without Parallel Data
    Alexis Conneau, Guillaume Lample, Marc'Aurelio Ranzato, Ludovic Denoyer, and Herve Jegou.
    https://arxiv.org/abs/1710.04087

    Website: https://fasttext.cc/

    To get the updated URLs to the externally hosted pre-trained token embedding files, visit
    https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md

    License for pre-trained embedding: https://creativecommons.org/licenses/by-sa/3.0/

    Available sources

    .. runblock:: pycon

        >>> import warnings; warnings.filterwarnings('ignore');
        >>> import gluonnlp as nlp
        >>> nlp.embedding.list_sources('FastText')


    Parameters
    ----------
    source : str, default 'wiki.simple'
        The name of the pre-trained token embedding file.
    embedding_root : str, default '$MXNET_HOME/embedding'
        The root directory for storing embedding-related files.
        MXNET_HOME defaults to '~/.mxnet'.
    load_ngrams : bool, default False
        Load vectors for ngrams so that computing vectors for OOV words is
        possible. This is disabled by default as it requires downloading an
        additional 2GB file containing the vectors for ngrams. Note that
        facebookresearch did not publish ngram vectors for all their models. If
        load_ngrams is True, but no ngram vectors are available for the chosen
        source this a RuntimeError is thrown. The ngram vectors are passed to
        the resulting TokenEmbedding as `unknown_lookup`.
    ctx : mx.Context, default mxnet.cpu()
        Context to load the FasttextEmbeddingModel for ngram vectors to. This
        parameter is ignored if load_ngrams is False.
    kwargs
        All other keyword arguments are passed to
        `gluonnlp.embedding.TokenEmbedding`.


    Attributes
    ----------
    idx_to_vec : mxnet.ndarray.NDArray
        For all the indexed tokens in this embedding, this NDArray maps each token's index to an
        embedding vector.
    unknown_token : hashable object
        The representation for any unknown token. In other words, any unknown token will be indexed
        as the same representation.
    """

    # Map a pre-trained token embedding file and its SHA-1 hash.
    source_file_hash = C.FAST_TEXT_NPZ_SHA1
    source_bin_file_hash = C.FAST_TEXT_BIN_SHA1

    def __init__(self, source='wiki.simple', embedding_root=os.path.join(
            _get_home_dir(), 'embedding'), load_ngrams=False, ctx=cpu(), **kwargs):
        self._check_source(self.source_file_hash, source)

        if load_ngrams:
            try:
                self._check_source(self.source_bin_file_hash, source)
            except KeyError:
                raise KeyError(
                    'No ngrams are available for {}. '
                    'Ngram features were published for the following embeddings: {}'.
                    format(source, ', '.join(self.source_bin_file_hash.keys())))

            pretrained_bin_file_path = FastText._get_file_path(self.source_bin_file_hash,
                                                               embedding_root, source)
            unknown_lookup = FasttextEmbeddingModel.load_fasttext_format(
                pretrained_bin_file_path, ctx=ctx)
        else:
            unknown_lookup = None

        super(FastText, self).__init__(unknown_lookup=unknown_lookup, **kwargs)
        pretrained_file_path = FastText._get_file_path(self.source_file_hash, embedding_root,
                                                       source)

        self._load_embedding(pretrained_file_path, elem_delim=' ')


@register
class Word2Vec(TokenEmbedding):
    """The Word2Vec word embedding.

    Word2Vec is an unsupervised learning algorithm for obtaining vector
    representations for words. Training is performed with continuous
    bag-of-words or skip-gram architecture for computing vector
    representations of words.

    References:

    [1] Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient
    Estimation of Word Representations in Vector Space. In Proceedings of
    Workshop at ICLR, 2013.

    [2] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey
    Dean. Distributed Representations of Words and Phrases and their
    Compositionality. In Proceedings of NIPS, 2013.

    [3] Tomas Mikolov, Wen-tau Yih, and Geoffrey Zweig. Linguistic Regularities
    in Continuous Space Word Representations. In Proceedings of NAACL HLT,
    2013.

    Website: https://code.google.com/archive/p/word2vec/

    License for pre-trained embedding: Unspecified

    Available sources

    .. runblock:: pycon

        >>> import warnings; warnings.filterwarnings('ignore');
        >>> import gluonnlp as nlp
        >>> nlp.embedding.list_sources('Word2Vec')

    Parameters
    ----------
    source : str, default 'GoogleNews-vectors-negative300'
        The name of the pre-trained token embedding file.
    embedding_root : str, default '$MXNET_HOME/embedding'
        The root directory for storing embedding-related files.
        MXNET_HOME defaults to '~/.mxnet'.
    kwargs
        All other keyword arguments are passed to
        `gluonnlp.embedding.TokenEmbedding`.

    Attributes
    ----------
    idx_to_vec : mxnet.ndarray.NDArray
        For all the indexed tokens in this embedding, this NDArray maps each token's index to an
        embedding vector.
    unknown_token : hashable object
        The representation for any unknown token. In other words, any unknown token will be indexed
        as the same representation.

    """

    # Map a pre-trained token embedding file and its SHA-1 hash.
    source_file_hash = C.WORD2VEC_NPZ_SHA1

    def __init__(self, source='GoogleNews-vectors-negative300',
                 embedding_root=os.path.join(_get_home_dir(), 'embedding'), **kwargs):
        self._check_source(self.source_file_hash, source)

        super(Word2Vec, self).__init__(**kwargs)
        pretrained_file_path = self._get_file_path(self.source_file_hash, embedding_root, source)

        self._load_embedding(pretrained_file_path, elem_delim=' ')
