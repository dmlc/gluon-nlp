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

__all__ = [
    'list_sources', 'load_embeddings'
]

import io
import logging
import os
import warnings

import numpy as np
from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url

from . import _constants as C
from ..base import get_home_dir
from ..data import Vocab

text_embedding_reg = {
    'glove' : C.GLOVE_NPZ_SHA1,
    'word2vec' : C.WORD2VEC_NPZ_SHA1,
    'fasttext' : C.FAST_TEXT_NPZ_SHA1
}
def list_sources(embedding_name=None):
    """Get valid token embedding names and their pre-trained file names.

    Parameters
    ----------
    embedding_name : str or None, default None
        The pre-trained token embedding name.

    Returns
    -------
    dict or list:
        A list of all the valid pre-trained token embedding file names (`source`) for the
        specified token embedding name (`embedding_name`). If the text embedding name is set to
        None, returns a dict mapping each valid token embedding name to a list of valid pre-trained
        files (`source`).
    """
    if embedding_name is not None:
        embedding_name = embedding_name.lower()
        if embedding_name not in text_embedding_reg:
            raise KeyError('Cannot find `embedding_name` {}. Use '
                           '`list_sources(embedding_name=None).keys()` to get all the valid'
                           'embedding names.'.format(embedding_name))
        return list(text_embedding_reg[embedding_name].keys())
    else:
        return {embedding_name: list(embedding_cls.keys())
                for embedding_name, embedding_cls in text_embedding_reg.items()}

def _load_embedding_txt(file_path, vocab, unknown_token):
    hit_flags = np.zeros(len(vocab), dtype=bool)
    with open(file_path, 'r', encoding='utf-8') as f:
        line = f.readline().strip()
        parts = line.split()
        start_idx = 0
        if len(parts) == 2:
            dim = int(parts[1])
            start_idx += 1
        else:
            dim = len(parts) - 1
            f.seek(0)
        matrix = np.random.randn(len(vocab), dim).astype('float32')
        for idx, line in enumerate(f, start_idx):
            try:
                parts = line.strip().split()
                word = ''.join(parts[:-dim])
                nums = parts[-dim:]
                if word == unknown_token and vocab.unk_token is not None:
                    word = vocab.unk_token
                if word in vocab:
                    index = vocab[word]
                    matrix[index] = np.fromstring(' '.join(nums), sep=' ', dtype='float32', count=dim)
                    hit_flags[index] = True
            except Exception as e:
                logging.error("Error occurred at the {} line.".format(idx))
                raise e
    return matrix, hit_flags

def _load_embedding_npz(file_path, vocab, unknown):
    hit_flags = np.zeros(len(vocab), dtype=bool)
    npz_dict = np.load(file_path, allow_pickle=True)
    unknown_token = npz_dict['unknown_token']
    if not unknown_token:
        unknown_token = unknown
    else:
        if isinstance(unknown_token, np.ndarray):
            if unknown_token.dtype.kind == 'S':
                unknown_token = unknown_token.tobytes().decode()
            else:
                unknown_token = str(unknown_token)
    if unknown != unknown_token:
        warnings.warn("You may not assign correct unknown token in the pretrained file"
                      "Use {} as the unknown mark.".format(unknown_token))

    idx_to_token = npz_dict['idx_to_token'].tolist()
    token2idx = {x : i for i, x in enumerate(idx_to_token)}
    idx_to_vec = npz_dict['idx_to_vec']
    matrix = np.random.randn(len(vocab), idx_to_vec.shape[-1]).astype('float32')
    for token, i in vocab.token_to_idx.items():
        if token == vocab.unk_token and unknown_token is not None:
            word = unknown_token
        else:
            word = token
        if word in token2idx:
            index = token2idx[word]
            matrix[i] = idx_to_vec[index]
            hit_flags[i] = True
    return matrix, hit_flags

def _get_file_url(cls_name, file_name):
    namespace = 'gluon/embeddings/{}'.format(cls_name)
    return _get_repo_file_url(namespace, file_name)

def _check_and_get_path(pretrained_name_or_dir):
    if os.path.exists(pretrained_name_or_dir):
        return pretrained_name_or_dir
    root_path = os.path.expanduser(os.path.join(get_home_dir(), 'embedding'))
    for cls_name, embedding_cls in text_embedding_reg.items():
        if pretrained_name_or_dir in embedding_cls:
            source = pretrained_name_or_dir
            embedding_dir = os.path.join(root_path, cls_name)
            file_name, file_hash = embedding_cls[source]
            url = _get_file_url(cls_name, file_name)
            file_path = os.path.join(embedding_dir, file_name)
            if not os.path.exists(file_path) or not check_sha1(file_path, file_hash):
                logging.info('Embedding file {} is not found. Downloading from Gluon Repository. '
                             'This may take some time.'.format(file_name))
                download(url, file_path, sha1_hash=file_hash)
            return file_path

    return None

def load_embeddings(vocab, pretrained_name_or_dir='glove.6B.50d', unknown='<unk>',
                    unk_method=None):
    """Load pretrained word embeddings for building an embedding matrix for a given Vocab.

    This function supports loading GloVe, Word2Vec and FastText word embeddings from remote sources.
    You can also load your own embedding file(txt with Word2Vec or GloVe format) from a given file path.

    Glove: an unsupervised learning algorithm for obtaining vector representations for words.
    Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and
    the resulting representations showcase interesting linear substructures of the word vector
    space. (Source from https://nlp.stanford.edu/projects/glove/)
        Available sources:
            ['glove.42B.300d', 'glove.6B.100d', 'glove.6B.200d', 'glove.6B.300d', 'glove.6B.50d', \
             'glove.840B.300d', 'glove.twitter.27B.100d', 'glove.twitter.27B.200d', \
             'glove.twitter.27B.25d', 'glove.twitter.27B.50d']
    Word2Vec: an unsupervised learning algorithm for obtaining vector representations for words.
    Training is performed with continuous bag-of-words or skip-gram architecture for computing vector
    representations of words.
        Available sources:
            ['GoogleNews-vectors-negative300', 'freebase-vectors-skipgram1000',
             'freebase-vectors-skipgram1000-en']
    FastText: an open-source, free, lightweight library that allows users to learn text
    representations and text classifiers. It works on standard, generic hardware. Models can later
    be reduced in size to even fit on mobile devices. (Source from https://fasttext.cc/)
        Available sources:
            ['cc.af.300', ..., 'cc.en.300', ..., 'crawl-300d-2M', 'crawl-300d-2M-subword', \
             'wiki-news-300d-1M', 'wiki-news-300d-1M-subword', \
             'wiki.aa', ..., 'wiki.multi.ar', ..., 'wiki.zu']
            Detailed sources can be founded by `gluonnlp.embedding.list_sources('FastText')`
        For 'wiki.multi' embedding:

        Word Translation Without Parallel Data
        Alexis Conneau, Guillaume Lample, Marc'Aurelio Ranzato, Ludovic Denoyer, and Herve Jegou.
        https://arxiv.org/abs/1710.04087

    Parameters
    ----------
    vocab : gluonnlp.data.Vocab object, required
        Any unknown token will be replaced by unknown_token and consequently
        will be indexed as the same representation. Only used if oov_imputer is
        not specified.
    pretrained_name_or_dir : str, default 'glove.6B.50d'
        A file path for a pretrained embedding file or the name of the pretrained token embedding file.
        This method would first check if it is a file path.
        If not, the method will search it in the registry.
    unknown : str, default '<unk>'
        Unknown token in the pretrained file.
    unk_method : Callable, default None
        A function which receives `List[str]` and returns `numpy.ndarray`.
        The input of the function is a list of words which do not occur in the pretrained file.
        And the function is aimed to return an embedding matrix for these words.
        If `unk_method` is None, we generate vectors for these words,
        by sampling from normal distribution with the same std and mean of the embedding matrix.

    Returns
    -------
    numpy.ndarray:
        An embedding matrix for the given vocabulary.
    """
    assert isinstance(vocab, Vocab), "Only gluonnlp.data.Vocab is supported."
    file_path = _check_and_get_path(pretrained_name_or_dir)
    if file_path is None:
        raise ValueError("Cannot recognize `{}`".format(pretrained_name_or_dir))

    if file_path.endswith('.npz'):
        matrix, hit_flags = _load_embedding_npz(file_path, vocab, unknown)
    else:
        matrix, hit_flags = _load_embedding_txt(file_path, vocab, unknown)
    dim = matrix.shape[-1]
    total_hits = sum(hit_flags)
    logging.info("Pre-trained embedding dim: {}".format(dim))
    logging.info("Found {} out of {} words in the pretrained embedding.".format(total_hits, len(vocab)))
    if total_hits != len(vocab):
        if unk_method is None:
            found_vectors = matrix[hit_flags]
            mean = np.mean(found_vectors, axis=0, keepdims=True)
            std = np.std(found_vectors, axis=0, keepdims=True)
            unfound_vec_num = len(vocab) - total_hits
            r_vecs = np.random.randn(unfound_vec_num, dim).astype('float32') * std + mean
            matrix[hit_flags == False] = r_vecs
        else:
            unk_idxs = (hit_flags == False).nonzero()[0]
            matrix[hit_flags == False] = unk_method(vocab.to_tokens(unk_idxs))

    return matrix
