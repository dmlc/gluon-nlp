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
"""Load token embedding"""

__all__ = [
    'list_sources', 'load_embeddings', 'get_fasttext_model'
]

import io
import logging
import os
import warnings
import fasttext

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
        if embedding_name == 'fasttext.bin':
            return list(C.FAST_TEXT_BIN_SHA1.keys())
        if embedding_name not in text_embedding_reg:
            raise KeyError('Cannot find `embedding_name` {}. Use '
                           '`list_sources(embedding_name=None).keys()` to get all the valid'
                           'embedding names.'.format(embedding_name))
        return list(text_embedding_reg[embedding_name].keys())
    else:
        return {embedding_name: list(embedding_cls.keys())
                for embedding_name, embedding_cls in text_embedding_reg.items()}

def _append_unk_vecs(matrix, vocab_size):
    append_dim = vocab_size - len(matrix)
    assert append_dim in [0, 1], "Error occurs in the embedding file."
    if append_dim == 1:
        # there is no unknown_token in the embedding file
        mean = np.mean(found_vectors, axis=0, keepdims=True)
        std = np.std(found_vectors, axis=0, keepdims=True)
        vecs = np.random.randn(append_dim, dim).astype('float32') * std + mean
        return np.concatenate([matrix, vecs], axis=0)
    return matrix

def _load_embedding_txt(file_path, vocab, unknown_token):
    if vocab is not None:
        result = np.zeros(len(vocab), dtype=bool)
    else:
        result = []
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
        if vocab is None:
            matrix = []
        else: matrix = np.random.randn(len(vocab), dim).astype('float32')
        for idx, line in enumerate(f, start_idx):
            try:
                parts = line.strip().split()
                word = ''.join(parts[:-dim])
                nums = parts[-dim:]
                if vocab is None:
                    result.append(word)
                    matrix.append(np.fromstring(' '.join(nums), sep=' ', dtype='float32', count=dim))
                else:
                    if word == unknown_token and vocab.unk_token is not None:
                        word = vocab.unk_token
                    if word in vocab:
                        index = vocab[word]
                        matrix[index] = np.fromstring(' '.join(nums), sep=' ', dtype='float32', count=dim)
                        result[index] = True
            except Exception as e:
                logging.error("Error occurred at the {} line.".format(idx))
                raise e
    if vocab is None:
        result = Vocab(result, unk_token=unknown_token)
        matrix = _append_unk_vecs(np.array(matrix), len(result))
    return matrix, result

def _load_embedding_npz(file_path, vocab, unknown):
    if vocab is not None:
        result = np.zeros(len(vocab), dtype=bool)
    else:
        result = []
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
    if vocab is None:
        result = Vocab(idx_to_token, unk_token=unknown_token)
        idx_to_vec = _append_unk_vecs(idx_to_vec, len(result))
        return idx_to_vec, result
    else:
        matrix = np.random.randn(len(vocab), idx_to_vec.shape[-1]).astype('float32')
        for i, token in enumerate(vocab.all_tokens):
            if token == vocab.unk_token and unknown_token is not None:
                word = unknown_token
            else:
                word = token
            if word in token2idx:
                index = token2idx[word]
                matrix[i] = idx_to_vec[index]
                result[i] = True
        return matrix, result

def _get_file_url(cls_name, file_name):
    namespace = 'gluon/embeddings/{}'.format(cls_name)
    return _get_repo_file_url(namespace, file_name)

def _get_file_path(cls_name, file_name, file_hash):
    root_path = os.path.expanduser(os.path.join(get_home_dir(), 'embedding'))
    embedding_dir = os.path.join(root_path, cls_name)
    url = _get_file_url(cls_name, file_name)
    file_path = os.path.join(embedding_dir, file_name)
    if not os.path.exists(file_path) or not check_sha1(file_path, file_hash):
        logging.info('Embedding file {} is not found. Downloading from Gluon Repository. '
                        'This may take some time.'.format(file_name))
        download(url, file_path, sha1_hash=file_hash)
    return file_path

def _check_and_get_path(pretrained_name_or_dir):
    if os.path.exists(pretrained_name_or_dir):
        return pretrained_name_or_dir
    for cls_name, embedding_cls in text_embedding_reg.items():
        if pretrained_name_or_dir in embedding_cls:
            source = pretrained_name_or_dir
            file_name, file_hash = embedding_cls[source]
            return _get_file_path(cls_name, file_name, file_hash)

    return None

def load_embeddings(vocab=None, pretrained_name_or_dir='glove.6B.50d', unknown='<unk>',
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
    ['GoogleNews-vectors-negative300', 'freebase-vectors-skipgram1000', \
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
    vocab : gluonnlp.data.Vocab object, default None
        A vocabulary on which an embedding matrix is built.
        If `vocab` is `None`, then all tokens in the pretrained file will be used.
    pretrained_name_or_dir : str, default 'glove.6B.50d'
        A file path for a pretrained embedding file or the name of the pretrained token embedding file.
        This method would first check if it is a file path.
        If not, the method will load from cache or download.
    unknown : str, default '<unk>'
        To specify the unknown token in the pretrained file.
    unk_method : Callable, default None
        A function which receives `List[str]` and returns `numpy.ndarray`.
        The input of the function is a list of words which are in the `vocab`,
        but do not occur in the pretrained file.
        And the function is aimed to return an embedding matrix for these words.
        If `unk_method` is None, we generate vectors for these words,
        by sampling from normal distribution with the same std and mean of the embedding matrix.
        It is only useful when `vocab` is not `None`.

    Returns
    -------
    If `vocab` is `None`
        numpy.ndarray:
            An embedding matrix in the pretrained file.
        gluonnlp.data.Vocab:
            The vocabulary in the pretrained file.
    Otherwise,
        numpy.ndarray:
            An embedding matrix for the given vocabulary.
    """
    assert isinstance(vocab, (Vocab, type(None))), "Only gluonnlp.data.Vocab is supported."
    file_path = _check_and_get_path(pretrained_name_or_dir)
    if file_path is None:
        raise ValueError("Cannot recognize `{}`".format(pretrained_name_or_dir))

    if file_path.endswith('.npz'):
        matrix, result = _load_embedding_npz(file_path, vocab, unknown)
    else:
        matrix, result = _load_embedding_txt(file_path, vocab, unknown)
    dim = matrix.shape[-1]
    logging.info("Pre-trained embedding dim: {}".format(dim))
    if vocab is None:
        return matrix, result
    else:
        hit_flags = result
        total_hits = sum(hit_flags)
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

def get_fasttext_model(model_name_or_dir='cc.en.300'):
    """ Load fasttext model from the binaray file

    This method will load fasttext model binaray file from a given file path or remote sources,
    and return a `fasttext` model object. See `fasttext.cc` for more usage information.

    Available sources:
    ['wiki-news-300d-1M-subword', 'crawl-300d-2M-subword', \
     'cc.af.300', ..., 'cc.en.300', ..., 'wiki.aa', ..., 'wiki.en', ..., 'wiki.zu']
    Detailed sources can be founded by `gluonnlp.embedding.list_sources('FastText.bin')`

    Parameters
    ----------
    model_name_or_dir : str, default 'cc.en.300'
        A file path for a FastText binary file or the name of the FastText model.
        This method would first check if it is a file path.
        If not, the method will load from cache or download.

    Returns
    -------
    fasttext.FastText._FastText:
        A FastText model based on `fasttext` package.
    """
    if os.path.exists(model_name_or_dir):
        file_path = model_name_or_dir
    else:
        source = model_name_or_dir
        root_path = os.path.expanduser(os.path.join(get_home_dir(), 'embedding'))
        embedding_dir = os.path.join(root_path, 'fasttext')
        if source not in C.FAST_TEXT_BIN_SHA1:
            raise ValueError('Cannot recognize {} for the bin file'.format(source))
        file_name, file_hash = C.FAST_TEXT_BIN_SHA1[source]
        file_path = _get_file_path('fasttext', file_name, file_hash)
    return fasttext.load_model(file_path)

