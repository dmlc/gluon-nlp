
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

# pylint: disable=
"""Sentiment analysis datasets."""

__all__ = ['IMDB', 'MR', 'TREC', 'SUBJ', 'SST_1', 'SST_2']

import json
import os
import pickle
import tarfile

from mxnet.gluon.data import SimpleDataset
from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url

from .registry import register
from .utils import _get_home_dir

@register(segment=['train', 'test', 'unsup'])
class IMDB(SimpleDataset):
    """IMDB reviews for sentiment analysis.

    From
    http://ai.stanford.edu/~amaas/data/sentiment/

    Parameters
    ----------
    segment : str, default 'train'
        Dataset segment. Options are 'train', 'test', and 'unsup' for unsupervised.
    root : str, default '$MXNET_HOME/datasets/imdb'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """
    def __init__(self, segment='train', root=os.path.join(_get_home_dir(), 'datasets', 'imdb')):
        self._data_file = {'train': ('train.json',
                                     '516a0ba06bca4e32ee11da2e129f4f871dff85dc'),
                           'test': ('test.json',
                                    '7d59bd8899841afdc1c75242815260467495b64a'),
                           'unsup': ('unsup.json',
                                     'f908a632b7e7d7ecf113f74c968ef03fadfc3c6c')}
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        self._segment = segment
        self._get_data()
        super(IMDB, self).__init__(self._read_data())

    def _get_data(self):
        data_file_name, data_hash = self._data_file[self._segment]
        root = self._root
        path = os.path.join(root, data_file_name)
        if not os.path.exists(path) or not check_sha1(path, data_hash):
            download(_get_repo_file_url('gluon/dataset/imdb', data_file_name),
                     path=root, sha1_hash=data_hash)

    def _read_data(self):
        with open(os.path.join(self._root, self._segment + '.json')) as f:
            samples = json.load(f)
        return samples

class MR(SimpleDataset):
    """Movie reviews for sentiment analysis.

    From
    https://www.cs.cornell.edu/people/pabo/movie-review-data/

    Parameters
    ----------
    segment : str, default 'train'
        Dataset segment. Options are 'train' and 'test'.
    root : str, default '$MXNET_HOME/datasets/mr'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """
    def __init__(self, segment='train', root=os.path.join(_get_home_dir(), 'datasets', 'mr')):
        self._data_file = {'all': ('MR.tar.gz',
                                   'efadad45a46d280f4bd9084a386b71eaec16d0a8')}
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        self._segment = segment
        self._get_data()
        super(MR, self).__init__(self._read_data())

    def _get_data(self):
        data_file_name, data_hash = self._data_file['all']
        root = self._root
        path = os.path.join(root, data_file_name)
        if not os.path.exists(path) or not check_sha1(path, data_hash):
            fname = download(_get_repo_file_url('gluon/dataset/mr', data_file_name),
                             path=root, sha1_hash=data_hash)
            with tarfile.open(fname, 'r') as f:
                f.extractall(root)

    def _read_data(self):
        with open(os.path.join(self._root, self._segment + '_data.pkl'), 'rb') as f:
            samples = pickle.load(f)
        return samples

class TREC(SimpleDataset):
    """Question dataset for sentiment analysis.

    From
    http://cogcomp.cs.illinois.edu/Data/QA/QC/

    Parameters
    ----------
    segment : str, default 'train'
        Dataset segment. Options are 'train' and 'test'.
    root : str, default '$MXNET_HOME/datasets/trec'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """
    def __init__(self, segment='train', root=os.path.join(_get_home_dir(), 'datasets', 'trec')):
        self._data_file = {'all': ('TREC.tar.gz',
                                   'd3b6a01aa9ea8794631b1c2cc494ce3c3536e2b7')}
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        self._segment = segment
        self._get_data()
        super(TREC, self).__init__(self._read_data())

    def _get_data(self):
        data_file_name, data_hash = self._data_file['all']
        root = self._root
        path = os.path.join(root, data_file_name)
        if not os.path.exists(path) or not check_sha1(path, data_hash):
            fname = download(_get_repo_file_url('gluon/dataset/trec', data_file_name),
                             path=root, sha1_hash=data_hash)
            with tarfile.open(fname, 'r') as f:
                f.extractall(root)

    def _read_data(self):
        with open(os.path.join(self._root, self._segment + '_data.pkl'), 'rb') as f:
            samples = pickle.load(f)
        return samples

class SUBJ(SimpleDataset):
    """Subjectivity dataset for sentiment analysis.

    Parameters
    ----------
    segment : str, default 'train'
        Dataset segment. Options are 'train' and 'test'.
    root : str, default '$MXNET_HOME/datasets/subj'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """
    def __init__(self, segment='train', root=os.path.join(_get_home_dir(), 'datasets', 'subj')):
        self._data_file = {'all': ('Subj.tar.gz',
                                   '1b7886f337717779a2a9809763b0c2da9698d5f6')}
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        self._segment = segment
        self._get_data()
        super(SUBJ, self).__init__(self._read_data())

    def _get_data(self):
        data_file_name, data_hash = self._data_file['all']
        root = self._root
        path = os.path.join(root, data_file_name)
        if not os.path.exists(path) or not check_sha1(path, data_hash):
            fname = download(_get_repo_file_url('gluon/dataset/subj', data_file_name),
                             path=root, sha1_hash=data_hash)
            with tarfile.open(fname, 'r') as f:
                f.extractall(root)

    def _read_data(self):
        with open(os.path.join(self._root, self._segment + '_data.pkl'), 'rb') as f:
            samples = pickle.load(f)
        return samples

class SST_1(SimpleDataset):
    """Stanford Sentiment Treebank—an extension of MR.

    From
    http://nlp.stanford.edu/sentiment/

    Parameters
    ----------
    segment : str, default 'train'
        Dataset segment. Options are 'train' and 'test'.
    root : str, default '$MXNET_HOME/datasets/sst-1'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """
    def __init__(self, segment='train', root=os.path.join(_get_home_dir(), 'datasets', 'sst-1')):
        self._data_file = {'all': ('SST-1.tar.gz',
                                   '68a6f45ff891af973bd2182f598cd8d23a5c35f4')}
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        self._segment = segment
        self._get_data()
        super(SST_1, self).__init__(self._read_data())

    def _get_data(self):
        data_file_name, data_hash = self._data_file['all']
        root = self._root
        path = os.path.join(root, data_file_name)
        if not os.path.exists(path) or not check_sha1(path, data_hash):
            fname = download(_get_repo_file_url('gluon/dataset/sst-1', data_file_name),
                             path=root, sha1_hash=data_hash)
            with tarfile.open(fname, 'r') as f:
                f.extractall(root)

    def _read_data(self):
        with open(os.path.join(self._root, self._segment + '_data.pkl'), 'rb') as f:
            samples = pickle.load(f)
        return samples

class SST_2(SimpleDataset):
    """Stanford Sentiment Treebank—an extension of MR.

    Same as SST-1, but with neutral reviews removed and binary labels(positive, negative).

    From
    http://nlp.stanford.edu/sentiment/

    Parameters
    ----------
    segment : str, default 'train'
        Dataset segment. Options are 'train' and 'test'.
    root : str, default '$MXNET_HOME/datasets/sst-2'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """
    def __init__(self, segment='train', root=os.path.join(_get_home_dir(), 'datasets', 'sst-2')):
        self._data_file = {'all': ('SST-2.tar.gz',
                                   '8716e310d09658328701f76e5703c1a57df49a64')}
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        self._segment = segment
        self._get_data()
        super(SST_2, self).__init__(self._read_data())

    def _get_data(self):
        data_file_name, data_hash = self._data_file['all']
        root = self._root
        path = os.path.join(root, data_file_name)
        if not os.path.exists(path) or not check_sha1(path, data_hash):
            fname = download(_get_repo_file_url('gluon/dataset/sst-2', data_file_name),
                             path=root, sha1_hash=data_hash)
            with tarfile.open(fname, 'r') as f:
                f.extractall(root)

    def _read_data(self):
        with open(os.path.join(self._root, self._segment + '_data.pkl'), 'rb') as f:
            samples = pickle.load(f)
        return samples
