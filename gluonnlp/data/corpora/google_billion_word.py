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
"""Google billion words dataset."""

__all__ = ['GBWStream']

import glob
import hashlib
import io
import os
import tarfile
import zipfile

from mxnet.gluon.utils import _get_repo_file_url, check_sha1, download

from ... import _constants as C
from ...vocab import Vocab
from ..stream import SimpleDatasetStream
from ..dataset import CorpusDataset
from ..utils import _get_home_dir


class _GBWStream(SimpleDatasetStream):
    def __init__(self, namespace, segment, bos, eos, skip_empty, root):
        """Directory layout:
           - root ($MXNET_HOME/datasets/gbw)
             - archive_file (1-billion-word-language-modeling-benchmark-r13output.tar.gz)
             - dir (1-billion-word-language-modeling-benchmark-r13output)
               - subdir (training-monolingual.tokenized.shuffled)
               - subdir (heldout-monolingual.tokenized.shuffled)
        """
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        self._dir = os.path.join(root, '1-billion-word-language-modeling-benchmark-r13output')
        self._namespace = 'gluon/dataset/{}'.format(namespace)
        subdir_name, pattern, data_hash = self._data_file[segment]
        self._subdir = os.path.join(self._dir, subdir_name)
        self._file_pattern = os.path.join(self._subdir, pattern)
        self._data_hash = data_hash
        self._get_data()
        sampler = 'sequential' if segment != 'train' else 'random'
        super(_GBWStream, self).__init__(
            dataset=CorpusDataset,
            file_pattern=self._file_pattern,
            skip_empty=skip_empty,
            bos=bos,
            eos=eos,
            file_sampler=sampler)

    def _get_data(self):
        archive_file_name, archive_hash = self._archive_data
        archive_file_path = os.path.join(self._root, archive_file_name)
        exists = False
        if os.path.exists(self._dir) and os.path.exists(self._subdir):
            # verify sha1 for all files in the subdir
            sha1 = hashlib.sha1()
            filenames = sorted(glob.glob(self._file_pattern))
            for filename in filenames:
                with open(filename, 'rb') as f:
                    while True:
                        data = f.read(1048576)
                        if not data:
                            break
                        sha1.update(data)
            if sha1.hexdigest() == self._data_hash:
                exists = True
        if not exists:
            # download archive
            if not os.path.exists(archive_file_path) or \
               not check_sha1(archive_file_path, archive_hash):
                download(_get_repo_file_url(self._namespace, archive_file_name),
                         path=self._root, sha1_hash=archive_hash)
            # extract archive
            with tarfile.open(archive_file_path, 'r:gz') as tf:
                tf.extractall(path=self._root)

    def _get_vocab(self):
        archive_file_name, archive_hash = self._archive_vocab
        vocab_file_name, vocab_hash = self._vocab_file
        namespace = 'gluon/dataset/vocab'
        root = self._root
        path = os.path.join(root, vocab_file_name)
        if not os.path.exists(path) or not check_sha1(path, vocab_hash):
            downloaded_path = download(_get_repo_file_url(namespace, archive_file_name),
                                       path=root, sha1_hash=archive_hash)

            with zipfile.ZipFile(downloaded_path, 'r') as zf:
                zf.extractall(path=root)
        return path

class GBWStream(_GBWStream):
    """1-Billion-Word word-level dataset for language modeling, from Google.

    The GBWSream iterates over CorpusDatasets(flatten=False).

    Source http://www.statmt.org/lm-benchmark

    License: Apache

    Parameters
    ----------
    segment : {'train', 'test'}, default 'train'
        Dataset segment.
    skip_empty : bool, default True
        Whether to skip the empty samples produced from sample_splitters. If False, `bos` and `eos`
        will be added in empty samples.
    bos : str or None, default None
        The token to add at the begining of each sentence. If None, nothing is added.
    eos : str or None, default '<eos>'
        The token to add at the end of each sentence. If None, nothing is added.
    root : str, default '$MXNET_HOME/datasets/gbw'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """
    def __init__(self, segment='train', skip_empty=True, bos=None, eos=C.EOS_TOKEN,
                 root=os.path.join(_get_home_dir(), 'datasets', 'gbw')):
        self._archive_data = ('1-billion-word-language-modeling-benchmark-r13output.tar.gz',
                              '4df859766482e12264a5a9d9fb7f0e276020447d')
        self._archive_vocab = ('gbw-ebb1a287.zip',
                               '63b335dcc27b6804d0a14acb88332d2602fe0f59')
        self._data_file = {'train': ('training-monolingual.tokenized.shuffled',
                                     'news.en-00*-of-00100',
                                     '5e0d7050b37a99fd50ce7e07dc52468b2a9cd9e8'),
                           'test': ('heldout-monolingual.tokenized.shuffled',
                                    'news.en.heldout-00000-of-00050',
                                    '0a8e2b7496ba0b5c05158f282b9b351356875445')}
        self._vocab_file = ('gbw-ebb1a287.vocab',
                            'ebb1a287ca14d8fa6f167c3a779e5e7ed63ac69f')
        super(GBWStream, self).__init__('gbw', segment, bos, eos, skip_empty, root)

    @property
    def vocab(self):
        path = self._get_vocab()
        with io.open(path, 'r', encoding='utf-8') as in_file:
            return Vocab.from_json(in_file.read())
