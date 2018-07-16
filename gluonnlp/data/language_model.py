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
"""Language model datasets."""

__all__ = ['WikiText2', 'WikiText103', 'WikiText2Raw', 'WikiText103Raw', 'GBWStream']

import os
import zipfile
import hashlib
import glob
import shutil
import tarfile

from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url

from .. import _constants as C
from .dataset import LanguageModelDataset
from .stream import LanguageModelStream
from .registry import register
from .utils import _get_home_dir


class _WikiText(LanguageModelDataset):
    def __init__(self, namespace, segment, bos, eos, skip_empty, root,
                 **kwargs):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        self._namespace = 'gluon/dataset/{}'.format(namespace)
        self._segment = segment
        super(_WikiText, self).__init__(self._get_data(), bos=bos, eos=eos,
                                        skip_empty=skip_empty, **kwargs)

    def _get_data(self):
        archive_file_name, archive_hash = self._archive_file
        data_file_name, data_hash = self._data_file[self._segment]
        root = self._root
        path = os.path.join(root, data_file_name)
        if not os.path.exists(path) or not check_sha1(path, data_hash):
            downloaded_file_path = download(_get_repo_file_url(self._namespace, archive_file_name),
                                            path=root,
                                            sha1_hash=archive_hash)

            with zipfile.ZipFile(downloaded_file_path, 'r') as zf:
                for member in zf.namelist():
                    filename = os.path.basename(member)
                    if filename:
                        dest = os.path.join(root, filename)
                        with zf.open(member) as source, \
                                open(dest, 'wb') as target:
                            shutil.copyfileobj(source, target)
        return path


@register(segment=['train', 'val', 'test'])
class WikiText2(_WikiText):
    """WikiText-2 word-level dataset for language modeling, from Salesforce research.

    From
    https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset

    License: Creative Commons Attribution-ShareAlike

    Parameters
    ----------
    segment : {'train', 'val', 'test'}, default 'train'
        Dataset segment.
    skip_empty : bool, default True
        Whether to skip the empty samples produced from sample_splitters. If False, `bos` and `eos`
        will be added in empty samples.
    tokenizer : function, default str.split
        A function that splits each sample string into list of tokens.
    bos : str or None, default None
        The token to add at the begining of each sentence. If None, nothing is added.
    eos : str or None, default '<eos>'
        The token to add at the end of each sentence. If None, nothing is added.
    root : str, default '$MXNET_HOME/datasets/wikitext-2'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """

    def __init__(self, segment='train', skip_empty=True,
                 tokenizer=lambda s: s.split(),
                 bos=None, eos=C.EOS_TOKEN, root=os.path.join(
                     _get_home_dir(), 'datasets', 'wikitext-2'), **kwargs):
        self._archive_file = ('wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')
        self._data_file = {'train': ('wiki.train.tokens',
                                     '863f29c46ef9d167fff4940ec821195882fe29d1'),
                           'val': ('wiki.valid.tokens',
                                   '0418625c8b4da6e4b5c7a0b9e78d4ae8f7ee5422'),
                           'test': ('wiki.test.tokens',
                                    'c7b8ce0aa086fb34dab808c5c49224211eb2b172')}
        super(WikiText2,
              self).__init__('wikitext-2', segment, bos, eos, skip_empty, root,
                             tokenizer=tokenizer, **kwargs)


@register(segment=['train', 'val', 'test'])
class WikiText103(_WikiText):
    """WikiText-103 word-level dataset for language modeling, from Salesforce research.

    From
    https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset

    License: Creative Commons Attribution-ShareAlike

    Parameters
    ----------
    segment : {'train', 'val', 'test'}, default 'train'
        Dataset segment.
    skip_empty : bool, default True
        Whether to skip the empty samples produced from sample_splitters. If False, `bos` and `eos`
        will be added in empty samples.
    tokenizer : function, default str.split
        A function that splits each sample string into list of tokens.
    bos : str or None, default None
        The token to add at the begining of each sentence. If None, nothing is added.
    eos : str or None, default '<eos>'
        The token to add at the end of each sentence. If None, nothing is added.
    root : str, default '$MXNET_HOME/datasets/wikitext-103'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """

    def __init__(self, segment='train', skip_empty=True,
                 tokenizer=lambda s: s.split(),
                 bos=None, eos=C.EOS_TOKEN, root=os.path.join(
                     _get_home_dir(), 'datasets', 'wikitext-103'), **kwargs):
        self._archive_file = ('wikitext-103-v1.zip',
                              '0aec09a7537b58d4bb65362fee27650eeaba625a')
        self._data_file = {
            'train': ('wiki.train.tokens',
                      'b7497e2dfe77e72cfef5e3dbc61b7b53712ac211'),
            'val': ('wiki.valid.tokens',
                    'c326ac59dc587676d58c422eb8a03e119582f92b'),
            'test': ('wiki.test.tokens',
                     '8a5befc548865cec54ed4273cf87dbbad60d1e47')
        }
        super(WikiText103,
              self).__init__('wikitext-103', segment, bos, eos, skip_empty,
                             root, tokenizer=tokenizer, **kwargs)


@register(segment=['train', 'val', 'test'])
class WikiText2Raw(_WikiText):
    """WikiText-2 character-level dataset for language modeling

    From Salesforce research:
    https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset

    License: Creative Commons Attribution-ShareAlike

    Parameters
    ----------
    segment : {'train', 'val', 'test'}, default 'train'
        Dataset segment.
    skip_empty : bool, default True
        Whether to skip the empty samples produced from sample_splitters. If False, `bos` and `eos`
        will be added in empty samples.
    tokenizer : function, default s.encode('utf-8')
        A function that splits each sample string into list of tokens.
        The tokenizer can also be used to convert everything to lowercase.
        E.g. with tokenizer=lambda s: s.lower().encode('utf-8')
    bos : str or None, default None
        The token to add at the begining of each sentence. If None, nothing is added.
    eos : str or None, default '<eos>'
        The token to add at the end of each sentence. If None, nothing is added.
    root : str, default '$MXNET_HOME/datasets/wikitext-2'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """

    def __init__(self, segment='train', skip_empty=True, bos=None, eos=None,
                 tokenizer=lambda s: s.encode('utf-8'),
                 root=os.path.join(_get_home_dir(), 'datasets',
                                   'wikitext-2'), **kwargs):
        self._archive_file = ('wikitext-2-raw-v1.zip',
                              '3b6993c138fc61c95f7fffd900fef68f8411371d')
        self._data_file = {
            'train': ('wiki.train.raw',
                      'd33faf256327882db0edc7c67cd098d1051a2112'),
            'val': ('wiki.valid.raw',
                    'db78d4db83700cba1b1bf4a9381087043db2876d'),
            'test': ('wiki.test.raw',
                     '6f1fe2054a940eebfc76b284b09680763b37f5ea')
        }

        super(WikiText2Raw,
              self).__init__('wikitext-2', segment, bos, eos, skip_empty, root,
                             tokenizer=tokenizer, **kwargs)


@register(segment=['train', 'val', 'test'])
class WikiText103Raw(_WikiText):
    """WikiText-103 character-level dataset for language modeling

    From Salesforce research:
    https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset

    License: Creative Commons Attribution-ShareAlike

    Parameters
    ----------
    segment : {'train', 'val', 'test'}, default 'train'
        Dataset segment.
    skip_empty : bool, default True
        Whether to skip the empty samples produced from sample_splitters. If False, `bos` and `eos`
        will be added in empty samples.
    tokenizer : function, default s.encode('utf-8')
        A function that splits each sample string into list of tokens.
        The tokenizer can also be used to convert everything to lowercase.
        E.g. with tokenizer=lambda s: s.lower().encode('utf-8')
    bos : str or None, default None
        The token to add at the begining of each sentence. If None, nothing is added.
    eos : str or None, default '<eos>'
        The token to add at the end of each sentence. If None, nothing is added.
    root : str, default '$MXNET_HOME/datasets/wikitext-103'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """

    def __init__(self, segment='train', skip_empty=True,
                 tokenizer=lambda s: s.encode('utf-8'), bos=None,
                 eos=None, root=os.path.join(_get_home_dir(), 'datasets',
                                             'wikitext-103'), **kwargs):
        self._archive_file = ('wikitext-103-raw-v1.zip',
                              '86f2375181b9247049d9c9205fad2b71b274b568')
        self._data_file = {
            'train': ('wiki.train.raw',
                      '3d06627c15e834408cfee91293f862c11c1cc9ef'),
            'val': ('wiki.valid.raw',
                    'db78d4db83700cba1b1bf4a9381087043db2876d'),
            'test': ('wiki.test.raw',
                     '6f1fe2054a940eebfc76b284b09680763b37f5ea')
        }
        super(WikiText103Raw,
              self).__init__('wikitext-103', segment, bos, eos, skip_empty,
                             root, tokenizer=tokenizer, **kwargs)

class _GBWStream(LanguageModelStream):
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
        super(_GBWStream, self).__init__(self._file_pattern, skip_empty=skip_empty, bos=bos,
                                         eos=eos)

    def _get_data(self):
        archive_file_name, archive_hash = self._archive_file
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

class GBWStream(_GBWStream):
    """1-Billion-Word word-level dataset for language modeling, from Google.

    From
    http://www.statmt.org/lm-benchmark

    License: Apache

    Parameters
    ----------
    segment : {'train',}, default 'train'
        Dataset segment.
    skip_empty : bool, default True
        Whether to skip the empty samples produced from sample_splitters. If False, `bos` and `eos`
        will be added in empty samples.
    bos : str or None, default '<bos>'
        The token to add at the begining of each sentence. If None, nothing is added.
    eos : str or None, default '<eos>'
        The token to add at the end of each sentence. If None, nothing is added.
    root : str, default '$MXNET_HOME/datasets/gbw'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """
    def __init__(self, segment='train', skip_empty=True, bos=C.BOS_TOKEN, eos=C.EOS_TOKEN,
                 root=os.path.join(_get_home_dir(), 'datasets', 'gbw')):
        assert segment == 'train', 'Only train segment is supported for GBW'
        self._archive_file = ('1-billion-word-language-modeling-benchmark-r13output.tar.gz',
                              '4df859766482e12264a5a9d9fb7f0e276020447d')
        self._data_file = {'train': ('training-monolingual.tokenized.shuffled',
                                     'news.en-00*-of-00100',
                                     '5e0d7050b37a99fd50ce7e07dc52468b2a9cd9e8')}
        super(GBWStream, self).__init__('gbw', segment, bos, eos, skip_empty, root)
