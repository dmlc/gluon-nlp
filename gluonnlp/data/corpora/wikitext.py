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
"""WikiText corpora."""

__all__ = ['WikiText2', 'WikiText103', 'WikiText2Raw', 'WikiText103Raw']

import os
import shutil
import zipfile

from mxnet.gluon.utils import _get_repo_file_url, check_sha1, download

from ... import _constants as C
from ..dataset import CorpusDataset
from ..registry import register
from ..utils import _get_home_dir


class _WikiText(CorpusDataset):
    def __init__(self, namespace, segment, bos, eos, flatten, skip_empty, root,
                 **kwargs):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        self._namespace = 'gluon/dataset/{}'.format(namespace)
        self._segment = segment
        super(_WikiText, self).__init__(
            self._get_data(),
            bos=bos,
            eos=eos,
            flatten=flatten,
            skip_empty=skip_empty,
            **kwargs)

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

    WikiText2 is implemented as CorpusDataset with the default flatten=True.

    From
    https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset

    License: Creative Commons Attribution-ShareAlike

    Parameters
    ----------
    segment : {'train', 'val', 'test'}, default 'train'
        Dataset segment.
    flatten : bool, default True
        Whether to return all samples as flattened tokens. If True, each sample is a token.
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

    def __init__(self,
                 segment='train',
                 flatten=True,
                 skip_empty=True,
                 tokenizer=lambda s: s.split(),
                 bos=None,
                 eos=C.EOS_TOKEN,
                 root=os.path.join(_get_home_dir(), 'datasets', 'wikitext-2'),
                 **kwargs):
        self._archive_file = ('wikitext-2-v1.zip',
                              '3c914d17d80b1459be871a5039ac23e752a53cbe')
        self._data_file = {
            'train': ('wiki.train.tokens',
                      '863f29c46ef9d167fff4940ec821195882fe29d1'),
            'val': ('wiki.valid.tokens',
                    '0418625c8b4da6e4b5c7a0b9e78d4ae8f7ee5422'),
            'test': ('wiki.test.tokens',
                     'c7b8ce0aa086fb34dab808c5c49224211eb2b172')
        }
        super(WikiText2, self).__init__(
            'wikitext-2',
            segment=segment,
            bos=bos,
            eos=eos,
            flatten=flatten,
            skip_empty=skip_empty,
            root=root,
            tokenizer=tokenizer,
            **kwargs)


@register(segment=['train', 'val', 'test'])
class WikiText103(_WikiText):
    """WikiText-103 word-level dataset for language modeling, from Salesforce research.

    WikiText103 is implemented as CorpusDataset with the default flatten=True.

    From
    https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset

    License: Creative Commons Attribution-ShareAlike

    Parameters
    ----------
    segment : {'train', 'val', 'test'}, default 'train'
        Dataset segment.
    flatten : bool, default True
        Whether to return all samples as flattened tokens. If True, each sample is a token.
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

    def __init__(self,
                 segment='train',
                 flatten=True,
                 skip_empty=True,
                 tokenizer=lambda s: s.split(),
                 bos=None,
                 eos=C.EOS_TOKEN,
                 root=os.path.join(_get_home_dir(), 'datasets',
                                   'wikitext-103'),
                 **kwargs):
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
        super(WikiText103, self).__init__(
            'wikitext-103',
            segment=segment,
            bos=bos,
            eos=eos,
            flatten=flatten,
            skip_empty=skip_empty,
            root=root,
            tokenizer=tokenizer,
            **kwargs)


@register(segment=['train', 'val', 'test'])
class WikiText2Raw(_WikiText):
    """WikiText-2 character-level dataset for language modeling

    WikiText2Raw is implemented as CorpusDataset with the default flatten=True.

    From Salesforce research:
    https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset

    License: Creative Commons Attribution-ShareAlike

    Parameters
    ----------
    segment : {'train', 'val', 'test'}, default 'train'
        Dataset segment.
    flatten : bool, default True
        Whether to return all samples as flattened tokens. If True, each sample is a token.
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

    def __init__(self,
                 segment='train',
                 flatten=True,
                 skip_empty=True,
                 bos=None,
                 eos=None,
                 tokenizer=lambda s: s.encode('utf-8'),
                 root=os.path.join(_get_home_dir(), 'datasets', 'wikitext-2'),
                 **kwargs):
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

        super(WikiText2Raw, self).__init__(
            'wikitext-2',
            segment=segment,
            bos=bos,
            eos=eos,
            flatten=flatten,
            skip_empty=skip_empty,
            root=root,
            tokenizer=tokenizer,
            **kwargs)


@register(segment=['train', 'val', 'test'])
class WikiText103Raw(_WikiText):
    """WikiText-103 character-level dataset for language modeling

    WikiText103Raw is implemented as CorpusDataset with the default flatten=True.

    From Salesforce research:
    https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset

    License: Creative Commons Attribution-ShareAlike

    Parameters
    ----------
    segment : {'train', 'val', 'test'}, default 'train'
        Dataset segment.
    flatten : bool, default True
        Whether to return all samples as flattened tokens. If True, each sample is a token.
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

    def __init__(self,
                 segment='train',
                 flatten=True,
                 skip_empty=True,
                 tokenizer=lambda s: s.encode('utf-8'),
                 bos=None,
                 eos=None,
                 root=os.path.join(_get_home_dir(), 'datasets',
                                   'wikitext-103'),
                 **kwargs):
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
        super(WikiText103Raw, self).__init__(
            'wikitext-103',
            segment=segment,
            bos=bos,
            eos=eos,
            flatten=flatten,
            skip_empty=skip_empty,
            root=root,
            tokenizer=tokenizer,
            **kwargs)
