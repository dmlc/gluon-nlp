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
"""WikiText wordpiece tokenized corpora."""

__all__ = ['WikiText2WordPiece', 'WikiText103WordPiece']

import os

import mxnet as mx
from mxnet.gluon.data import SimpleDataset
from gluonnlp.data import register, corpora
from gluonnlp.data.utils import _get_home_dir
from gluonnlp import _constants as C


class TransformedCorpusBatchify(object):
    """Batchify the transformed dataset into N independent sequences, where N is the batch size.

    Parameters
    ----------
    batch_size : int
        The number of samples in each batch.
    """

    def __init__(self, batch_size):
        self._batch_size = batch_size

    def __call__(self, data):
        """Batchify a dataset.

        Parameters
        ----------
        data : mxnet.gluon.data.Dataset
            A flat dataset to be batchified.

        Returns
        -------
        mxnet.gluon.data.Dataset
            NDArray of shape (len(data) // N, N) where N is the batch_size
            wrapped by a mxnet.gluon.data.SimpleDataset. Excessive tokens that
            don't align along the batches are discarded.
        """
        batch_num = len(data) // self._batch_size
        return SimpleDataset(
            mx.nd.array(
                data[:batch_num * self._batch_size]).reshape(
                    self._batch_size, -1).T)


def int_transformed_whitespace_splitter(s):
    """Split a string at whitespace (space, tab, newline, return, formfeed).

    Parameters
    ----------
    s : str
        The string to be split

    Returns
    --------
    List[int]
        List of int. Obtained by calling s.split().
    """
    ids = [int(t) for t in s.split()]
    return ids


@register(segment=['train', 'val', 'test'])
class WikiText2WordPiece(corpora.wikitext._WikiText):

    """WikiText-2 wordpiece tokenized dataset for language modeling.

    WikiText2WordPiece is implemented as CorpusDataset with the default flatten=True.

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
        The token to add at the beginning of each sentence. If None, nothing is added.
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
                 tokenizer=int_transformed_whitespace_splitter,
                 bos=None,
                 eos=C.EOS_TOKEN,
                 root=os.path.join(_get_home_dir(), 'datasets', 'wikitext-2'),
                 **kwargs):
        self._archive_file = ('wikitext-2-wp.zip',
                              '1cd397cbc946c5f5e94c1a4b3ed051134b82bfa7')
        self._data_file = {
            'train': ('wiki.train.wp.tokens',
                      '3aa7b785ce78ac5f9506cfed0623d2c7c635aa64'),
            'val': ('wiki.valid.wp.tokens',
                    '281bdcda51c35d5952fee7c538b7f1502efcaad1'),
            'test': ('wiki.test.wp.tokens',
                     'a25072abd6a69d256a57f7af5e434f0d2d091dd5')
        }
        super(WikiText2WordPiece, self).__init__(
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
class WikiText103WordPiece(corpora.wikitext._WikiText):
    """WikiText-103 wordpiece tokenized dataset for language modeling.

    WikiText103WordPiece is implemented as CorpusDataset with the default flatten=True.

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
        The token to add at the beginning of each sentence. If None, nothing is added.
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
                 tokenizer=int_transformed_whitespace_splitter,
                 bos=None,
                 eos=C.EOS_TOKEN,
                 root=os.path.join(_get_home_dir(), 'datasets',
                                   'wikitext-103'),
                 **kwargs):
        self._archive_file = ('wikitext-103-wp.zip',
                              'c9e52bca45ca88e01b84e258251eec2be707186a')
        self._data_file = {
            'train': ('wiki.train.wp.tokens',
                      'a3def8aed2d6152209b73fbf21e6cbb9def23ab6'),
            'val': ('wiki.valid.wp.tokens',
                    '5b730ac5ef35a1c3e6490ae196a2bf597acbffda'),
            'test': ('wiki.test.wp.tokens',
                     'c250f55bc98ff7b30323c2b93da5f7aaa04fb4ef')
        }
        super(WikiText103WordPiece, self).__init__(
            'wikitext-103',
            segment=segment,
            bos=bos,
            eos=eos,
            flatten=flatten,
            skip_empty=skip_empty,
            root=root,
            tokenizer=tokenizer,
            **kwargs)
