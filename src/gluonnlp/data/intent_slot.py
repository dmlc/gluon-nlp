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
"""Datasets for intent classification and slot labeling."""

import io
import os
import zipfile
import numpy as np
from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url
from mxnet.gluon.data import SimpleDataset

from .dataset import TSVDataset
from .registry import register
from .utils import Splitter
from ..base import get_home_dir
from ..vocab import Vocab

__all__ = ['ATISDataset', 'SNIPSDataset']


class _BaseICSLDataset(SimpleDataset):
    """Base Class of Datasets for Joint Intent Classification and Slot Labeling.

    Parameters
    ----------
    segment : str
        Dataset segment.
    root : str
        Path to temp folder for storing data.
    """
    def __init__(self, segment, root):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._segment = segment
        self._root = root
        self._intent_vocab = None
        self._slot_vocab = None
        self._get_data()
        super(_BaseICSLDataset, self).__init__(self._read_data(segment))

    @property
    def _download_info(self):
        """Download file information.

        Returns
        -------
        filename_format : str
            The filename format with slot for short hash.
        sha1_hash : str
            Expected sha1 hash of the file content.
        """
        raise NotImplementedError

    @property
    def intent_vocab(self):
        if self._intent_vocab is None:
            with open(os.path.join(self._root, 'intent_vocab.json'), 'r') as f:
                self._intent_vocab = Vocab.from_json(f.read())
        return self._intent_vocab

    @property
    def slot_vocab(self):
        if self._slot_vocab is None:
            with open(os.path.join(self._root, 'slot_vocab.json'), 'r') as f:
                self._slot_vocab = Vocab.from_json(f.read())
        return self._slot_vocab

    def _get_data(self):
        filename_format, sha1_hash = self._download_info
        filename = filename_format.format(sha1_hash[:8])
        data_filename = os.path.join(self._root, filename)
        url = _get_repo_file_url('gluon/dataset', filename)
        if not os.path.exists(data_filename) or not check_sha1(data_filename, sha1_hash):
            download(url, path=data_filename, sha1_hash=sha1_hash)
            with zipfile.ZipFile(data_filename, 'r') as zf:
                zf.extractall(self._root)

    def _read_data(self, segment):
        sentences = TSVDataset(os.path.join(self._root, '{}_sentence.txt'.format(segment)),
                               field_separator=Splitter(' '))
        tags = TSVDataset(os.path.join(self._root, '{}_tags.txt'.format(segment)),
                          field_separator=Splitter(' '))
        with io.open(os.path.join(self._root, '{}_intent.txt'.format(segment)), 'r',
                     encoding='utf-8') as f:
            intents = []
            for line in f:
                line = line.strip()
                intents.append(np.array([self.intent_vocab[ele] for ele in line.split(';')],
                                        dtype=np.int32))
        return list(zip(sentences, tags, intents))


@register(segment=['train', 'dev', 'test'])
class ATISDataset(_BaseICSLDataset):
    """Airline Travel Information System dataset from MS CNTK.

    From
    https://github.com/Microsoft/CNTK/tree/master/Examples/LanguageUnderstanding/ATIS/Data

    License: Unspecified

    Each sample has three fields: tokens, slot labels, intent label.

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/atis'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

    Examples
    --------
    >>> atis = gluonnlp.data.ATISDataset(root='./datasets/atis')
    -etc-
    >>> len(atis)
    4478
    >>> len(atis[0])
    3
    >>> len(atis[0][0])
    10
    >>> atis[0][0]
    ['i', 'want', 'to', 'fly', 'from', 'baltimore', 'to', 'dallas', 'round', 'trip']
    >>> len(atis[0][1])
    10
    >>> atis[0][1][:8]
    ['O', 'O', 'O', 'O', 'O', 'B-fromloc.city_name', 'O', 'B-toloc.city_name']
    >>> atis[0][2]
    array([10], dtype=int32)
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'atis')):
        super(ATISDataset, self).__init__(segment, root)

    @property
    def _download_info(self):
        return 'atis-{}.zip', 'fb75a1b595566d5c5ec06ee6f2296d6629b8c225'


@register(segment=['train', 'dev', 'test'])
class SNIPSDataset(_BaseICSLDataset):
    """Snips Natural Language Understanding Benchmark dataset.

    Coucke et al. (2018). Snips Voice Platform: an embedded Spoken Language Understanding system
    for private-by-design voice interfaces. https://arxiv.org/abs/1805.10190

    From
    https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines

    License: Unspecified

    Each sample has three fields: tokens, slot labels, intent label.

    Parameters
    ----------
    segment : {'train', 'dev', 'test'}, default 'train'
        Dataset segment.
    root : str, default '$MXNET_HOME/datasets/snips'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

    Examples
    --------
    >>> snips = gluonnlp.data.SNIPSDataset(root='./datasets/snips')
    -etc-
    >>> len(snips)
    13084
    >>> len(snips[0])
    3
    >>> len(snips[1][0])
    8
    >>> snips[1][0]
    ['put', 'United', 'Abominations', 'onto', 'my', 'rare', 'groove', 'playlist']
    >>> len(snips[1][1])
    8
    >>> snips[1][1][:5]
    ['O', 'B-entity_name', 'I-entity_name', 'O', 'B-playlist_owner']
    >>> snips[1][2]
    array([0], dtype=int32)
    """
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'snips')):
        super(SNIPSDataset, self).__init__(segment, root)

    @property
    def _download_info(self):
        return 'snips-{}.zip', 'f22420cc0f2a26078337dc375606be46a4cc8c51'
