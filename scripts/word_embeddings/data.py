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
"""Word embedding training datasets."""

__all__ = ['WikiDumpStream']

import io
import json
import os

from gluonnlp import Vocab
from gluonnlp.data import SimpleDatasetStream, CorpusDataset


class WikiDumpStream(SimpleDatasetStream):
    """Stream for preprocessed Wikipedia Dumps.

    Expects data in format
    - root/date/wiki.language/*.txt
    - root/date/wiki.language/vocab.json
    - root/date/wiki.language/counts.json

    Parameters
    ----------
    path : str
        Path to a folder storing the dataset and preprocessed vocabulary.
    skip_empty : bool, default True
        Whether to skip the empty samples produced from sample_splitters. If False, `bos` and `eos`
        will be added in empty samples.
    bos : str or None, default None
        The token to add at the begining of each sentence. If None, nothing is added.
    eos : str or None, default None
        The token to add at the end of each sentence. If None, nothing is added.

    Attributes
    ----------
    vocab : gluonnlp.Vocab
        Vocabulary object constructed from vocab.json.
    idx_to_counts : list[int]
        Mapping from vocabulary word indices to word counts.

    """

    def __init__(self, root, language, date, skip_empty=True, bos=None,
                 eos=None):
        self._root = root
        self._language = language
        self._date = date
        self._path = os.path.join(root, date, 'wiki.' + language)

        if not os.path.isdir(self._path):
            raise ValueError('{} is not valid. '
                             'Please make sure that the path exists and '
                             'contains the preprocessed files.'.format(
                                 self._path))

        self._file_pattern = os.path.join(self._path, '*.txt')
        super(WikiDumpStream, self).__init__(
            dataset=CorpusDataset,
            file_pattern=self._file_pattern,
            skip_empty=skip_empty,
            bos=bos,
            eos=eos)

    @property
    def vocab(self):
        path = os.path.join(self._path, 'vocab.json')
        with io.open(path, 'r', encoding='utf-8') as in_file:
            return Vocab.from_json(in_file.read())

    @property
    def idx_to_counts(self):
        path = os.path.join(self._path, 'counts.json')
        with io.open(path, 'r', encoding='utf-8') as in_file:
            return json.load(in_file)
