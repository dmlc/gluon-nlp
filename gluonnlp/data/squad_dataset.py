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
"""SQuAD dataset."""

__all__ = ['SQuAD']

import json
import os
import shutil
import zipfile

from mxnet.gluon.data import ArrayDataset
from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url
from .registry import register


@register(segment=['train', 'test'])
class SQuAD(ArrayDataset):
    """Stanford Question Answering Dataset (SQuAD) - reading comprehension dataset.

    From
    https://rajpurkar.github.io/SQuAD-explorer/

    License: CreativeCommons BY-SA 4.0

    Parameters
    ----------
    segment : str, default 'train'
        Dataset segment. Options are 'train' and 'dev'.
    root : str, default '~/.mxnet/datasets/squad'
        Path to temp folder for storing data.
    """
    def __init__(self, segment='train', root=os.path.join('~', '.mxnet', 'datasets', 'squad')):
        self._data_file = {'train': ('train-v1.1.zip', 'train-v1.1.json',
                                     '052a75bf8fdb3e843b8649971658eae8133f9b0e'),
                           'dev': ('dev-v1.1.zip', 'dev-v1.1.json',
                                   'e31ad736582b72a8eabd5c0b0a38cb779ed913d7')}
        root = os.path.expanduser(root)

        if not os.path.isdir(root):
            os.makedirs(root)

        self._root = root
        self._segment = segment
        self._get_data()

        super(SQuAD, self).__init__(self._read_data())

    def _get_data(self):
        """Load data from the file. Does nothing if data was loaded before
        """
        data_archive_name, _, data_hash = self._data_file[self._segment]
        path = os.path.join(self._root, data_archive_name)

        if not os.path.exists(path) or not check_sha1(path, data_hash):
            file_path = download(_get_repo_file_url('gluon/dataset/squad', data_archive_name),
                                 path=self._root, sha1_hash=data_hash)

            with zipfile.ZipFile(file_path, 'r') as zf:
                for member in zf.namelist():
                    filename = os.path.basename(member)

                    if filename:
                        dest = os.path.join(self._root, filename)

                        with zf.open(member) as source, open(dest, 'wb') as target:
                            shutil.copyfileobj(source, target)

    def _read_data(self):
        """Read data.json from disk and flats it to the following format:
        Entry = (question_id, question, context, list_of_answers).
        Question id and list_of_answers also substituted with indices, so it could be later
        converted into nd.array

        Returns
        -------
        List[Tuple]
            Flatten list of questions
        """
        _, data_file_name, _ = self._data_file[self._segment]

        with open(os.path.join(self._root, data_file_name)) as f:
            samples = json.load(f)

        return SQuAD._get_records(samples)

    @staticmethod
    def _get_records(json_dict):
        """Provides a list of tuples with records where answers are flatten"""
        records = []

        record_index = 0

        for title in json_dict['data']:
            for paragraph in title['paragraphs']:
                for qas in paragraph['qas']:

                    record = (
                        qas['id'], qas['question'], paragraph['context'], SQuAD._get_answers(qas)
                    )

                    record_index += 1
                    records.append(record)

        return records

    @staticmethod
    def _get_answers(qas_dict):
        answers = []

        for answer in qas_dict['answers']:
            answers.append((answer['answer_start'], answer['text']))

        return answers
