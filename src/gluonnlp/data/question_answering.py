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
from ..base import get_home_dir

@register(segment=['train', 'dev'])
class SQuAD(ArrayDataset):
    """Stanford Question Answering Dataset (SQuAD) - reading comprehension dataset.

    From
    https://rajpurkar.github.io/SQuAD-explorer/

    License: CreativeCommons BY-SA 4.0

    The original data format is json, which has multiple contexts (a context is a paragraph of text
    from which questions are drawn). For each context there are multiple questions, and for each of
    these questions there are multiple (usually 3) answers.

    This class loads the json and flattens it to a table view. Each record is a single question.
    Since there are more than one question per context in the original dataset, some records shares
    the same context. Number of records in the dataset is equal to number of questions in json file.

    The format of each record of the dataset is following:

    - record_index:  An index of the record, generated on the fly (0 ... to # of last question)
    - question_id:   Question Id. It is a string and taken from the original json file as-is
    - question:      Question text, taken from the original json file as-is
    - context:       Context text.  Will be the same for questions from the same context
    - answer_list:   All answers for this question. Stored as python list
    - start_indices: All answers' starting indices. Stored as python list.
      The position in this list is the same as the position of an answer in answer_list
    - is_impossible: The question is unanswerable, if version is '2.0'.
      In SQuAd2.0, there are some unanswerable questions.

    Parameters
    ----------
    segment : str, default 'train'
        Dataset segment. Options are 'train' and 'dev'.
    version : str, default '1.1'
        Dataset version. Options are '1.1' and '2.0'.
    root : str, default '~/.mxnet/datasets/squad'
        Path to temp folder for storing data.

    Examples
    --------
    >>> squad = gluonnlp.data.SQuAD('dev', '1.1', root='./datasets/squad')
    -etc-
    >>> len(squad)
    10570
    >>> len(squad[0])
    6
    >>> tuple(type(squad[0][i]) for i in range(6))
    (<class 'int'>, <class 'str'>, <class 'str'>, <class 'str'>, <class 'list'>, <class 'list'>)
    >>> squad[0][0]
    0
    >>> squad[0][1]
    '56be4db0acb8001400a502ec'
    >>> squad[0][2]
    'Which NFL team represented the AFC at Super Bowl 50?'
    >>> squad[0][3][:70]
    'Super Bowl 50 was an American football game to determine the champion '
    >>> squad[0][4]
    ['Denver Broncos', 'Denver Broncos', 'Denver Broncos']
    >>> squad[0][5]
    [177, 177, 177]
    >>> squad2 = gluonnlp.data.SQuAD('dev', '2.0', root='./datasets/squad')
    -etc-
    >>> len(squad2)
    11873
    >>> len(squad2[0])
    7
    >>> type(squad2[0][6])
    <class 'bool'>
    >>> squad2[0][6]
    False
    """

    def __init__(self, segment='train', version='1.1',
                 root=os.path.join(get_home_dir(), 'datasets', 'squad')):
        self._data_file = {'1.1': {'train': (('train-v1.1.zip',
                                              '052a75bf8fdb3e843b8649971658eae8133f9b0e'),
                                             ('train-v1.1.json',
                                              '1faea1252438a64f9718412a55036b786cfcc636')),
                                   'dev': (('dev-v1.1.zip',
                                            'e31ad736582b72a8eabd5c0b0a38cb779ed913d7'),
                                           ('dev-v1.1.json',
                                            'e1621aae0683b346ee9743bd5609266ba0cc34fc'))},
                           '2.0': {'train': (('train-v2.0.zip',
                                              'fe497797fc090ee61a046b74eadfee51320b54fb'),
                                             ('train-v2.0.json',
                                              'ceb2acdea93b9d82ab1829c7b1e03bee9e302c99')),
                                   'dev': (('dev-v2.0.zip',
                                            'de4dad80b3de9194484ca013e95a96a3e2d5603f'),
                                           ('dev-v2.0.json',
                                            '846082d15ed71cb5220645b9d473441e00070778'))}}

        root = os.path.expanduser(root)

        if not os.path.isdir(root):
            os.makedirs(root)

        self._root = root
        self._segment = segment
        self._version = version
        self._get_data()

        super(SQuAD, self).__init__(SQuAD._get_records(self._read_data()))

    def _get_data(self):
        """Load data from the file. Does nothing if data was loaded before.
        """
        (data_archive_name, archive_hash), (data_name, data_hash) \
            = self._data_file[self._version][self._segment]
        data_path = os.path.join(self._root, data_name)

        if not os.path.exists(data_path) or not check_sha1(data_path, data_hash):
            file_path = download(_get_repo_file_url('gluon/dataset/squad', data_archive_name),
                                 path=self._root, sha1_hash=archive_hash)

            with zipfile.ZipFile(file_path, 'r') as zf:
                for member in zf.namelist():
                    filename = os.path.basename(member)

                    if filename:
                        dest = os.path.join(self._root, filename)

                        with zf.open(member) as source, open(dest, 'wb') as target:
                            shutil.copyfileobj(source, target)

    def _read_data(self):
        """Read data.json from disk and flats it to the following format:
        Entry = (record_index, question_id, question, context, answer_list, answer_start_indices).
        Question id and list_of_answers also substituted with indices, so it could be later
        converted into nd.array

        Returns
        -------
        List[Tuple]
            Flatten list of questions
        """
        (_, _), (data_file_name, _) \
            = self._data_file[self._version][self._segment]

        with open(os.path.join(self._root, data_file_name)) as f:
            json_data = json.load(f)

        return json_data

    @staticmethod
    def _get_records(json_dict):
        """Provides a list of tuples with records where answers are flatten

        :param dict, json_dict: File content loaded into json dictionary

        Returns
        -------
        List[Tuple]
            Flatten list of records in format: record_index, question_id, question, context,
            answer, answer_start_index, is_impossible(if version is '2.0)
        """
        records = []

        record_index = 0

        for title in json_dict['data']:
            for paragraph in title['paragraphs']:
                for qas in paragraph['qas']:
                    answers = SQuAD._get_answers(qas)
                    is_impossible = qas.get('is_impossible', None)
                    if is_impossible is not None:
                        record = (
                            record_index, qas['id'], qas['question'],
                            paragraph['context'], answers[0], answers[1], is_impossible
                        )
                    else:
                        record = (
                            record_index, qas['id'], qas['question'],
                            paragraph['context'], answers[0], answers[1])

                    record_index += 1
                    records.append(record)

        return records

    @staticmethod
    def _get_answers(qas_dict):

        answer_list = []
        answer_start_list = []

        for answer in qas_dict['answers']:
            answer_list.append(answer['text'])
            answer_start_list.append(answer['answer_start'])

        return answer_list, answer_start_list
