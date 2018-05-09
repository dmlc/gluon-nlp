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
import numpy as np

from gluonnlp import Vocab

__all__ = ['SQuAD', 'SQuADTransform']

import io
import json
import os

from mxnet.gluon.data import ArrayDataset
from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url
from .registry import register


@register(segment=['train', 'test'])
class SQuAD(ArrayDataset):
    """Stanford Question Answering Dataset (SQuAD) - reading comprehension dataset.

    From
    https://rajpurkar.github.io/SQuAD-explorer/

    Parameters
    ----------
    segment : str, default 'train'
        Dataset segment. Options are 'train' and 'dev'.
    root : str, default '~/.mxnet/datasets/squad'
        Path to temp folder for storing data.
    """
    def __init__(self, segment='train', root=os.path.join('~', '.mxnet', 'datasets', 'squad')):
        self._data_file = {'train': ('train-v1.1.json',
                                     '1faea1252438a64f9718412a55036b786cfcc636'),
                           'dev': ('dev-v1.1.json',
                                    'e1621aae0683b346ee9743bd5609266ba0cc34fc'),
                           'word_vocab': ('word_vocab.json',
                                   '13fd4dc500916612387c46fce46769638dccba5b'),
                           'char_vocab': ('char_vocab.json',
                                   'ca8fd555a45136e1643713041de53207c9aea2ad')}
        root = os.path.expanduser(root)

        if not os.path.isdir(root):
            os.makedirs(root)

        self._root = root
        self._segment = segment
        self._get_data()
        self._word_vocab = None
        self._char_vocab = None

        super(SQuAD, self).__init__(self._read_data())

    def _get_data(self):
        """Load data from the file. Does nothing if data was loaded before
        """
        data_file_name, data_hash = self._data_file[self._segment]
        path = os.path.join(self._root, data_file_name)

        if not os.path.exists(path) or not check_sha1(path, data_hash):
            download(_get_repo_file_url('gluon/dataset/squad', data_file_name),
                     path=self._root, sha1_hash=data_hash)

    def _read_data(self):
        """Read data.json from disk and flats it to the following format:
        Entry = (question_id, question, context, list_of_answers)

        Returns
        -------
        List[Tuple]
            Flatten list of questions
        """
        data_file_name, data_hash = self._data_file[self._segment]

        with open(os.path.join(self._root, data_file_name)) as f:
            samples = json.load(f)

        return _SQuADJsonParser().get_records(samples)

    @property
    def word_vocab(self):
        """Word-level vocabulary of the training dataset.

        Returns
        -------
        train_vocab : Vocab
            Word-level training set vocabulary.
        """
        if self._word_vocab is None:
            self._word_vocab = self._load_vocab('word_vocab')

        return self._word_vocab

    @property
    def char_vocab(self):
        """Char-level vocabulary of the ttraining dataset"""
        if self._char_vocab is None:
            self._char_vocab = self._load_vocab('char_vocab')

        return self._char_vocab

    def _load_vocab(self, vocab_key):
        data_file_name, data_hash = self._data_file[vocab_key]
        path = os.path.join(self._root, data_file_name)

        if not os.path.exists(path) or not check_sha1(path, data_hash):
            download(_get_repo_file_url('gluon/dataset/squad', data_file_name),
                     path=self._root, sha1_hash=data_hash)

        with io.open(path, 'r', encoding='utf-8') as in_file:
            vocab = Vocab.from_json(in_file.read())

        return vocab


class _SQuADJsonParser:
    def __init__(self):
        pass

    def get_records(self, json_dict):
        records = []

        for title in json_dict["data"]:
            for paragraph in title["paragraphs"]:
                for qas in paragraph["qas"]:
                    record = (
                        qas["id"], qas["question"], paragraph["context"], self._get_answers(qas)
                    )

                    records.append(record)

        return records

    def _get_answers(self, qas_dict):
        answers = []

        for answer in qas_dict["answers"]:
            answers.append((answer["answer_start"], answer["text"]))

        return answers


class SQuADTransform(object):
    def __init__(self, word_vocab, char_vocab, question_max_length, context_max_length):
        self._word_vocab = word_vocab
        self._char_vocab = char_vocab

        self._question_max_length = question_max_length
        self._context_max_length = context_max_length

    def __call__(self, q_id, question, context, answers):
        """
        Method converts text into numeric arrays based on Vocabulary.
        Answers are not processed, as they are not needed in input
        """
        question_words = self._word_vocab[question.split()[:self._question_max_length]]
        context_words = self._word_vocab[context.split()[:self._context_max_length]]

        question_chars = [self._char_vocab[list(iter(word))]
                          for word in question.split()[:self._question_max_length]]

        context_chars = [self._char_vocab[list(iter(word))]
                          for word in context.split()[:self._context_max_length]]

        question_words_npy = np.array(question_words, dtype=np.int32)
        question_chars_npy = np.array(question_chars, dtype=np.int32)

        context_words_npy = np.array(context_words, dtype=np.int32)
        context_chars_npy = np.array(context_chars, dtype=np.int32)

        return q_id, question_words_npy, question_chars_npy, \
               context_words_npy, context_chars_npy, answers
