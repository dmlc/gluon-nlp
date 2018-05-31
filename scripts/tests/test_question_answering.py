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
import os

from mxnet.gluon.data import DataLoader, SimpleDataset

from gluonnlp.data import SQuAD
from scripts.question_answering.data_processing import SQuADTransform, VocabProvider

question_max_length = 30
context_max_length = 256


def test_transform_to_nd_array():
    record_index = 0
    dataset = SQuAD(segment='dev', root=os.path.join('tests', 'data', 'squad'))
    vocab_provider = VocabProvider(dataset)
    transformer = SQuADTransform(vocab_provider, question_max_length, context_max_length)
    record = dataset[record_index]

    transformed_record = transformer(record_index, record[1], record[2])
    assert transformed_record is not None


def test_data_loader_able_to_read():
    record_index = 0
    dataset = SQuAD(segment='dev', root=os.path.join('tests', 'data', 'squad'))
    vocab_provider = VocabProvider(dataset)
    transformer = SQuADTransform(vocab_provider, question_max_length, context_max_length)
    record = dataset[record_index]

    transformed_dataset = SimpleDataset([transformer(record_index, record[1], record[2])])
    dataloader = DataLoader(transformed_dataset, batch_size=1)

    for data in dataloader:
        record_index, question_words, question_chars, context_words, context_chars = data

        assert record_index is not None
        assert question_words is not None
        assert question_chars is not None
        assert context_words is not None
        assert context_chars is not None


def test_load_vocabs():
    dataset = SQuAD(segment='dev', root=os.path.join('tests', 'data', 'squad'))
    vocab_provider = VocabProvider(dataset)

    assert vocab_provider.get_word_level_vocab() is not None
    assert vocab_provider.get_char_level_vocab() is not None
