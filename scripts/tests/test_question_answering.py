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
import pytest

from mxnet.gluon.data import DataLoader

from ..question_answering.data_pipeline import SQuADDataLoaderTransformer


@pytest.mark.remote_required
@pytest.mark.serial
def test_data_loader_able_to_read_spacy(squad_dev_and_vocab_spacy_provider):
    _, _, train_dataset, dev_dataset, word_vocab, char_vocab = squad_dev_and_vocab_spacy_provider
    dataloader = DataLoader(train_dataset.transform(SQuADDataLoaderTransformer()), batch_size=1)

    assert word_vocab is not None
    assert char_vocab is not None

    for record_index, context, query, context_char, query_char, begin, end in dataloader:
        assert record_index is not None
        assert context is not None
        assert query is not None
        assert context_char is not None
        assert query_char is not None
        assert begin is not None
        assert end is not None
        break


def test_data_loader_able_to_read_nltk(squad_dev_and_vocab_nltk_provider):
    _, _, train_dataset, dev_dataset, word_vocab, char_vocab = squad_dev_and_vocab_nltk_provider
    dataloader = DataLoader(train_dataset.transform(SQuADDataLoaderTransformer()), batch_size=1)

    assert word_vocab is not None
    assert char_vocab is not None

    for record_index, context, query, context_char, query_char, begin, end in dataloader:
        assert record_index is not None
        assert context is not None
        assert query is not None
        assert context_char is not None
        assert query_char is not None
        assert begin is not None
        assert end is not None
        break
