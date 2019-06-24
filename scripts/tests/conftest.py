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
"""conftest.py contains configuration for pytest."""

import os

import pytest

from ..question_answering.data_pipeline import SQuADDataPipeline


###############################################################################
# Datasets
###############################################################################
@pytest.fixture(scope='session')
def squad_dev_and_vocab_spacy_provider():
    path = os.path.join('tests', 'data', 'squad')
    pipeline = SQuADDataPipeline(400, 50, 1000, 100, 30, 16, 'glove.840B.300d')
    return pipeline.get_processed_data(squad_data_root=path)


@pytest.fixture(scope='session')
def squad_dev_and_vocab_nltk_provider():
    path = os.path.join('tests', 'data', 'squad')
    pipeline = SQuADDataPipeline(400, 50, 1000, 100, 30, 16, 'glove.6B.100d')
    return pipeline.get_processed_data(use_spacy=False, squad_data_root=path)
