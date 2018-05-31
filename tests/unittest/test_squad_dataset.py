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
from gluonnlp.data.squad_dataset import SQuAD


def test_load_dev_squad():
    dataset = SQuAD(segment='dev', root=os.path.join('tests', 'data', 'squad'))

    # number of records in dataset is equal to number of different questions
    assert len(dataset) == 10570

    # Each record is a tuple of 4 elements: question Id, question, context, list of answers
    for record in dataset:
        assert len(record) == 4
