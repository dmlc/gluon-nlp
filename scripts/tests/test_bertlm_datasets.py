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

from __future__ import print_function

import datetime
import os
import io
import random

from flaky import flaky
import pytest

from ..language_model.transformer_lm_data import WikiText2WordPiece, WikiText103WordPiece


###############################################################################
# Language model
###############################################################################
@pytest.mark.remote_required
def test_wikitext2wp():
    train_dataset = WikiText2WordPiece(
        root=os.path.join('tests', 'data', 'wikitext2wp'), segment='train')
    test_dataset = WikiText2WordPiece(
        root=os.path.join('tests', 'data', 'wikitext2wp'), segment='test')
    val_dataset = WikiText2WordPiece(
        root=os.path.join('tests', 'data', 'wikitext2wp'), segment='val')

    assert len(train_dataset) == 2502795, len(train_dataset)
    assert len(test_dataset) == 312381, len(test_dataset)
    assert len(val_dataset) == 271452, len(val_dataset)


@pytest.mark.remote_required
def test_wikitext103wp():
    train_dataset = WikiText103WordPiece(
        root=os.path.join('tests', 'data', 'wikitext103wp'), segment='train')
    test_dataset = WikiText103WordPiece(
        root=os.path.join('tests', 'data', 'wikitext103wp'), segment='test')
    val_dataset = WikiText103WordPiece(
        root=os.path.join('tests', 'data', 'wikitext103wp'), segment='val')

    assert len(train_dataset) == 119748814, len(train_dataset)
    assert len(test_dataset) == 288026, len(test_dataset)
    assert len(val_dataset) == 251764, len(val_dataset)
