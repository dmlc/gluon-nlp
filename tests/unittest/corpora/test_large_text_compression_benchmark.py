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

import itertools
import pytest

import gluonnlp as nlp


@pytest.mark.remote_required
def test_text8():
    data = nlp.data.Text8()
    freq = nlp.data.utils.Counter(itertools.chain.from_iterable(data))
    assert len(freq) == 253854
    assert sum(c for c in freq.values()) == 17005207
    assert freq['english'] == 11868


@pytest.mark.remote_required
def test_fil9():
    data = nlp.data.Fil9()
    freq = nlp.data.utils.Counter(itertools.chain.from_iterable(data))
    assert len(freq) == 833184
    assert sum(c for c in freq.values()) == 124301826
    assert freq['english'] == 56767


@pytest.mark.remote_required
@pytest.mark.parametrize('segment', ['test', 'train', 'val', 'testraw', 'trainraw', 'valraw'])
def test_enwik8(segment):
    _ = nlp.data.Enwik8(segment=segment)
