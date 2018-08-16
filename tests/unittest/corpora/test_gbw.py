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
import json
import os

import gluonnlp as nlp
import mxnet as mx


def test_gbw():
    batch_size = 80
    seq_len = 35

    stream = nlp.data.GBWStream(segment='test')
    freq = nlp.data.utils.Counter(
        itertools.chain.from_iterable(itertools.chain.from_iterable(stream)))
    assert len(freq) == 21545
    assert sum(c for c in freq.values()) == 159658
    assert freq['English'] == 14
