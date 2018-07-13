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
import mxnet as mx
import pytest

import gluonnlp as nlp


@pytest.mark.parametrize('prefetch', [0, 2])
@pytest.mark.parametrize('use_multiprocessing', [False, True])
def test_streaming_data_loader(prefetch, use_multiprocessing, batch_size=3):
    if use_multiprocessing and not prefetch:
        return

    stream = (mx.nd.zeros(10) for _ in range(10))
    batches = nlp.data.StreamDataLoader(
        stream, batch_size=batch_size, last_batch='discard', prefetch=prefetch,
        use_multiprocessing=use_multiprocessing)
    batches = list(batches)
    assert len(batches) == 3
    assert all(len(b) == batch_size for b in batches)

    stream = (mx.nd.zeros(10) for _ in range(10))
    batches = nlp.data.StreamDataLoader(
        stream, batch_size=3, last_batch='keep', prefetch=prefetch,
        use_multiprocessing=use_multiprocessing)
    batches = list(batches)
    assert len(batches) == 4
    assert all(len(b) == batch_size for b in batches[:-1])
    assert len(batches[-1]) == 10 % batch_size
