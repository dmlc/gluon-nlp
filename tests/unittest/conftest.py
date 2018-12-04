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

import functools
import glob
import itertools
import os

import pytest

import gluonnlp as nlp


###############################################################################
# Datasets
###############################################################################
@pytest.fixture(scope="session")
def wikitext2_train_and_counter():
    path = os.path.join('tests', 'data', 'wikitext-2')
    data = nlp.data.WikiText2(segment='train', root=path)
    counter = nlp.data.utils.Counter(data)
    return data, counter


@pytest.fixture(scope="session")
def wikitext2_test_and_counter():
    path = os.path.join('tests', 'data', 'wikitext-2')
    data = nlp.data.WikiText2(segment='test', root=path)
    counter = nlp.data.utils.Counter(data)
    return data, counter


@pytest.fixture(scope="session")
def wikitext2_val_and_counter():
    path = os.path.join('tests', 'data', 'wikitext-2')
    data = nlp.data.WikiText2(segment='val', root=path)
    counter = nlp.data.utils.Counter(data)
    return data, counter


###############################################################################
# Stream
###############################################################################
@pytest.fixture(params=["prefetch_process", "prefetch_thread", "none"])
def stream_identity_wrappers(request):
    """DataStream wrappers that don't change the content of a Stream.

    All DataStreams included in Gluon-NLP should support being wrapped by one
    of the wrappers returned by this test fixture. When writing a test to test
    some Stream, make sure to parameterize it by stream_identity_wrappers so
    that the stream is tested with all possible stream wrappers.

    """
    if request.param == "prefetch_process":
        return functools.partial(
            nlp.data.PrefetchingStream, worker_type='process')
    elif request.param == "prefetch_thread":
        return functools.partial(
            nlp.data.PrefetchingStream, worker_type='thread')
    elif request.param == "none":
        return lambda x: x
    else:
        raise RuntimeError


@pytest.fixture(scope="session")
def wikitext2_simpledatasetstream_skipempty_and_counter(
        wikitext2_train_and_counter, wikitext2_test_and_counter,
        wikitext2_val_and_counter):
    token_path = os.path.join('tests', 'data', 'wikitext-2/*.tokens')
    assert len(glob.glob(token_path)) == 3
    stream = nlp.data.SimpleDatasetStream(
        nlp.data.CorpusDataset,
        token_path,
        skip_empty=True,
        eos=nlp._constants.EOS_TOKEN)
    counter = nlp.data.Counter(
        itertools.chain.from_iterable(itertools.chain.from_iterable(stream)))
    return stream, counter


@pytest.fixture(scope="session")
def wikitext2_simpledatasetstream_skipempty_flatten_and_counter(
        wikitext2_train_and_counter, wikitext2_test_and_counter,
        wikitext2_val_and_counter):
    token_path = os.path.join('tests', 'data', 'wikitext-2/*.tokens')
    assert len(glob.glob(token_path)) == 3
    stream = nlp.data.SimpleDatasetStream(
        nlp.data.CorpusDataset,
        token_path,
        flatten=True,
        skip_empty=True,
        eos=nlp._constants.EOS_TOKEN)
    counter = nlp.data.Counter(
        itertools.chain.from_iterable(itertools.chain.from_iterable(stream)))
    return stream, counter
