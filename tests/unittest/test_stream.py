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
import os
import pytest
import gluonnlp as nlp
import numpy as np
import mxnet as mx

@pytest.mark.serial
@pytest.mark.remote_required
def test_corpus_stream(
        stream_identity_wrappers,
        wikitext2_simpledatasetstream_skipempty_flatten_and_counter):
    stream, _ = wikitext2_simpledatasetstream_skipempty_flatten_and_counter

    stream = stream_identity_wrappers(stream)
    counter = nlp.data.Counter(itertools.chain.from_iterable(stream))
    assert len(counter) == 33278, len(counter)
    # examine aggregated vocab
    vocab = nlp.vocab.Vocab(counter, bos_token=None, padding_token=None)
    assert len(vocab) == 33278, len(vocab)
    # examine aggregated stats
    assert sum(counter.values()) == 2075677 + 216347 + 244102
    counter = nlp.data.Counter(itertools.chain.from_iterable(stream))
    assert len(counter) == 33278, len(counter)


@pytest.mark.serial
@pytest.mark.remote_required
def test_lazy_stream(stream_identity_wrappers):
    EOS = nlp._constants.EOS_TOKEN
    path = os.path.join('tests', 'data', 'wikitext-2')
    token_path = os.path.join('tests', 'data', 'wikitext-2/*test*.tokens')
    corpus = nlp.data.WikiText2(segment='test', root=path)

    # We don't use wikitext2_simpledatasetstream_skipempty_flatten_and_counter
    # here as there is no need to work on more than the 'test' segment. The
    # fixture goes over all segments.
    stream = nlp.data.SimpleDatasetStream(
        nlp.data.CorpusDataset,
        token_path,
        flatten=True,
        skip_empty=True,
        eos=EOS)
    wrapped_stream = stream_identity_wrappers(stream)
    transformed_stream = wrapped_stream.transform(lambda d: [s.lower() for s in d])

    wrapped_stream_iter = iter(wrapped_stream)
    transformed_stream_iter = iter(transformed_stream)
    for dataset in stream:
        prefetched_dataset = next(wrapped_stream_iter)
        transformed_dataset = next(transformed_stream_iter)
        assert all([
            w1.lower() == w2.lower() == w3 == w4.lower() for w1, w2, w3, w4 in
            zip(dataset, prefetched_dataset, transformed_dataset, corpus)
        ])


@pytest.mark.serial
def test_h5py_stream():
    import h5py
    file_path = os.path.join('tests', 'data', 'test_h5py_stream.hdf5')
    with h5py.File(file_path, 'w') as f:
        data = f.create_dataset("g1/g2/g3/mydataset.data", (5,3), dtype='i')
        data[0] = a = [1,2,3]
        data[1] = b = [4,5,6]

        label = f.create_dataset("g1/g2/g3/mydataset.label", (5,3), dtype='i')
        label[0] = 0
        label[1] = 1

        var_int = h5py.special_dtype(vlen=np.dtype('int32'))
        var_len_data = f.create_dataset("g1/g2/g3/mydataset.varlen", (10,), dtype=var_int)
        var_len_data[0] = c = [7,8,9]
        var_len_data[1] = d = [10,11]

    stream = nlp.data.H5PyDatasetStream(file_path)
    datasets = [data for data in stream]
    assert len(datasets) == 3
    
    stream = nlp.data.H5PyDatasetStream(file_path, select='.*\.data')
    datasets = [data for data in stream]
    assert len(datasets) == 1
    dataset = datasets[0]
    assert np.all(dataset[0] == a)
    assert np.all(dataset[1] == b)
    
    stream = nlp.data.H5PyDatasetStream(file_path, select='.*\.varlen')
    datasets = [data for data in stream]
    assert len(datasets) == 1
    dataset = datasets[0]
    assert np.all(dataset[0] == c)
    assert np.all(dataset[1] == d)

@pytest.mark.serial
def test_array_stream():
    import h5py
    file_path = os.path.join('tests', 'data', 'test_array_stream.hdf5')
    with h5py.File(file_path, 'w') as f:
        data1 = f.create_dataset("g1/g2/g3/1.data", (5,3), dtype='i')
        data2 = f.create_dataset("g1/g2/g3/2.data", (5,3), dtype='i')
        label1 = f.create_dataset("g1/g2/g3/1.label", (5,3), dtype='i')
        label2 = f.create_dataset("g1/g2/g3/2.label", (5,3), dtype='i')
        invalid = f.create_dataset("g1/g2/g3/1.invalid", (5,3), dtype='i')

    data_stream = nlp.data.H5PyDatasetStream(file_path, select='.*\.data')
    label_stream = nlp.data.H5PyDatasetStream(file_path, select='.*\.label')
    array_stream = nlp.data.ArrayDatasetStream(data_stream, label_stream)
    for arr in array_stream:
        assert isinstance(arr, mx.gluon.data.ArrayDataset)

    invalid_stream = nlp.data.H5PyDatasetStream(file_path, select='.*\.invalid')
    array_stream2 = nlp.data.ArrayDatasetStream(data_stream, label_stream, invalid_stream)
    try:
        for arr in array_stream2:
            assert isinstance(arr, mx.gluon.data.ArrayDataset)
        raise AssertionError('No exception caught')
    except ValueError:
        pass

@pytest.mark.parametrize('num_prefetch', [0, 1, 10])
@pytest.mark.parametrize('worker_type', ['thread', 'process'])
@pytest.mark.serial
@pytest.mark.remote_required
def test_prefetch_stream(num_prefetch, worker_type):
    EOS = nlp._constants.EOS_TOKEN
    path = os.path.join('tests', 'data', 'wikitext-2')
    token_path = os.path.join('tests', 'data', 'wikitext-2/*test*.tokens')
    test = nlp.data.WikiText2(segment='test', root=path)
    corpus = nlp.data.SimpleDatasetStream(
        nlp.data.CorpusDataset, token_path, flatten=True, skip_empty=True)
    if num_prefetch < 1:
        with pytest.raises(ValueError):
            prefetch_corpus = nlp.data.PrefetchingStream(
                corpus, num_prefetch=num_prefetch, worker_type=worker_type)
    else:
        prefetch_corpus = nlp.data.PrefetchingStream(
            corpus, num_prefetch=num_prefetch, worker_type=worker_type)
        prefetch_corpus_iter = iter(prefetch_corpus)
        for x in corpus:
            y = next(prefetch_corpus_iter)
            assert all([sx == sy for sx, sy in zip(x, y)])


class EagerIterWorksException(Exception):
    pass


@pytest.mark.parametrize('transform', [True, False])
def test_eager_iter_lazytransform(transform, stream_identity_wrappers):
    """Test that calling iter(stream.transform(fn)) eagerly calls iter(stream).

    If this test fails, PrefetchingStream(stream.transform(fn)) will not do any
    prefetching until next(iter(stream.transform(fn))) is called.

    """

    class ExceptionStream(nlp.data.DataStream):
        def __iter__(self):
            raise EagerIterWorksException

    stream = stream_identity_wrappers(ExceptionStream())
    if transform:
        stream = stream.transform(lambda x: x)

    with pytest.raises(EagerIterWorksException):
        iter(stream)

@pytest.mark.serial
@pytest.mark.remote_required
def test_dataset_stream_sampler():
    path = os.path.join('tests', 'data', 'wikitext-2')
    token_path = os.path.join('tests', 'data', 'wikitext-2/*.tokens')
    test = nlp.data.WikiText2(segment='test', root=path)
    num_parts = 2
    lengths = []
    for part_idx in range(num_parts):
        sampler = nlp.data.SplitSampler(3, num_parts, part_idx)
        corpus = nlp.data.SimpleDatasetStream(
            nlp.data.CorpusDataset, token_path, sampler)
        for c in corpus:
            assert len(c) not in lengths
            lengths.append(len(c))
    assert len(lengths) == 3
