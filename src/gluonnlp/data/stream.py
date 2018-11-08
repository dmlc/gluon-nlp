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

# pylint: disable=undefined-all-variable
"""NLP Toolkit Data Stream API. It allows easy and customizable streaming of
corpora and dataset files. Files can be streamed into formats that are
ready for training and evaluation."""
__all__ = [
    'DataStream', 'SimpleDataStream', 'DatasetStream', 'SimpleDatasetStream',
    'PrefetchingStream'
]

import glob
import multiprocessing
import os
import random
import threading

import numpy as np

import mxnet as mx
from mxnet.gluon.data import RandomSampler, SequentialSampler

try:
    import Queue as queue
except ImportError:
    import queue


class DataStream(object):
    """Abstract Data Stream Interface.

    DataStreams are useful to avoid loading big datasets to memory. A
    DataStream is a iterable object (it implements the __iter__ function).
    Whenever an iteration over the DataStream is requested (e.g. in a for loop
    or by calling iter(datastream)), a new iterator over all samples in the
    DataStream is returned. DataStreams can be lazily transformed by calling
    `transform()` which returns a DataStream over the transformed samples.

    """

    def __iter__(self):
        """Return an iterator over all elements of the DataStream.

        This method returns a new iterator object that can iterate over
        all the objects in the DataStream.

        Returns
        -------
        iterator
            An object implementing the Python *iterator protocol*.

        """
        raise NotImplementedError

    def transform(self, fn):
        """Transform a DataStream lazily.

        Returns
        -------
        DataStream
            The data stream that lazily transforms the data while streaming.
        """

        return _LazyTransformDataStream(self, fn)


class SimpleDataStream(DataStream):
    """SimpleDataStream wraps iterables to expose the DataStream API.

    Unlike the iterable itself, the SimpleDataStream exposes the DataStream API
    and allows lazy transformation of the iterable.

    """
    def __init__(self, iterable):
        self._stream = iterable

    def __iter__(self):
        return iter(self._stream)


class _LazyTransformDataStream(DataStream):
    """Data stream that lazily transforms the data."""
    def __init__(self, stream, fn):
        self._stream = stream
        self._fn = fn

    def __iter__(self):
        stream_iter = iter(self._stream)
        try:
            item = next(stream_iter)
        except StopIteration:
            return
        istuple = isinstance(item, tuple)
        if istuple:
            yield self._fn(*item)
            for item in stream_iter:
                yield self._fn(*item)
        else:
            yield self._fn(item)
            for item in stream_iter:
                yield self._fn(item)


class DatasetStream(DataStream):
    """Abstract Dataset Stream Interface.

    A DatasetStream is a DataStream where each sample is a
    `mxnet.gluon.data.Dataset`. An iteration over a DatasetStream iterates over
    `mxnet.gluon.data.Dataset` objects, representing a chunk or shards of some
    large datasets.

    Iterating over sizeable chunks of a dataset can be helpful to speed up
    preprocessing as the overhead of preprocessing each sample individually is
    reduced (this is similar to the idea of using batches for training a
    model).

    """

    def __iter__(self):
        raise NotImplementedError


class SimpleDatasetStream(DatasetStream):
    """A simple stream of Datasets.

    The SimpleDatasetStream is created from multiple files based on provided
    `file_pattern`. One file is read at a time and a corresponding Dataset is
    returned. The Dataset is created based on the file and the kwargs passed to
    SimpleDatasetStream.

    Parameters
    ----------
    dataset : class
        The class for which to create an object for every file. kwargs are
        passed to this class.
    file_pattern: str
        Path to the input text files.
    file_sampler : str, {'sequential', 'random'}, defaults to 'random'
        The sampler used to sample a file.

        - 'sequential': SequentialSampler
        - 'random': RandomSampler
    kwargs
        All other keyword arguments are passed to the dataset constructor.
    """
    def __init__(self, dataset, file_pattern, file_sampler='random', **kwargs):
        if not isinstance(file_pattern, str):
            raise TypeError('file_pattern must be str, but got %s'%type(file_pattern))
        self._dataset = dataset
        self._file_pattern = os.path.expanduser(file_pattern)
        self._file_sampler = file_sampler
        self._kwargs = kwargs

    def _get_sampler(self, sampler):
        assert isinstance(sampler, str), 'Expected sampler to be a str, but got %s'%type(sampler)
        if sampler == 'random':
            return RandomSampler
        if sampler == 'sequential':
            return SequentialSampler
        raise ValueError('sampler must be either "random" or "sequential", but got %s'%(sampler))

    def __iter__(self):
        file_sampler = self._get_sampler(self._file_sampler)
        # generate file samples
        files = sorted(glob.glob(self._file_pattern))
        if len(files) == 0:
            raise ValueError('Cannot find any file with path "%s"'%self._file_pattern)
        for file_idx in iter(file_sampler(len(files))):
            filename = files[file_idx]
            yield self._dataset(filename, **self._kwargs)


class _Prefetcher(object):
    """Internal shared prefetcher logic."""
    data_queue = None
    control_queue = None

    def __init__(self, stream, num_prefetch, seed, np_seed, mx_seed):
        super(_Prefetcher, self).__init__()
        self.stream = stream
        assert num_prefetch > 0, 'Unbounded Prefetcher is unsupported.'
        self.num_prefetch = num_prefetch
        self.seed = seed
        self.np_seed = np_seed
        self.mx_seed = mx_seed

    def run(self):
        """Method representing the process’s activity."""
        random.seed(self.seed)
        np.random.seed(self.np_seed)
        mx.random.seed(self.mx_seed)

        stream_iter = iter(self.stream)
        while True:
            try:  # Check control queue
                c = self.control_queue.get(False)
                if c is None:
                    break
            except queue.Empty:
                pass

            try:
                data = next(stream_iter)
                self.data_queue.put(data)
            except StopIteration:
                self.data_queue.put(None)

    def __next__(self):
        next_item = self.data_queue.get()
        if next_item is None:
            self.control_queue.put(None)
            raise StopIteration
        return next_item

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self


class _ProcessPrefetcher(_Prefetcher, multiprocessing.Process):
    """Internal multi-processing prefetcher."""

    def __init__(self, *args, **kwargs):
        super(_ProcessPrefetcher, self).__init__(*args, **kwargs)
        self.data_queue = multiprocessing.Queue(self.num_prefetch)
        self.control_queue = multiprocessing.Queue()
        self.daemon = True
        self.start()


class _ThreadPrefetcher(_Prefetcher, threading.Thread):
    """Internal threaded prefetcher."""

    def __init__(self, *args, **kwargs):
        super(_ThreadPrefetcher, self).__init__(*args, **kwargs)
        self.data_queue = queue.Queue(self.num_prefetch)
        self.control_queue = queue.Queue()
        self.daemon = True
        self.start()


class PrefetchingStream(DataStream):
    """Prefetch a DataStream in a separate Thread or Process.

    This iterator will create another thread or process to perform
    ``iter_next`` and then store the data in memory. It potentially accelerates
    the data read, at the cost of more memory usage.

    The python, numpy and mxnet random states in the launched Thread or Process
    will be initialized randomly based on the next 32 bit integer in the
    python, numpy and mxnet random generator of the caller respectively
    (random.getrandbits(32), numpy.random.randint(0, 2**32),
    int(mx.nd.random.uniform(0, 2**32).asscalar())).

    Parameters
    ----------
    stream : DataStream
        Source stream.
    num_prefetch : int, default 1
        Number of elements to prefetch from the stream. Must be greater 0.
    worker_type : 'thread' or 'process', default 'thread'
        Use a separate Python Thread or Process to prefetch.

    """

    def __init__(self, stream, num_prefetch=1, worker_type='thread'):
        self._stream = stream
        self._num_prefetch = num_prefetch
        if num_prefetch < 1:
            raise ValueError('num_prefetch must be greater 0.')
        assert worker_type.lower() in ['thread', 'process']
        self._multiprocessing = worker_type.lower() == 'process'

    def __iter__(self):
        seed = random.getrandbits(32)
        np_seed = np.random.randint(0, 2**32)
        mx_seed = int(mx.nd.random.uniform(0, 2**32).asscalar())
        if self._multiprocessing:
            return _ProcessPrefetcher(self._stream, self._num_prefetch,
                                      seed=seed, np_seed=np_seed,
                                      mx_seed=mx_seed)
        else:
            return _ThreadPrefetcher(self._stream, self._num_prefetch,
                                     seed=seed, np_seed=np_seed,
                                     mx_seed=mx_seed)
