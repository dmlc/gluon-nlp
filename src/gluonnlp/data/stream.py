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

from __future__ import print_function

import glob
import multiprocessing
import os
import random
import sys
import threading
import traceback
import re

import numpy as np

import mxnet as mx
from gluonnlp.base import _str_types
from mxnet.gluon.data import RandomSampler, SequentialSampler, Sampler, SimpleDataset, ArrayDataset

try:
    import Queue as queue
except ImportError:
    import queue

__all__ = [
    'DataStream', 'SimpleDataStream', 'DatasetStream', 'SimpleDatasetStream',
    'PrefetchingStream', 'H5PyDatasetStream', 'ArrayDatasetStream']


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

        # Yield must be hidden in closure so that __iter__ is called before
        # __next__ is called. This is important, as calling iter(self._stream)
        # may trigger multi-threaded or multi-processing prefetching of the
        # stream.
        def _closure():
            try:
                item = next(stream_iter)
            except StopIteration:
                return
            istuple = isinstance(item, tuple)
            if istuple:
                yield self._fn(*item)
                while True:
                    try:
                        yield self._fn(*next(stream_iter))
                    except StopIteration:
                        return
            else:
                yield self._fn(item)
                while True:
                    try:
                        yield self._fn(next(stream_iter))
                    except StopIteration:
                        return

        return _closure()


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

class ArrayDatasetStream(DatasetStream):
    """A dataset stream that combines multiple DatasetStream objects,
    and returns a stream of ArrayDataset.

    Parameters
    ----------
    *args : one or more DatasetStream objects
        The data arrays.
    """
    def __init__(self, *args):
        assert len(args) > 0, "Needs at least 1 arrays"
        self._data = list(args)

    def __iter__(self):
        streams = [iter(data) for data in self._data]
        num_active = len(streams)
        inactive_idx = None
        while True:
            datasets = []
            for i, stream in enumerate(streams):
                try:
                    dataset = next(stream)
                    datasets.append(dataset)
                except StopIteration:
                    inactive_idx = i
                    num_active -= 1
            if num_active == 0:
                return
            if len(datasets) != len(streams):
                raise ValueError('All dataset stream must have the same number of datasets.'\
                    ' While streams[%d] contains fewer datasets'%inactive_idx)
            yield ArrayDataset(*datasets)



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
    file_sampler : str or gluon.data.Sampler, defaults to 'random'
        The sampler used to sample a file. The following string values are supported:

        - 'sequential': SequentialSampler
        - 'random': RandomSampler
    kwargs
        All other keyword arguments are passed to the dataset constructor.
    """
    def __init__(self, dataset, file_pattern, file_sampler='random', **kwargs):
        if not isinstance(file_pattern, str):
            raise TypeError('file_pattern must be str, but got %s'%type(file_pattern))
        self._dataset = dataset
        file_pattern = os.path.expanduser(file_pattern)
        self._files = sorted(glob.glob(file_pattern))
        if len(self._files) == 0:
            raise ValueError('Cannot find any file with path "%s"'%file_pattern)
        self._file_sampler = self._get_sampler(file_sampler)
        self._kwargs = kwargs

    def _get_sampler(self, sampler):
        if isinstance(sampler, Sampler):
            return sampler
        if isinstance(sampler, str):
            length = len(self._files)
            if sampler == 'random':
                return RandomSampler(length)
            if sampler == 'sequential':
                return SequentialSampler(length)
        raise ValueError('file_sampler must be a supported str ("random", "sequential") or'
                         'a `gluon.data.Sampler`, but got %s'%(sampler))

    def __iter__(self):
        # generate file samples
        for file_idx in iter(self._file_sampler):
            filename = self._files[file_idx]
            yield self._dataset(filename, **self._kwargs)

class H5PyDatasetStream(DatasetStream):
    """A dataset stream wrapping over a h5py file.

    Parameters
    ----------
    filename : str
        Path to the .h5py file.
    dataset_sampler : str or gluon.data.Sampler, defaults to 'random'
        The sampler used to sample a file. The following string values are supported:

        - 'sequential': SequentialSampler
        - 'random': RandomSampler
    select : str or list of str, default is '.*'
        Regular expression to select a subset of datasets. If a list of str is provided,
        multiple datasets from the h5py is returned at each iteration.
    """
    def __init__(self, filename, dataset_sampler='random', select='.*'):
        import h5py
        if isinstance(select, _str_types):
            select = [select]
        patterns = [re.compile(s) for s in select]
        self._num_selects = len(patterns)
        self._data = h5py.File(filename, 'r')
        dataset_keys = self._collect_datasets(patterns)
        self._dataset_keys = [sorted(keys) for keys in dataset_keys]
        self._sampler = self._get_sampler(dataset_sampler)

    def _collect_datasets(self, patterns):
        # collect dataset
        import h5py
        dataset_keys = [[] for _ in range(self._num_selects)]
        def collect_datasets(name, dataset):
            if not isinstance(dataset, h5py.Dataset):
                return 
            for i in range(self._num_selects):
                if patterns[i].match(name):
                    dataset_keys[i].append(name)
        self._data.visititems(collect_datasets)
        return dataset_keys

    def _get_sampler(self, sampler):
        if isinstance(sampler, Sampler):
            return sampler
        if isinstance(sampler, str):
            length = len(self._dataset_keys[0])
            if sampler == 'random':
                return RandomSampler(length)
            if sampler == 'sequential':
                return SequentialSampler(length)
        raise ValueError('file_sampler must be a supported str ("random", "sequential") or'
                         'a `gluon.data.Sampler`, but got %s'%(sampler))

    def __iter__(self):
        # generate dataset samples
        for dataset_idx in iter(self._sampler):
            datasets = []
            for i in range(self._num_selects):
                dataset_key = self._dataset_keys[i][dataset_idx]
                datasets.append(SimpleDataset(self._data[dataset_key]))
            yield tuple(datasets) if len(datasets) > 1 else datasets[0]

    @property
    def keys(self):
        return self._dataset_keys

class _Prefetcher(object):
    """Internal shared prefetcher logic."""
    _dataq = None  # Data queue transmits prefetched elements
    _controlq = None  # Control queue to instruct thread / process shutdown
    _errorq = None  # Error queue to transmit exceptions from worker to master

    _checked_start = False  # True once startup has been checkd by _check_start

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
        if not isinstance(self, multiprocessing.Process):
            # Calling mxnet methods in a subprocess will raise an exception if
            # mxnet is built with GPU support
            # https://github.com/apache/incubator-mxnet/issues/4659
            mx.random.seed(self.mx_seed)

        # Startup - Master waits for this
        try:
            stream_iter = iter(self.stream)
            self._errorq.put(None)
        except Exception as e:  # pylint: disable=broad-except
            tb = traceback.format_exc()
            self._errorq.put((e, tb))

        # Async work
        while True:
            try:  # Check control queue
                c = self._controlq.get(False)
                if c is None:
                    break
                else:
                    raise RuntimeError('Got unexpected control code {}'.format(repr(c)))
            except queue.Empty:
                pass
            except RuntimeError as e:
                tb = traceback.format_exc()
                self._errorq.put((e, tb))
                self._dataq.put(None)

            try:
                data = next(stream_iter)
                error = None
            except Exception as e:  # pylint: disable=broad-except
                tb = traceback.format_exc()
                error = (e, tb)
                data = None
            finally:
                self._errorq.put(error)
                self._dataq.put(data)

    def __next__(self):
        next_item = self._dataq.get()
        next_error = self._errorq.get()

        if next_error is None:
            return next_item
        else:
            self._controlq.put(None)
            if isinstance(next_error[0], StopIteration):
                raise StopIteration
            else:
                return self._reraise(*next_error)

    def _reraise(self, e, tb):
        print('Reraising exception from Prefetcher', file=sys.stderr)
        print(tb, file=sys.stderr)
        raise e

    def _check_start(self):
        assert not self._checked_start
        self._checked_start = True
        next_error = self._errorq.get(block=True)
        if next_error is not None:
            self._reraise(*next_error)

    def next(self):
        return self.__next__()


class _ProcessPrefetcher(_Prefetcher, multiprocessing.Process):
    """Internal multi-processing prefetcher."""

    def __init__(self, *args, **kwargs):
        super(_ProcessPrefetcher, self).__init__(*args, **kwargs)
        self._dataq = multiprocessing.Queue(self.num_prefetch)
        self._controlq = multiprocessing.Queue()
        self._errorq = multiprocessing.Queue(self.num_prefetch)
        self.daemon = True
        self.start()
        self._check_start()


class _ThreadPrefetcher(_Prefetcher, threading.Thread):
    """Internal threaded prefetcher."""

    def __init__(self, *args, **kwargs):
        super(_ThreadPrefetcher, self).__init__(*args, **kwargs)
        self._dataq = queue.Queue(self.num_prefetch)
        self._controlq = queue.Queue()
        self._errorq = queue.Queue(self.num_prefetch)
        self.daemon = True
        self.start()
        self._check_start()


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
