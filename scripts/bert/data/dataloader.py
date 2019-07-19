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

# coding: utf-8
# pylint: disable=ungrouped-imports
"""Dataset generator."""

__all__ = ['DatasetLoader', 'SamplerFn', 'DatasetFn', 'DataLoaderFn']

import multiprocessing
from gluonnlp.data.stream import _PathDataset

class DatasetFn:
    """Callable object to generate a gluon.data.Dataset given a url.

    Subclasses should override the __call__ method.
    """
    def __call__(self, dataset_url):
        raise NotImplementedError

class SamplerFn:
    """Callable object to generate a gluon.data.sampler.Sampler given a dataset.

    Subclasses should override the __call__ method.
    """
    def __call__(self, dataset):
        raise NotImplementedError

class DataLoaderFn:
    """Callable object to generate a DataLoader object given a dataset and sampler.

    Subclasses should override the __call__ method.
    """
    def __call__(self, dataset, sampler):
        raise NotImplementedError

class SimpleDataLoaderFn:
    """A simple callable object that geneartes a data loader by applying
    dataloader_cls(dataset, batch_sampler=sampler, **dataset_params)
    """
    def __init__(self, dataloader_cls, dataloader_params):
        self._dataloader_cls = dataloader_cls
        self._dataloader_params = dataloader_params

    def __call__(self, dataset, sampler):
        return self._dataloader_cls(dataset, batch_sampler=sampler,
                                    **self._dataloader_params)

class SimpleDatasetFn(DatasetFn):
    """A simple callable object that geneartes a dataset by applying
    dataset_cls(url, **dataset_params)
    """
    def __init__(self, dataset_cls, dataset_params):
        self._dataset_cls = dataset_cls
        self._dataset_params = dataset_params

    def __call__(self, dataset_url):
        return self._dataset_cls(dataset_url, **self._dataset_params)

def _worker_fn(url, dataset_fn, sampler_fn):
    """Function to generate the dataset and sampler for each worker."""
    dataset = dataset_fn(url)
    sampler = sampler_fn(dataset)
    return (dataset, sampler)

class _MultiWorkerIter:
    """Internal multi-worker iterator for DataLoader."""
    def __init__(self, worker_pool, worker_fn, dataset, file_sampler,
                 dataset_fn, sampler_fn, dataloader_fn, prefetch):
        self._worker_pool = worker_pool
        self._worker_fn = worker_fn
        self._dataset = dataset
        self._dataset_fn = dataset_fn
        self._sampler_fn = sampler_fn
        self._dataloader_fn = dataloader_fn
        self._prefetch = prefetch

        # send and receive index for datasets
        self._rcvd_idx = 0
        self._sent_idx = 0
        self._data_buffer = {}

        self._dataset_iter = iter(file_sampler)
        self._num_datasets = len(self._dataset)

        # need to keep a reference of the dataloader
        self._dataloader_ref = None
        self._dataloader = None

        # pre-fetch
        for _ in range(self._prefetch):
            self._push_next_dataset()

    def _push_next_dataset(self):
        """Assign next dataset workload to workers."""
        if self._sent_idx < len(self._dataset):
            url = self._dataset[self._sent_idx]
        else:
            return
        # push to worker asynchronously
        async_ret = self._worker_pool.apply_async(
            self._worker_fn, (url, self._dataset_fn, self._sampler_fn))
        # data buffer stores the async result
        self._data_buffer[self._sent_idx] = async_ret
        self._sent_idx += 1

    def _next_dataset(self):
        """Retrieve the next dataset. Returns None if no dataset is available."""
        if self._rcvd_idx == self._sent_idx:
            assert not self._data_buffer, 'Data buffer should be empty at this moment'
            return None

        assert self._rcvd_idx < self._sent_idx, \
               'rcvd_idx must be smaller than sent_idx'
        assert self._rcvd_idx in self._data_buffer, \
               'fatal error with _next_dataset, rcvd_idx missing'

        ret = self._data_buffer.pop(self._rcvd_idx)
        dataset, sampler = ret.get()
        self._rcvd_idx += 1
        return dataset, sampler

    def __next__(self):
        """Next mini-batch"""
        while True:
            if self._dataloader_ref is None:
                # load next dataset and create a data loader
                self._push_next_dataset()
                result = self._next_dataset()

                if result is None:
                    raise StopIteration

                dataset, sampler = result
                self._dataloader_ref = self._dataloader_fn(dataset, sampler)
                self._dataloader = iter(self._dataloader_ref)
            try:
                # load next mini-batch from the dataloader
                result = next(self._dataloader)
                return result
            except StopIteration:
                self._dataloader = None
                self._dataloader_ref = None

    def next(self):
        """Next mini-batch"""
        return self.__next__()

    def __iter__(self):
        """Returns the iterator object"""
        return self


class DatasetLoader:
    """Loads data from a list of datasets and returns mini-batches of data.

    One dataset is loaded at a time.

    Parameters
    ----------
    file_pattern: str
        Path to the input text files.
    file_sampler : str or gluon.data.Sampler, defaults to 'random'
        The sampler used to sample a file. The following string values are supported:

        - 'sequential': SequentialSampler
        - 'random': RandomSampler
    dataset_fn : DatasetFn, callable
        Callable object to generate a gluon.data.Dataset given a url.
    sampler_fn : SamplerFn, callable
        Callable object to generate a gluon.data.sampler.Sampler given a dataset.
    dataloader_fn : DataloaderFn, callable
        Callable object to generate a data loader object given a url.
    num_dataset_workers : int
        Number of worker process for dataset creation.
    prefetch : int, default is `num_dataset_workers`
        The number of prefetching datasets only works if `num_workers` > 0.
        If `prefetch` > 0, it allow worker process to prefetch certain datasets before
        acquiring data from iterators.
        Note that using large prefetching batch will provide smoother bootstrapping performance,
        but will consume more memory. Using smaller number may forfeit the purpose of using
        multiple worker processes, try reduce `num_workers` in this case.
        By default it defaults to `num_workers`.
    """
    def __init__(self, file_patterns, file_sampler, dataset_fn,
                 sampler_fn, dataloader_fn, num_dataset_workers=1, prefetch=None):
        self._dataset = _PathDataset(file_patterns)
        self._file_sampler = file_sampler
        self._dataset_fn = dataset_fn
        self._sampler_fn = sampler_fn
        self._dataloader_fn = dataloader_fn
        self._num_dataset_workers = num_dataset_workers
        self._prefetch = max(0, int(prefetch) if prefetch is not None else num_dataset_workers)
        self._worker_pool = None
        if self._num_dataset_workers > 0:
            self._worker_pool = multiprocessing.Pool(self._num_dataset_workers)
        assert self._num_dataset_workers >= 0, \
               'num_dataset_workers must be non-negative'
        assert isinstance(sampler_fn, SamplerFn), \
               'sampler_fn must be an instance of SamplerFn'
        assert isinstance(dataloader_fn, DataLoaderFn), \
               'dataloader_fn must be an instance of DataLoaderFn'

    def __iter__(self):
        if self._num_dataset_workers == 0:
            def _same_process_iter():
                for idx in self._file_sampler:
                    url = self._dataset[idx]
                    dataset, sampler = _worker_fn(url, self._dataset_fn, self._sampler_fn)
                    dataloader = self._dataloader_fn(dataset, sampler)
                    for batch in dataloader:
                        yield batch
            return _same_process_iter()

        # multi-worker
        return _MultiWorkerIter(self._worker_pool,
                                worker_fn=_worker_fn,
                                dataset=self._dataset,
                                file_sampler=self._file_sampler,
                                dataset_fn=self._dataset_fn,
                                sampler_fn=self._sampler_fn,
                                dataloader_fn=self._dataloader_fn,
                                prefetch=self._prefetch)

    def __del__(self):
        if self._worker_pool:
            # manually terminate due to a bug that pool is not automatically terminated
            # https://bugs.python.org/issue34172
            assert isinstance(self._worker_pool, multiprocessing.pool.Pool)
            self._worker_pool.terminate()
