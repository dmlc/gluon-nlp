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
"""DataLoader. An extension of Gluon data loader that allows multi-shard sampling."""
__all__ = ['ShardedDataLoader']

import io
import pickle
import multiprocessing
from multiprocessing.pool import ThreadPool
from mxnet import context
from mxnet.gluon.data.dataloader import ForkingPickler, _as_in_context
from mxnet.gluon.data.dataloader import default_mp_batchify_fn, default_batchify_fn
from mxnet.gluon.data import sampler as _sampler

_worker_dataset = None
def _worker_initializer(dataset):
    """Initializer for processing pool."""
    # global dataset is per-process based and only available in worker processes
    # this is only necessary to handle MXIndexedRecordIO because otherwise dataset
    # can be passed as argument
    global _worker_dataset
    _worker_dataset = dataset

def _worker_fn(samples, batchify_fn, dataset=None):
    """Function for processing data in worker process."""
    # pylint: disable=unused-argument
    # it is required that each worker process has to fork a new MXIndexedRecordIO handle
    # preserving dataset as global variable can save tons of overhead and is safe in new process
    global _worker_dataset
    if isinstance(samples[0], (list, tuple)):
        batch = [batchify_fn([_worker_dataset[i] for i in shard]) for shard in samples]
    else:
        batch = batchify_fn([_worker_dataset[i] for i in samples])
    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(batch)
    return buf.getvalue()

def _thread_worker_fn(samples, batchify_fn, dataset):
    """Threadpool worker function for processing data."""
    if isinstance(samples[0], (list, tuple)):
        batch = [batchify_fn([dataset[i] for i in shard]) for shard in samples]
    else:
        batch = batchify_fn([dataset[i] for i in samples])
    return batch

class _MultiWorkerIter(object):
    """Internal multi-worker iterator for DataLoader."""
    def __init__(self, worker_pool, batchify_fn, batch_sampler, pin_memory=False,
                 worker_fn=_worker_fn, prefetch=0, dataset=None):
        self._worker_pool = worker_pool
        self._batchify_fn = batchify_fn
        self._batch_sampler = batch_sampler
        self._data_buffer = {}
        self._rcvd_idx = 0
        self._sent_idx = 0
        self._iter = iter(self._batch_sampler)
        self._worker_fn = worker_fn
        self._pin_memory = pin_memory
        self._dataset = dataset
        # pre-fetch
        for _ in range(prefetch):
            self._push_next()

    def __len__(self):
        return len(self._batch_sampler)

    def _push_next(self):
        """Assign next batch workload to workers."""
        r = next(self._iter, None)
        if r is None:
            return
        async_ret = self._worker_pool.apply_async(
            self._worker_fn, (r, self._batchify_fn, self._dataset))
        self._data_buffer[self._sent_idx] = async_ret
        self._sent_idx += 1

    def __next__(self):
        self._push_next()
        if self._rcvd_idx == self._sent_idx:
            assert not self._data_buffer, 'Data buffer should be empty at this moment'
            raise StopIteration

        assert self._rcvd_idx < self._sent_idx, 'rcvd_idx must be smaller than sent_idx'
        assert self._rcvd_idx in self._data_buffer, 'fatal error with _push_next, rcvd_idx missing'
        ret = self._data_buffer.pop(self._rcvd_idx)
        batch = pickle.loads(ret.get()) if self._dataset is None else ret.get()
        if self._pin_memory:
            batch = _as_in_context(batch, context.cpu_pinned())
        self._rcvd_idx += 1
        return batch

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self


class ShardedDataLoader(object):
    """Loads data from a dataset and returns mini-batches of data.

    Parameters
    ----------
    dataset : Dataset
        Source dataset. Note that numpy and mxnet arrays can be directly used
        as a Dataset.
    batch_size : int
        Size of mini-batch.
    shuffle : bool
        Whether to shuffle the samples.
    sampler : Sampler
        The sampler to use. Either specify sampler or shuffle, not both.
    last_batch : {'keep', 'discard', 'rollover'}
        How to handle the last batch if batch_size does not evenly divide
        `len(dataset)`.

        keep - A batch with less samples than previous batches is returned.
        discard - The last batch is discarded if its incomplete.
        rollover - The remaining samples are rolled over to the next epoch.
    batch_sampler : Sampler
        A sampler that returns mini-batches. Do not specify batch_size,
        shuffle, sampler, and last_batch if batch_sampler is specified.
    batchify_fn : callable
        Callback function to allow users to specify how to merge samples
        into a batch. Defaults to `default_batchify_fn`::

            def default_batchify_fn(data):
                if isinstance(data[0], nd.NDArray):
                    return nd.stack(*data)
                elif isinstance(data[0], tuple):
                    data = zip(*data)
                    return [default_batchify_fn(i) for i in data]
                else:
                    data = np.asarray(data)
                    return nd.array(data, dtype=data.dtype)

    num_workers : int, default 0
        The number of multiprocessing workers to use for data preprocessing.
        `num_workers > 0` is not supported on Windows yet.
    pin_memory : boolean, default False
        If ``True``, the dataloader will copy NDArrays into pinned memory
        before returning them. Copying from CPU pinned memory to GPU is faster
        than from normal CPU memory.
    prefetch : int, default is `num_workers * 2`
        The number of prefetching batches only works if `num_workers` > 0.
        If `prefetch` > 0, it allow worker process to prefetch certain batches before
        acquiring data from iterators.
        Note that using large prefetching batch will provide smoother bootstrapping performance,
        but will consume more shared_memory. Using smaller number may forfeit the purpose of using
        multiple worker processes, try reduce `num_workers` in this case.
        By default it defaults to `num_workers * 2`.
    thread_pool : bool, default False
        If ``True``, use threading pool instead of multiprocessing pool. Using threadpool
        can avoid shared memory usage. If `DataLoader` is more IO bounded or GIL is not a killing
        problem, threadpool version may achieve better performance than multiprocessing.

    """
    def __init__(self, dataset, batch_size=None, shuffle=False, sampler=None,
                 last_batch=None, batch_sampler=None, batchify_fn=None,
                 num_workers=0, pin_memory=False, prefetch=None, thread_pool=False):
        self._dataset = dataset
        self._pin_memory = pin_memory
        self._thread_pool = thread_pool

        if batch_sampler is None:
            if batch_size is None:
                raise ValueError('batch_size must be specified unless ' \
                                 'batch_sampler is specified')
            if sampler is None:
                if shuffle:
                    sampler = _sampler.RandomSampler(len(dataset))
                else:
                    sampler = _sampler.SequentialSampler(len(dataset))
            elif shuffle:
                raise ValueError('shuffle must not be specified if sampler is specified')

            batch_sampler = _sampler.BatchSampler(
                sampler, batch_size, last_batch if last_batch else 'keep')
        elif batch_size is not None or shuffle or sampler is not None or \
                last_batch is not None:
            raise ValueError('batch_size, shuffle, sampler and last_batch must ' \
                             'not be specified if batch_sampler is specified.')

        self._batch_sampler = batch_sampler
        self._num_workers = num_workers if num_workers >= 0 else 0
        self._worker_pool = None
        self._prefetch = max(0, int(prefetch) if prefetch is not None else 2 * self._num_workers)
        if self._num_workers > 0:
            if self._thread_pool:
                self._worker_pool = ThreadPool(self._num_workers)
            else:
                self._worker_pool = multiprocessing.Pool(
                    self._num_workers, initializer=_worker_initializer, initargs=[self._dataset])
        if batchify_fn is None:
            if num_workers > 0:
                self._batchify_fn = default_mp_batchify_fn
            else:
                self._batchify_fn = default_batchify_fn
        else:
            self._batchify_fn = batchify_fn

    def __iter__(self):
        if self._num_workers == 0:
            def _same_process_iter():
                for batch in self._batch_sampler:
                    if isinstance(batch[0], (list, tuple)):
                        rets = [self._batchify_fn([self._dataset[idx] for idx in shard])
                                for shard in batch]
                        if self._pin_memory:
                            rets = [_as_in_context(ret, context.cpu_pinned()) for ret in rets]
                        yield rets
                    else:
                        ret = self._batchify_fn([self._dataset[idx] for idx in batch])
                        if self._pin_memory:
                            ret = _as_in_context(ret, context.cpu_pinned())
                        yield ret
            return _same_process_iter()

        # multi-worker
        return _MultiWorkerIter(self._worker_pool, self._batchify_fn, self._batch_sampler,
                                pin_memory=self._pin_memory,
                                worker_fn=_thread_worker_fn if self._thread_pool else _worker_fn,
                                prefetch=self._prefetch,
                                dataset=self._dataset if self._thread_pool else None)

    def __len__(self):
        return len(self._batch_sampler)

    def __del__(self):
        if self._worker_pool:
            # manually terminate due to a bug that pool is not automatically terminated
            # https://bugs.python.org/issue34172
            assert isinstance(self._worker_pool, multiprocessing.pool.Pool)
            self._worker_pool.terminate()
