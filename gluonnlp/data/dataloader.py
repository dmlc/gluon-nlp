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

import sys
import multiprocessing
import multiprocessing.queues
import threading
from mxnet import context
from mxnet.gluon.data.dataloader import Queue, SimpleQueue, DataLoader, \
    fetcher_loop, _as_in_context


def worker_loop(dataset, key_queue, data_queue, batchify_fn):
    """Worker loop for multiprocessing DataLoader."""
    dataset._fork()
    while True:
        idx, samples = key_queue.get()
        if idx is None:
            break
        if isinstance(samples[0], (list, tuple)):
            batch = [batchify_fn([dataset[i] for i in shard]) for shard in samples]
        else:
            batch = batchify_fn([dataset[i] for i in samples])
        data_queue.put((idx, batch))


class _ShardedMultiWorkerIter(object):
    """Interal multi-worker iterator for ShardedDataLoader."""
    def __init__(self, num_workers, dataset, batchify_fn, batch_sampler, pin_memory=False):
        assert num_workers > 0, '_MultiWorkerIter is not for {} workers'.format(num_workers)
        self._num_workers = num_workers
        self._dataset = dataset
        self._batchify_fn = batchify_fn
        self._batch_sampler = batch_sampler
        self._key_queue = Queue()
        self._data_queue = Queue() if sys.version_info[0] <= 2 else SimpleQueue()
        self._data_buffer = {}
        self._rcvd_idx = 0
        self._sent_idx = 0
        self._iter = iter(self._batch_sampler)
        self._shutdown = False

        workers = []
        for _ in range(self._num_workers):
            worker = multiprocessing.Process(
                target=worker_loop,
                args=(self._dataset, self._key_queue, self._data_queue, self._batchify_fn))
            worker.daemon = True
            worker.start()
            workers.append(worker)

        self._fetcher = threading.Thread(
            target=fetcher_loop,
            args=(self._data_queue, self._data_buffer, pin_memory))
        self._fetcher.daemon = True
        self._fetcher.start()

        # pre-fetch
        for _ in range(2 * self._num_workers):
            self._push_next()

    def __len__(self):
        return len(self._batch_sampler)

    def __del__(self):
        self.shutdown()

    def _push_next(self):
        """Assign next batch workload to workers."""
        r = next(self._iter, None)
        if r is None:
            return
        self._key_queue.put((self._sent_idx, r))
        self._sent_idx += 1

    def __next__(self):
        assert not self._shutdown, 'call __next__ after shutdown is forbidden'
        if self._rcvd_idx == self._sent_idx:
            assert not self._data_buffer, 'Data buffer should be empty at this moment'
            self.shutdown()
            raise StopIteration

        while True:
            if self._rcvd_idx in self._data_buffer:
                batch = self._data_buffer.pop(self._rcvd_idx)
                self._rcvd_idx += 1
                self._push_next()
                return batch

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

    def shutdown(self):
        """Shutdown internal workers by pushing terminate signals."""
        if not self._shutdown:
            for _ in range(self._num_workers):
                self._key_queue.put((None, None))
            self._data_queue.put((None, None))
            self._shutdown = True


class ShardedDataLoader(DataLoader):
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
    """

    def __init__(self, dataset, batch_size=None, shuffle=False, sampler=None,
                 last_batch=None, batch_sampler=None, batchify_fn=None,
                 num_workers=0, pin_memory=False):
        super(ShardedDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                                                sampler=sampler, last_batch=last_batch,
                                                batch_sampler=batch_sampler,
                                                batchify_fn=batchify_fn,
                                                num_workers=num_workers,
                                                pin_memory=pin_memory)


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
        return _ShardedMultiWorkerIter(self._num_workers, self._dataset,
                                       self._batchify_fn, self._batch_sampler, self._pin_memory)
