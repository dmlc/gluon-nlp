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


import pickle
import io
import sys
import numpy as np
import multiprocessing
import multiprocessing.queues
import mxnet as mx
from mxnet.gluon.data.dataloader import Queue, SimpleQueue, DataLoader, _MultiWorkerIter


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


class _ShardedMultiWorkerIter(_MultiWorkerIter):
    """Interal multi-worker iterator for ShardedDataLoader."""
    def __init__(self, num_workers, dataset, batchify_fn, batch_sampler):
        assert num_workers > 0, "_MultiWorkerIter is not for {} workers".format(num_workers)
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

        # pre-fetch
        for _ in range(2 * self._num_workers):
            self._push_next()


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
    """

    def __init__(self, dataset, batch_size=None, shuffle=False, sampler=None,
                 last_batch=None, batch_sampler=None, batchify_fn=None, num_workers=0):
        super(ShardedDataLoader, self).__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                                                sampler=sampler, last_batch=last_batch,
                                                batch_sampler=batch_sampler,
                                                batchify_fn=batchify_fn,
                                                num_workers=num_workers)


    def __iter__(self):
        if self._num_workers == 0:
            generator = lambda: [(yield self._batchify_fn([self._dataset[idx] for idx in batch]))
                                 if not isinstance(batch[0], (list, tuple)) else
                                 (yield [self._batchify_fn([self._dataset[idx] for idx in shard])
                                         for shard in batch])
                                 for batch in self._batch_sampler]
            return generator()

        # multi-worker
        return _ShardedMultiWorkerIter(self._num_workers, self._dataset,
                                       self._batchify_fn, self._batch_sampler)



