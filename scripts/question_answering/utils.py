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

"""Various utility methods for Question Answering"""
import collections
import contextlib
import itertools
import math
import multiprocessing as mp


def warm_up_lr(base_lr, iteration, lr_warmup_steps):
    """Returns learning rate based on current iteration. Used to implement learning rate warm up
    technique

    Parameters
    ----------
    base_lr : float
        Initial learning rage
    iteration : int
        Current iteration number
    lr_warmup_steps : int
        Learning rate warm up steps

    Returns
    -------
    learning_rate : float
        Learning rate
    """
    return min(base_lr, base_lr * (math.log(iteration) / math.log(lr_warmup_steps)))


class MapReduce:
    """A multiprocessing implementation of map-reduce processing flow"""
    def __init__(self, map_func, reduce_func, num_workers=None):
        """Init MapReduce object

        Parameters
        ----------
        map_func : Callable
            Map function that produces a list of (key, value) tuples
        reduce_func : Callable
            Reducing function that produces a (key, result) tuple
        num_workers : int
            A number of workers to use, if not provided by an initialized multiprocessing pool
        """
        self._map_func = map_func
        self._reduce_func = reduce_func
        self._num_workers = num_workers

    def __call__(self, inputs, pool=None):
        """Does map-reduce processing

        Parameters
        ----------
        inputs : list
            List of records to process
        pool : Pool, default None
            Multiprocessing pool of workers to use in both map and reduce step. If None, a new pool
            will be created using `num_workers` provided to the __init__ method

        Returns
        -------
        reduced_values : List[Tuple]
            Result of map-reduce process. Each tuple of (key, result) format
        """
        if pool:
            map_responses = pool.map(self._map_func, inputs)
        else:
            with contextlib.closing(mp.Pool(self._num_workers)) as p:
                map_responses = p.map(self._map_func, inputs)

        partitions = self._partition(
            itertools.chain(*map_responses)
        )

        if pool:
            reduced_values = pool.map(self._reduce_func, partitions)
        else:
            with contextlib.closing(mp.Pool(self._num_workers)) as p:
                reduced_values = p.map(self._reduce_func, partitions)

        return reduced_values

    @staticmethod
    def _partition(mapped_values):
        """Groups items with same keys into a single partition

        Parameters
        ----------
        mapped_values : List[Tuple]
            List of mapped (key, value) tuples

        Returns
        -------
        items: List[Tuple]
            List of partitions, where each partition is (key, List[value])
        """
        partitioned_data = collections.defaultdict(list)
        for key, value in mapped_values:
            partitioned_data[key].append(value)
        return partitioned_data.items()
