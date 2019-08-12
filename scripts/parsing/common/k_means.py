#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright 2016 Timothy Dozat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""KMeans utility."""

from collections import Counter

import numpy as np


class KMeans:
    """
    Cluster sentences by their lengths

    Parameters
    ----------
    k : int
        number of clusters
    len_cntr : Counter
        length counter
    """
    def __init__(self, k, len_cntr):
        # Error checking
        if len(len_cntr) < k:
            raise ValueError('Trying to sort %d data points into %d buckets' % (len(len_cntr), k))

        # Initialize variables
        self._k = k
        self._len_cntr = len_cntr
        self._lengths = sorted(self._len_cntr.keys())
        self._splits = []
        self._split2len_idx = {}
        self._len2split_idx = {}
        self._split_cntr = Counter()

        # Initialize the splits evenly
        lengths = []
        unique_length = []
        for length, count in list(self._len_cntr.items()):
            lengths.extend([length] * count)
            unique_length.append(length)
        lengths.sort()
        unique_length.sort()
        self._splits = [np.max(split) for split in np.array_split(lengths, self._k)]

        i = len(self._splits) - 1
        while i > 0:
            while self._splits[i - 1] >= self._splits[i]:
                index = unique_length.index(self._splits[i - 1])
                if index == 0:
                    break
                self._splits[i - 1] = unique_length[index - 1]
            i -= 1

        unique_length.reverse()
        i = 1
        while i < len(self._splits) - 1:
            while self._splits[i] <= self._splits[i - 1]:
                index = unique_length.index(self._splits[i])
                if index == 0:
                    break
                self._splits[i] = unique_length[index - 1]
            i += 1

        # Reindex everything
        split_idx = 0
        split = self._splits[split_idx]
        for len_idx, length in enumerate(self._lengths):
            count = self._len_cntr[length]
            self._split_cntr[split] += count
            if length == split:
                self._split2len_idx[split] = len_idx
                split_idx += 1
                if split_idx < len(self._splits):
                    split = self._splits[split_idx]
                    self._split_cntr[split] = 0
            elif length > split:
                raise IndexError()

        # Iterate
        old_splits = None
        # print('0) Initial splits: %s; Initial mass: %d' % (self._splits, self.get_mass()))
        i = 0
        while self._splits != old_splits:
            old_splits = list(self._splits)
            self._recenter()
            i += 1
        # print('%d) Final splits: %s; Final mass: %d' % (i, self._splits, self.get_mass()))

        self._reindex()

    def _recenter(self):
        """
        one iteration of k-means
        """
        for split_idx in range(len(self._splits)):
            split = self._splits[split_idx]
            len_idx = self._split2len_idx[split]
            if split == self._splits[-1]:
                continue
            right_split = self._splits[split_idx + 1]

            # Try shifting the centroid to the left
            if len_idx > 0 and self._lengths[len_idx - 1] not in self._split_cntr:
                new_split = self._lengths[len_idx - 1]
                left_delta = (self._len_cntr[split] * (right_split - new_split)
                              - self._split_cntr[split] * (split - new_split))
                if left_delta < 0:
                    self._splits[split_idx] = new_split
                    self._split2len_idx[new_split] = len_idx - 1
                    del self._split2len_idx[split]
                    self._split_cntr[split] -= self._len_cntr[split]
                    self._split_cntr[right_split] += self._len_cntr[split]
                    self._split_cntr[new_split] = self._split_cntr[split]
                    del self._split_cntr[split]

            # Try shifting the centroid to the right
            elif len_idx < len(self._lengths) - 2 \
                and self._lengths[len_idx + 1] not in self._split_cntr:
                new_split = self._lengths[len_idx + 1]
                right_delta = (self._split_cntr[split] * (new_split - split)
                               - self._len_cntr[split] * (new_split - split))
                if right_delta <= 0:
                    self._splits[split_idx] = new_split
                    self._split2len_idx[new_split] = len_idx + 1
                    del self._split2len_idx[split]
                    self._split_cntr[split] += self._len_cntr[split]
                    self._split_cntr[right_split] -= self._len_cntr[split]
                    self._split_cntr[new_split] = self._split_cntr[split]
                    del self._split_cntr[split]

    def _reindex(self):
        """
        Index every sentence into a cluster
        """
        self._len2split_idx = {}
        last_split = -1
        for split_idx, split in enumerate(self._splits):
            self._len2split_idx.update(
                dict(list(zip(list(range(last_split + 1, split)),
                              [split_idx] * (split - (last_split + 1))))))

    def __len__(self):
        return self._k

    def __iter__(self):
        return (split for split in self.splits)

    def __getitem__(self, key):
        return self._splits[key]

    @property
    def splits(self):
        """Get clusters

        Returns
        -------
        tuple
            (bucket, length) mapping
        """
        return self._splits

    @property
    def len2split_idx(self):
        """Get length to bucket mapping

        Returns
        -------
        tuple
             (length, bucket) mapping
        """
        return self._len2split_idx
