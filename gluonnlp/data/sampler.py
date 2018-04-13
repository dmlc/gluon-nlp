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
"""Samplers. They define how the samples in a dataset will be iterated
(e.g. in the order sorted by length). They can also be used to perform bucketing
for speeding up the processing of variable-length sequences."""
__all__ = ['SortedSampler', 'FixedBucketSampler', 'SortedBucketSampler']

import warnings
import numpy as np
from mxnet.gluon.data import Sampler


def _match_bucket_keys(bucket_keys, seq_lengths):
    bucket_key_npy = np.array(bucket_keys, dtype=np.int32)
    bucket_sample_ids = [list() for _ in range(len(bucket_keys))]
    batch_size = 10000
    bucket_key_npy = bucket_key_npy.reshape((1,) + bucket_key_npy.shape)
    for begin in range(0, len(seq_lengths), batch_size):
        end = min(begin + batch_size, len(seq_lengths))
        diff = bucket_key_npy - np.expand_dims(seq_lengths[begin:end], axis=1)
        if diff.ndim == 3:
            is_valid_bucket = np.prod(diff >= 0, axis=2)
            pad_val = np.sum(diff, axis=2)
        else:
            is_valid_bucket = diff >= 0
            pad_val = diff
        seq_ids_not_found = np.nonzero(is_valid_bucket.sum(axis=1) == 0)[0].tolist()
        masked_pad_val = np.ma.array(pad_val, mask=1 - is_valid_bucket)
        batch_bucket_id = masked_pad_val.argmin(axis=1).tolist()
        if len(seq_ids_not_found) > 0:
            raise ValueError('Find elements in seq_lengths that cannot fit in the '
                             'given buckets, seq_length=%s, bucket_keys=%s. ' \
                             'You must increase the bucket size.'
                             % (str(seq_lengths[seq_ids_not_found]), str(bucket_keys)))
        for i, bucket_id in enumerate(batch_bucket_id):
            bucket_sample_ids[bucket_id].append(i + begin)
    return bucket_sample_ids


class SortedSampler(Sampler):
    r"""Sort the samples based on the sort key and then sample sequentially.

    Parameters
    ----------
    sort_keys : list-like object
        List of the sort keys.
    reverse : bool, default True
        Whether to sort by descending order.
    """
    def __init__(self, sort_keys, reverse=True):
        assert len(sort_keys) > 0
        self._sorted_ids = sorted(range(len(sort_keys)),
                                  key=lambda i: sort_keys[i], reverse=reverse)

    def __iter__(self):
        return iter(self._sorted_ids)

    def __len__(self):
        return len(self._sorted_ids)


class FixedBucketSampler(Sampler):
    r"""Assign each data sample to a fixed bucket based on its length.
    The bucket keys are either given or generated from the input sequence lengths.

    Parameters
    ----------
    lengths : list of int or list of tuple/list of int
        The length of the sequences in the input data sample.
    batch_size : int
        The batch size of the sampler.
    num_buckets : int or None, default 10
        The number of buckets. This will not be used if bucket_keys is set.
    bucket_keys : None or list of int or list of tuple, default None
        The keys that will be used to create the buckets. It should usually be the lengths of the
        sequences. If it is None, the bucket_keys will be generated based on the maximum
        lengths of the data.
    ratio : float, default 0
        Ratio to scale up the batch size of smaller buckets.
        Assume the :math:`i` th key is :math:`K_i` ,
        the default batch size is :math:`B` , the ratio to scale the batch size is
        :math:`\alpha` and
        the batch size corresponds to the :math:`i` th bucket is :math:`B_i` . We have:

        .. math::

            B_i = \max(\alpha B \times \frac{\max_j sum(K_j)}{sum(K_i)}, B)

        Thus, setting this to a value larger than 0, like 0.5, will scale up the batch size of the
        smaller buckets.
    shuffle : bool, default False
        Whether to shuffle the batches.

    Examples
    --------
    >>> from gluonnlp.data import FixedBucketSampler
    >>> import numpy as np
    >>> lengths = [np.random.randint(1, 100) for _ in range(1000)]
    >>> sampler = FixedBucketSampler(lengths, 8)
    >>> print(sampler.stats())
    FixedBucketSampler:
      sample_num=1000, batch_num=128
      key=[9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
      cnt=[95, 103, 91, 97, 86, 79, 102, 100, 128, 119]
      batch_size=[8, 8, 8, 8, 8, 8, 8, 8, 8, 8]
    >>> sampler = FixedBucketSampler(lengths, 8, ratio=0.5)
    >>> print(sampler.stats())
    FixedBucketSampler:
      sample_num=1000, batch_num=104
      key=[9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
      cnt=[95, 103, 91, 97, 86, 79, 102, 100, 128, 119]
      batch_size=[44, 20, 13, 10, 8, 8, 8, 8, 8, 8]
    """
    def __init__(self, lengths, batch_size, num_buckets=10, bucket_keys=None,
                 ratio=0, shuffle=False):
        assert len(lengths) > 0, 'FixedBucketSampler does not support empty lengths.'
        assert batch_size > 0, 'Batch size must be larger than 0.'
        assert ratio >= 0, 'batch size scaling ratio cannot be negative.'
        self._batch_size = batch_size
        self._ratio = ratio
        self._lengths = np.array(lengths, dtype=np.int32)
        if self._lengths.ndim == 1:
            self._single_element = True
            attr_num = 1
        else:
            assert self._lengths.ndim == 2, \
                'Elements in lengths must be either int or tuple/list of int. ' \
                'Received lengths=%s' % str(lengths)
            self._single_element = False
            attr_num = self._lengths.shape[1]
        self._shuffle = shuffle
        max_lengths = self._lengths.max(axis=0)
        min_lengths = self._lengths.min(axis=0)
        if self._single_element:
            assert min_lengths > 0, 'Sequence lengths must all be larger than 0.'
        else:
            for _, ele in enumerate(min_lengths):
                assert ele > 0, 'Sequence lengths must all be larger than 0.'
        # Generate the buckets
        if bucket_keys is None:
            assert num_buckets > 0, 'num_buckets must be set when bucket_keys is None. Received ' \
                                    'num_buckets=%d' % num_buckets
            if not self._single_element:
                bucket_width_l = [max((max_len - min_len) // max(num_buckets - 1, 1), 1)
                                  for max_len, min_len in
                                  zip(max_lengths, min_lengths)]
                bucket_keys =\
                    [tuple(max(max_len - i * width, min_len) for max_len, min_len, width in
                           zip(max_lengths, min_lengths, bucket_width_l))
                     for i in range(num_buckets)]
            else:
                bucket_width = max((max_lengths - min_lengths) // max(num_buckets - 1, 1), 1)
                bucket_keys = [max(max_lengths - i * bucket_width, min_lengths)
                               for i in range(num_buckets)]
        else:
            if num_buckets is not None:
                warnings.warn('num_buckets will not be used if bucket_keys is not None. '
                              'bucket_keys=%s, num_buckets=%d' % (str(bucket_keys), num_buckets))
            assert len(bucket_keys) > 0
            if self._single_element:
                assert isinstance(bucket_keys[0], int)
            else:
                assert isinstance(bucket_keys[0], tuple)
                assert len(bucket_keys[0]) == attr_num
        bucket_keys = sorted(set(bucket_keys))
        # Assign instances to buckets
        bucket_sample_ids = _match_bucket_keys(bucket_keys, self._lengths)
        unused_bucket_keys = [key for key, sample_ids in zip(bucket_keys, bucket_sample_ids)
                              if len(sample_ids) == 0]
        if len(unused_bucket_keys) > 0:
            warnings.warn('Some buckets are empty and will be removed. Unused bucket keys=%s' %
                          str(unused_bucket_keys))
        # Remove empty buckets
        self._bucket_keys = [key for key, sample_ids in zip(bucket_keys, bucket_sample_ids)
                             if len(sample_ids) > 0]

        self._bucket_sample_ids = [sample_ids for sample_ids in bucket_sample_ids
                                   if len(sample_ids) > 0]
        scale_up_keys = [key if self._single_element else sum(key) for key in self._bucket_keys]
        max_scale_up_key = max(scale_up_keys)
        self._bucket_batch_sizes = [max(int(max_scale_up_key / float(scale_up_key)
                                            * self._ratio * batch_size), batch_size)
                                    for scale_up_key in scale_up_keys]
        self._batch_infos = []
        for bucket_id, sample_ids, bucket_batch_size in\
                zip(range(len(self._bucket_keys) - 1, -1, -1),
                        self._bucket_sample_ids[::-1],
                        self._bucket_batch_sizes[::-1]):
            for i in range(0, len(sample_ids), bucket_batch_size):
                self._batch_infos.append((bucket_id, i))

    def __iter__(self):
        if self._shuffle:
            np.random.shuffle(self._batch_infos)
        for bucket_id, batch_begin in self._batch_infos:
            batch_size = self._bucket_batch_sizes[bucket_id]
            batch_end = min(batch_begin + batch_size, len(self._bucket_sample_ids[bucket_id]))
            yield self._bucket_sample_ids[bucket_id][batch_begin:batch_end]

    def __len__(self):
        return len(self._batch_infos)

    def stats(self):
        """Return a string representing the statistics of the bucketing sampler.

        Returns
        -------
        ret : str
            String representing the statistics of the buckets.
        """
        ret = '{name}:\n' \
            '  sample_num={sample_num}, batch_num={batch_num}\n' \
            '  key={bucket_keys}\n' \
            '  cnt={bucket_counts}\n' \
            '  batch_size={bucket_batch_sizes}'\
            .format(name=self.__class__.__name__,
                    sample_num=len(self._lengths),
                    batch_num=len(self._batch_infos),
                    bucket_keys=self._bucket_keys,
                    bucket_counts=[len(sample_ids) for sample_ids in self._bucket_sample_ids],
                    bucket_batch_sizes=self._bucket_batch_sizes)
        return ret


class SortedBucketSampler(Sampler):
    r"""Batches are samled from sorted buckets of data.

    First, partition data in buckets of size `batch_size * mult`.
    Each bucket contains `batch_size * mult` elements. The samples inside each bucket are sorted
    based on sort_key and then batched.

    Parameters
    ----------
    sort_keys : list-like object
        The keys to sort the samples.
    batch_size : int
        Batch size of the sampler.
    mult : int or float, default 100
        The multiplier to determine the bucket size. Each bucket will have size `mult * batch_size`.
    reverse : bool, default True
        Whether to sort in descending order.
    shuffle : bool, default False
        Whether to shuffle the data.

    Examples
    --------
    >>> from gluonnlp.data import SortedBucketSampler
    >>> import numpy as np
    >>> lengths = [np.random.randint(1, 1000) for _ in range(1000)]
    >>> sampler = SortedBucketSampler(lengths, 16)
    >>> # The sequence lengths within the batch will be sorted
    >>> for i, indices in enumerate(sampler):
    ...     if i == 0:
    ...         print([lengths[ele] for ele in indices])
    [999, 999, 999, 997, 997, 996, 995, 993, 991, 991, 989, 989, 987, 987, 986, 985]
    """
    def __init__(self, sort_keys, batch_size, mult=100, reverse=True, shuffle=False):
        assert len(sort_keys) > 0
        assert batch_size > 0
        assert mult >= 1, 'Bucket size multiplier must be larger than 1'
        self._sort_keys = sort_keys
        self._batch_size = batch_size
        self._mult = mult
        self._total_sample_num = len(self._sort_keys)
        self._reverse = reverse
        self._shuffle = shuffle

    def __iter__(self):
        if self._shuffle:
            sample_ids = np.random.permutation(self._total_sample_num)
        else:
            sample_ids = list(range(self._total_sample_num))
        bucket_size = int(self._mult * self._batch_size)
        for bucket_begin in range(0, self._total_sample_num, bucket_size):
            bucket_end = min(bucket_begin + bucket_size, self._total_sample_num)
            sorted_sample_ids = sorted(sample_ids[bucket_begin:bucket_end],
                                       key=lambda i: self._sort_keys[i], reverse=self._reverse)
            batch_begins = list(range(0, len(sorted_sample_ids), self._batch_size))
            if self._shuffle:
                np.random.shuffle(batch_begins)
            for batch_begin in batch_begins:
                batch_end = min(batch_begin + self._batch_size, len(sorted_sample_ids))
                yield sorted_sample_ids[batch_begin:batch_end]

    def __len__(self):
        return (len(self._sort_keys) + self._batch_size - 1) // self._batch_size
