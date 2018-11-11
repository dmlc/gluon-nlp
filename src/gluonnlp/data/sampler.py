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
__all__ = ['ConstWidthBucket', 'LinearWidthBucket', 'ExpWidthBucket',
           'SortedSampler', 'FixedBucketSampler', 'SortedBucketSampler']

import math
import warnings
import numpy as np
from mxnet.gluon.data import Sampler
from .._constants import INT_TYPES


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


def _bucket_stats(bucket_sample_ids, seq_lengths):
    bucket_average_lengths = []
    bucket_length_stds = []
    for sample_ids in bucket_sample_ids:
        if len(sample_ids) > 0:
            lengths = seq_lengths[sample_ids]
            bucket_average_lengths.append(np.mean(lengths))
            bucket_length_stds.append(np.std(lengths))
        else:
            bucket_average_lengths.append(0)
            bucket_length_stds.append(0)
    return (bucket_average_lengths, bucket_length_stds)


class BucketScheme(object):
    r"""Base class for generating bucket keys."""
    def __call__(self, max_lengths, min_lengths, num_buckets):
        """Generate bucket keys based on the lengths of sequences and number of buckets.

        Parameters
        ----------
        max_lengths : int of list of int
            Maximum of lengths of sequences.
        min_lengths : int of list of int
            Minimum of lengths of sequences.
        num_buckets : int
            Number of buckets

        Returns
        -------
        bucket_keys : list of int
            A list including the keys of the buckets.
        """
        raise NotImplementedError


class ConstWidthBucket(BucketScheme):
    r"""Buckets with constant width."""
    def __call__(self, max_lengths, min_lengths, num_buckets):
        r"""This generate bucket keys given that all the buckets have the same width.

        Parameters
        ----------
        max_lengths : int of list of int
            Maximum of lengths of sequences.
        min_lengths : int of list of int
            Minimum of lengths of sequences.
        num_buckets : int
            Number of buckets

        Returns
        -------
        bucket_keys : list of int
            A list including the keys of the buckets.
        """
        if not isinstance(max_lengths, INT_TYPES):
            bucket_width_l = [max((1 + max_len - min_len) // num_buckets, 1)
                              for max_len, min_len in
                              zip(max_lengths, min_lengths)]
            bucket_keys = \
                [tuple(max(max_len - i * width, min_len) for max_len, min_len, width in
                       zip(max_lengths, min_lengths, bucket_width_l))
                 for i in range(num_buckets)]
        else:
            bucket_width = max((1 + max_lengths - min_lengths) // num_buckets, 1)
            bucket_keys = [max(max_lengths - i * bucket_width, min_lengths)
                           for i in range(num_buckets)]
        return bucket_keys


class LinearWidthBucket(BucketScheme):
    r""" Buckets with linearly increasing width:
    :math:`w_i = \alpha * i + 1` for all :math:`i \geq 1`.
    """
    def __call__(self, max_lengths, min_lengths, num_buckets):
        r"""This function generates bucket keys with linearly increasing bucket width:

        Parameters
        ----------
        max_lengths : int of list of int
            Maximum of lengths of sequences.
        min_lengths : int of list of int
            Minimum of lengths of sequences.
        num_buckets : int
            Number of buckets

        Returns
        -------
        bucket_keys : list of int
            A list including the keys of the buckets.
        """
        if not isinstance(max_lengths, INT_TYPES):
            alpha_l = [2 * float(max_len - min_len - num_buckets)
                       / (num_buckets * (num_buckets + 1))
                       for max_len, min_len in
                       zip(max_lengths, min_lengths)]
            bucket_keys = \
                [tuple(int(round(min_len + alpha * (((i + 1) * (i + 2)) / 2) + i + 1))
                       for min_len, alpha in zip(min_lengths, alpha_l))
                 for i in range(num_buckets)]
            bucket_keys[-1] = tuple(max(max_bucket_key, max_len)
                                    for max_bucket_key, max_len
                                    in zip(bucket_keys[-1], max_lengths))
        else:
            alpha = 2 * float(max_lengths - min_lengths - num_buckets) \
                    / (num_buckets * (num_buckets + 1))
            bucket_keys = [int(round(min_lengths + alpha * (((i + 1) * (i + 2)) / 2) + i + 1))
                           for i in range(num_buckets)]
            bucket_keys[-1] = max(bucket_keys[-1], max_lengths)
        return bucket_keys


class ExpWidthBucket(BucketScheme):
    r""" Buckets with exponentially increasing width:
    :math:`w_i = bucket\_len\_step * w_{i-1}` for all :math:`i \geq 2`.

    Parameters
    ----------
    bucket_len_step : float, default 1.1
        This is the increasing factor for the bucket width.
    """
    def __init__(self, bucket_len_step=1.1):
        self.bucket_len_step = bucket_len_step

    def __call__(self, max_lengths, min_lengths, num_buckets):
        r"""This function generates bucket keys exponentially increasing bucket width.

        Parameters
        ----------
        max_lengths : int of list of int
            Maximum of lengths of sequences.
        min_lengths : int of list of int
            Minimum of lengths of sequences.
        num_buckets : int
            Number of buckets

        Returns
        -------
        bucket_keys : list of int
            A list including the keys of the buckets.
        """
        if not isinstance(max_lengths, INT_TYPES):
            initial_width_l = [
                (max_len - min_len) * (self.bucket_len_step - 1)
                / (math.pow(self.bucket_len_step, num_buckets) - 1)
                for max_len, min_len in
                zip(max_lengths, min_lengths)]
            bucket_keys = \
                [tuple(
                    int(round(min_len + initial_width * (math.pow(self.bucket_len_step, i + 1) - 1)
                              / (self.bucket_len_step - 1)))
                    for min_len, initial_width in zip(min_lengths, initial_width_l))
                 for i in range(num_buckets)]
            bucket_keys[-1] = tuple(max(max_bucket_key, max_len)
                                    for max_bucket_key, max_len
                                    in zip(bucket_keys[-1], max_lengths))
        else:
            initial_width = (max_lengths - min_lengths) * (self.bucket_len_step - 1) \
                            / (math.pow(self.bucket_len_step, num_buckets) - 1)
            bucket_keys = [
                int(round(min_lengths + initial_width * (math.pow(self.bucket_len_step, i + 1) - 1)
                          / (self.bucket_len_step - 1)))
                for i in range(num_buckets)]
            bucket_keys[-1] = max(bucket_keys[-1], max_lengths)
        return bucket_keys


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
    use_average_length : bool, default False
        False: each batch contains batch_size sequences, number of sequence elements varies.
        True: each batch contains batch_size elements, number of sequences varies. In this case,
        ratio option is ignored.
    num_shards : int, default 0
        If num_shards > 0, the sampled batch is split into num_shards smaller batches.
        The output will have structure of list(list(int)).
        If num_shards = 0, the output will have structure of list(int).
        This is useful in multi-gpu training and can potentially reduce the number of paddings.
        In general, it is set to the number of gpus.
    bucket_scheme : BucketScheme, default ConstWidthBucket
        It is used to generate bucket keys. It supports:
        ConstWidthBucket: all the buckets have the same width
        LinearWidthBucket: the width of ith  bucket follows :math:`w_i = \alpha * i + 1`
        ExpWidthBucket: the width of ith bucket follows :math:`w_i = bucket_len_step * w_{i-1}`
    Examples
    --------
    >>> lengths = [np.random.randint(1, 100) for _ in range(1000)]
    >>> sampler = gluonnlp.data.FixedBucketSampler(lengths, 8, ratio=0.5)
    >>> print(sampler.stats())
    FixedBucketSampler:
    -etc-
    """
    def __init__(self, lengths, batch_size, num_buckets=10, bucket_keys=None,
                 ratio=0, shuffle=False, use_average_length=False, num_shards=0,
                 bucket_scheme=ConstWidthBucket()):
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
        self._num_shards = num_shards
        self._bucket_scheme = bucket_scheme
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
            bucket_keys = bucket_scheme(max_lengths, min_lengths, num_buckets)
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
        if not use_average_length:
            scale_up_keys = [key if self._single_element else sum(key) for key
                             in self._bucket_keys]
            max_scale_up_key = max(scale_up_keys)
            self._bucket_batch_sizes = [max(int(max_scale_up_key / float(scale_up_key)
                                                * self._ratio * batch_size), batch_size)
                                        for scale_up_key in scale_up_keys]
        else:
            if ratio > 0.:
                warnings.warn('ratio=%f is ignored when use_average_length is True' % self._ratio)
            bucket_average_lengths, bucket_length_stds = _bucket_stats(self._bucket_sample_ids,
                                                                       self._lengths)
            self._bucket_batch_sizes = [max(int(batch_size / (average_length + length_std)), 1)
                                        for average_length, length_std
                                        in zip(bucket_average_lengths, bucket_length_stds)]
        self._batch_infos = []
        for bucket_id, sample_ids, bucket_batch_size in\
                zip(range(len(self._bucket_keys) - 1, -1, -1),
                        self._bucket_sample_ids[::-1],
                        self._bucket_batch_sizes[::-1]):
            for i in range(0, len(sample_ids), bucket_batch_size):
                self._batch_infos.append((bucket_id, i))

        if self._num_shards > 0:
            self._sampler_size = int(math.ceil(len(self._batch_infos) / float(self._num_shards)))
        else:
            self._sampler_size = len(self._batch_infos)

    def __iter__(self):
        if self._shuffle:
            np.random.shuffle(self._batch_infos)
            for bucket_id in range(len(self._bucket_keys)):
                np.random.shuffle(self._bucket_sample_ids[bucket_id])

        if self._num_shards > 0:
            for batch_idx in range(0, len(self._batch_infos), self._num_shards):
                if batch_idx + self._num_shards > len(self._batch_infos):
                    batch_idx = len(self._batch_infos) - self._num_shards
                batch = self._batch_infos[batch_idx: batch_idx + self._num_shards]
                bucket_ids, batch_begins = list(zip(*batch))
                batch_sizes = [self._bucket_batch_sizes[bucket_id] for bucket_id in bucket_ids]
                batch_ends = [min(batch_begin + batch_size,
                                  len(self._bucket_sample_ids[bucket_id]))
                              for bucket_id, batch_begin, batch_size in zip(bucket_ids,
                                                                            batch_begins,
                                                                            batch_sizes)]
                yield [self._bucket_sample_ids[bucket_id][batch_begin:batch_end]
                       for bucket_id, batch_begin, batch_end in zip(bucket_ids,
                                                                    batch_begins,
                                                                    batch_ends)]
        else:
            for bucket_id, batch_begin in self._batch_infos:
                batch_size = self._bucket_batch_sizes[bucket_id]
                batch_end = min(batch_begin + batch_size, len(self._bucket_sample_ids[bucket_id]))
                yield self._bucket_sample_ids[bucket_id][batch_begin:batch_end]

    def __len__(self):
        return self._sampler_size

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
    >>> lengths = [np.random.randint(1, 1000) for _ in range(1000)]
    >>> sampler = gluonnlp.data.SortedBucketSampler(lengths, 16)
    >>> # The sequence lengths within the batch will be sorted
    >>> for i, indices in enumerate(sampler):
    ...     if i == 0:
    ...         print([lengths[ele] for ele in indices])
    [-etc-]
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
