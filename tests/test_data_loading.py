import os
import sys
import tempfile
import pytest

import mxnet as mx
import numpy as np
from numpy.testing import assert_almost_equal

from gluonnlp.data.loading import NumpyDataset, DatasetLoader
from gluonnlp.data.sampler import SplitSampler, FixedBucketSampler

mx.npx.set_np()


def prepare_dataset(filename, allow_pickle=False):
    return NumpyDataset(filename[0], allow_pickle=allow_pickle)


def prepare_bucket_sampler(dataset, batch_size, shuffle=False, num_buckets=1):
    lengths = dataset.transform(lambda x: len(x))
    sampler = FixedBucketSampler(lengths,
                                 batch_size=batch_size,
                                 num_buckets=num_buckets,
                                 ratio=0,
                                 shuffle=shuffle)
    return sampler


@pytest.mark.skipif(sys.version_info >= (3, 8),
                    reason='The test fails everytime in python3.8 due to the issues'
                           ' in MXNet: '
                           'https://github.com/apache/incubator-mxnet/issues/17782, '
                           'https://github.com/apache/incubator-mxnet/issues/17774')
def test_dataset_loader():
    with tempfile.TemporaryDirectory() as root:
        num_files = 5
        for i in range(num_files):
            np.save(os.path.join(root, 'part_{}.npy'.format(i)),
                    np.random.uniform(size=(100, 20)))
        data = os.path.join(root)
        split_sampler = SplitSampler(num_files, num_parts=1, part_index=0, shuffle=False)

        dataset_params = {'allow_pickle': True}
        sampler_params = {'batch_size': 2}
        all_data = np.load(os.path.join(data, 'part_{}.npy'.format(0)))
        for i in range(1, num_files):
            all_data = np.concatenate((all_data,
                                       np.load(os.path.join(data, 'part_{}.npy'.format(i)))))
        for num_dataset_workers in [1, 2]:
            for num_batch_workers in [1, 2]:
                dataloader = DatasetLoader(os.path.join(data, '*.npy'),
                                           file_sampler=split_sampler,
                                           dataset_fn=prepare_dataset,
                                           dataset_params=dataset_params,
                                           batch_sampler_fn=prepare_bucket_sampler,
                                           batch_sampler_params=sampler_params,
                                           num_dataset_workers=num_dataset_workers,
                                           num_batch_workers=num_batch_workers,
                                           pin_memory=True,
                                           circle_length=1)
                for i, x in enumerate(dataloader):
                    assert_almost_equal(x.asnumpy(), all_data[i * 2:(i + 1) * 2])

        # test cache
        split_sampler = SplitSampler(1, num_parts=1, part_index=0,
                                     repeat=2, shuffle=False)
        X = np.load(os.path.join(data, 'part_{}.npy'.format(0)))
        X = np.concatenate((X, X))
        for num_dataset_workers in [1]:
            for num_batch_workers in [1]:
                dataloader = DatasetLoader(os.path.join(data, 'part_0.npy'),
                                           file_sampler=split_sampler,
                                           dataset_fn=prepare_dataset,
                                           dataset_params=dataset_params,
                                           batch_sampler_fn=prepare_bucket_sampler,
                                           batch_sampler_params=sampler_params,
                                           num_dataset_workers=num_dataset_workers,
                                           num_batch_workers=num_batch_workers,
                                           pin_memory=True,
                                           circle_length=1,
                                           dataset_cached=True,
                                           num_max_dataset_cached=1)
                for i, x in enumerate(dataloader):
                    assert_almost_equal(x.asnumpy(), X[i * 2:(i + 1) * 2])
