import numpy as np
import mxnet as mx
from gluonnlp.data import FixedBucketSampler, ShardedDataLoader
from mxnet import gluon
import pytest


def test_sharded_data_loader():
    X = np.random.uniform(size=(100, 20))
    Y = np.random.uniform(size=(100,))
    dataset = gluon.data.ArrayDataset(X, Y)
    loader = ShardedDataLoader(dataset, 2)
    for i, (x, y) in enumerate(loader):
        assert mx.test_utils.almost_equal(x.asnumpy(), X[i*2:(i+1)*2])
        assert mx.test_utils.almost_equal(y.asnumpy(), Y[i*2:(i+1)*2])
    num_shards = 4
    batch_sampler = FixedBucketSampler(lengths=[X.shape[1]] * X.shape[0],
                                       batch_size=2,
                                       num_buckets=1,
                                       shuffle=False,
                                       num_shards=num_shards)
    for num_workers in [0, 1, 2, 3, 4]:
        loader = ShardedDataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)
        for i, seqs in enumerate(loader):
            assert len(seqs) == num_shards
            for j in range(num_shards):
                if i != len(loader) - 1:
                    assert mx.test_utils.almost_equal(seqs[j][0].asnumpy(),
                                                      X[(i*num_shards+j)*2:(i*num_shards+j+1)*2])
                    assert mx.test_utils.almost_equal(seqs[j][1].asnumpy(),
                                                      Y[(i*num_shards+j)*2:(i*num_shards+j+1)*2])
                else:
                    assert mx.test_utils.almost_equal(seqs[j][0].asnumpy(),
                                                      X[(i*num_shards+j)*2-num_shards:
                                                        (i*num_shards+j+1)*2-num_shards])
                    assert mx.test_utils.almost_equal(seqs[j][1].asnumpy(),
                                                      Y[(i*num_shards+j)*2-num_shards:
                                                        (i*num_shards+j+1)*2-num_shards])
