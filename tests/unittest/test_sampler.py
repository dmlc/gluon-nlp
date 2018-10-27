import pytest
import numpy as np
from mxnet.gluon import data
import gluonnlp as nlp
from gluonnlp.data import sampler as s


N = 1000
def test_sorted_sampler():
    dataset = data.SimpleDataset([np.random.normal(0, 1, (np.random.randint(10, 100), 1, 1))
                                  for _ in range(N)])
    gt_sample_id = sorted(range(len(dataset)), key=lambda i: dataset[i].shape, reverse=True)
    sample_ret = list(s.SortedSampler([ele.shape[0] for ele in dataset]))
    for lhs, rhs in zip(gt_sample_id, sample_ret):
        assert lhs == rhs

@pytest.mark.parametrize('seq_lengths', [[np.random.randint(10, 100) for _ in range(N)],
                                         [(np.random.randint(10, 100), np.random.randint(10, 100))
                                           for _ in range(N)]])
@pytest.mark.parametrize('ratio', [0.0, 0.5])
@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('num_buckets', [1, 10, 100, 5000])
@pytest.mark.parametrize('bucket_scheme', [s.ConstWidthBucket(),
                                           s.LinearWidthBucket(),
                                           s.ExpWidthBucket()])
@pytest.mark.parametrize('use_average_length', [False, True])
@pytest.mark.parametrize('num_shards', range(4))
def test_fixed_bucket_sampler(seq_lengths, ratio, shuffle, num_buckets, bucket_scheme,
                              use_average_length, num_shards):
    sampler = s.FixedBucketSampler(seq_lengths,
                                   batch_size=8,
                                   num_buckets=num_buckets,
                                   ratio=ratio, shuffle=shuffle,
                                   use_average_length=use_average_length,
                                   bucket_scheme=bucket_scheme,
                                   num_shards=num_shards)
    print(sampler.stats())
    total_sampled_ids = []
    for batch_sample_ids in sampler:
        if num_shards > 0:
            assert len(batch_sample_ids) == num_shards
        else:
            total_sampled_ids.extend(batch_sample_ids)
    if num_shards == 0:
        assert len(set(total_sampled_ids)) == len(total_sampled_ids) == N

@pytest.mark.parametrize('bucket_keys', [[1, 5, 10, 100], [10, 100], [200]])
@pytest.mark.parametrize('ratio', [0.0, 0.5])
@pytest.mark.parametrize('shuffle', [False, True])
def test_fixed_bucket_sampler_with_single_key(bucket_keys, ratio, shuffle):
    seq_lengths = [np.random.randint(10, 100) for _ in range(N)]
    sampler = s.FixedBucketSampler(seq_lengths, batch_size=8, num_buckets=None,
                                   bucket_keys=bucket_keys, ratio=ratio, shuffle=shuffle)
    print(sampler.stats())
    total_sampled_ids = []
    for batch_sample_ids in sampler:
        total_sampled_ids.extend(batch_sample_ids)
    assert len(set(total_sampled_ids)) == len(total_sampled_ids) == N

@pytest.mark.parametrize('bucket_keys', [[(1, 1), (5, 10), (10, 20), (20, 10), (100, 100)],
                                         [(20, 20), (30, 15), (100, 100)],
                                         [(100, 200)]])
@pytest.mark.parametrize('ratio', [0.0, 0.5])
@pytest.mark.parametrize('shuffle', [False, True])
def test_fixed_bucket_sampler_with_single_key(bucket_keys, ratio, shuffle):
    seq_lengths = [(np.random.randint(10, 100), np.random.randint(10, 100)) for _ in range(N)]
    sampler = s.FixedBucketSampler(seq_lengths, batch_size=8, num_buckets=None,
                                   bucket_keys=bucket_keys, ratio=ratio, shuffle=shuffle)
    print(sampler.stats())
    total_sampled_ids = []
    for batch_sample_ids in sampler:
        total_sampled_ids.extend(batch_sample_ids)
    assert len(set(total_sampled_ids)) == len(total_sampled_ids) == N


def test_fixed_bucket_sampler_compactness():
    samples = list(
        s.FixedBucketSampler(
            np.arange(16, 32), 8, num_buckets=2,
            bucket_scheme=nlp.data.ConstWidthBucket()))
    assert len(samples) == 2


@pytest.mark.parametrize('seq_lengths', [[np.random.randint(10, 100) for _ in range(N)],
                                         [(np.random.randint(10, 100), np.random.randint(10, 100))
                                          for _ in range(N)]])
@pytest.mark.parametrize('mult', [10, 100])
@pytest.mark.parametrize('batch_size', [5, 7])
@pytest.mark.parametrize('shuffle', [False, True])
def test_sorted_bucket_sampler(seq_lengths, mult, batch_size, shuffle):
    sampler = s.SortedBucketSampler(sort_keys=seq_lengths,
                                    batch_size=batch_size,
                                    mult=mult, shuffle=shuffle)
    total_sampled_ids = []
    for batch_sample_ids in sampler:
        total_sampled_ids.extend(batch_sample_ids)
    assert len(set(total_sampled_ids)) == len(total_sampled_ids) == N
