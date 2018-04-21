from gluonnlp.data.sampler import SortedSampler, FixedBucketSampler, SortedBucketSampler
from mxnet.gluon.data import SimpleDataset
import numpy as np


def test_sorted_sampler():
    N = 1000
    dataset = SimpleDataset([np.random.normal(0, 1, (np.random.randint(10, 100), 1, 1))
                             for _ in range(N)])
    gt_sample_id = sorted(range(len(dataset)), key=lambda i: dataset[i].shape, reverse=True)
    sample_ret = list(SortedSampler([ele.shape[0] for ele in dataset]))
    for lhs, rhs in zip(gt_sample_id, sample_ret):
        assert lhs == rhs


def test_fixed_bucket_sampler():
    N = 1000
    for seq_lengths in [[np.random.randint(10, 100) for _ in range(N)],
                        [(np.random.randint(10, 100), np.random.randint(10, 100)) for _ in range(N)]]:
        for ratio in [0.0, 0.5]:
            for shuffle in [False, True]:
                for num_buckets in [1, 10, 100, 5000]:
                    sampler = FixedBucketSampler(seq_lengths, batch_size=8, num_buckets=num_buckets, ratio=ratio, shuffle=shuffle)
                    print(sampler.stats())
                    total_sampled_ids = []
                    for batch_sample_ids in sampler:
                        total_sampled_ids.extend(batch_sample_ids)
                    assert len(set(total_sampled_ids)) == len(total_sampled_ids) == N
    for seq_lengths in [[np.random.randint(10, 100) for _ in range(N)]]:
        for bucket_keys in [[1, 5, 10, 100], [10, 100], [200]]:
            sampler = FixedBucketSampler(seq_lengths, batch_size=8, num_buckets=None,
                                         bucket_keys=bucket_keys, ratio=ratio, shuffle=shuffle)
            print(sampler.stats())
            total_sampled_ids = []
            for batch_sample_ids in sampler:
                total_sampled_ids.extend(batch_sample_ids)
            assert len(set(total_sampled_ids)) == len(total_sampled_ids) == N
    for seq_lengths in [[(np.random.randint(10, 100), np.random.randint(10, 100)) for _ in range(N)]]:
        for bucket_keys in [[(1, 1), (5, 10), (10, 20), (20, 10), (100, 100)], [(20, 20), (30, 15), (100, 100)], [(100, 200)]]:
            sampler = FixedBucketSampler(seq_lengths, batch_size=8, num_buckets=None,
                                         bucket_keys=bucket_keys, ratio=ratio, shuffle=shuffle)
            print(sampler.stats())
            total_sampled_ids = []
            for batch_sample_ids in sampler:
                total_sampled_ids.extend(batch_sample_ids)
            assert len(set(total_sampled_ids)) == len(total_sampled_ids) == N


def test_sorted_bucket_sampler():
    N = 1000
    for seq_lengths in [[np.random.randint(10, 100) for _ in range(N)],
                        [(np.random.randint(10, 100), np.random.randint(10, 100)) for _ in
                         range(N)]]:
        for mult in [10, 100]:
            for batch_size in [5, 7]:
                for shuffle in [False, True]:
                    sampler = SortedBucketSampler(sort_keys=seq_lengths, batch_size=batch_size,
                                                  mult=mult, shuffle=shuffle)
                    total_sampled_ids = []
                    for batch_sample_ids in sampler:
                        total_sampled_ids.extend(batch_sample_ids)
                    assert len(set(total_sampled_ids)) == len(total_sampled_ids) == N
