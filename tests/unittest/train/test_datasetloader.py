import numpy as np
import os
import mxnet as mx
import gluonnlp as nlp


def test_dataset_loader():
    num_files = 3
    for i in range(num_files):
        np.save(os.path.join('tests', 'data', 'part_{}'.format(i)), np.random.uniform(size=(100, 20)))
    data = os.path.join('tests', 'data')
    split_sampler = nlp.data.SplitSampler(num_files, num_parts=1, part_index=0)

    def prepare_dataset(filename, allow_pickle=False):
        """Create dataset based on the files"""
        return nlp.data.NumpyDataset(filename, allow_pickle=allow_pickle)

    def prepare_bucket_sampler(dataset, batch_size, shuffle=False, num_buckets=1):
        sampler = nlp.data.FixedBucketSampler([dataset.shape[1]] * dataset.shape[0],
                                              batch_size=batch_size,
                                              num_buckets=num_buckets,
                                              ratio=0,
                                              shuffle=shuffle)
        return sampler

    dataset_params = {'allow_pickle': True}
    sampler_params = {'batch_size': 2}
    X = np.load(os.path.join(data, 'part_{}'.format(0)))
    for i in range(1, num_files):
        X = np.concatenate((X, np.load(os.path.join(data, 'part_{}'.format(i)))))
    for num_dataset_workers in [0, 1, 2]:
        for num_batch_workers in [0, 1, 2]:
            dataloader = nlp.data.DatasetLoader(data, file_sampler=split_sampler,
                                                dataset_fn=prepare_dataset,
                                                dataset_params=dataset_params,
                                                batch_sampler_fn=prepare_bucket_sampler,
                                                batch_sampler_params=sampler_params,
                                                num_dataset_workers=num_dataset_workers,
                                                num_batch_workers=num_batch_workers,
                                                pin_memory=True, circle_length=1)
            for i, x in enumerate(dataloader):
                assert mx.test_utils.almost_equal(x.asnumpy(), X[i * 2:(i + 1) * 2])
