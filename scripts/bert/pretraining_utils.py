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
# pylint:disable=redefined-outer-name,logging-format-interpolation
import glob
import time
import os
import logging
import mxnet as mx
from mxnet.gluon.data import DataLoader

import gluonnlp as nlp
from gluonnlp.data.batchify import Tuple, Stack, Pad

__all__ = ['get_model', 'get_pretrain_dataset', 'get_dummy_dataloader']

def get_model(ctx, model, pretrained, dataset_name, dtype, ckpt_dir=None, start_step=None):
    """Get model for pre-training."""
    # model
    model, vocabulary = nlp.model.get_model(model,
                                            dataset_name=dataset_name,
                                            pretrained=pretrained, ctx=ctx)

    if not pretrained:
        model.initialize(init=mx.init.Normal(0.02), ctx=ctx)

    if ckpt_dir and start_step:
        param_path = os.path.join(ckpt_dir, '%07d.params'%start_step)
        model.load_parameters(param_path, ctx=ctx)
        logging.info('Loading step %d checkpoints from %s.', start_step, param_path)

    model.cast(dtype)
    model.hybridize(static_alloc=True)

    # losses
    nsp_loss = mx.gluon.loss.SoftmaxCELoss()
    mlm_loss = mx.gluon.loss.SoftmaxCELoss()
    nsp_loss.hybridize(static_alloc=True)
    mlm_loss.hybridize(static_alloc=True)

    return model, nsp_loss, mlm_loss, vocabulary

def get_pretrain_dataset(data, batch_size, num_ctxes, shuffle, use_avg_len,
                         num_buckets, num_parts=1, part_idx=0, prefetch=True):
    """create dataset for pretraining."""
    num_files = len(glob.glob(os.path.expanduser(data)))
    assert num_files >= num_parts, \
        'Number of training files must be greater than the number of partitions'
    split_sampler = nlp.data.SplitSampler(num_files, num_parts=num_parts, part_index=part_idx)
    stream = nlp.data.SimpleDatasetStream(nlp.data.NumpyDataset, data, split_sampler)
    if prefetch:
        stream = nlp.data.PrefetchingStream(stream)

    def get_dataloader(dataset):
        """create data loader based on the dataset chunk"""
        t0 = time.time()
        lengths = dataset.get_field('valid_lengths')
        logging.debug('Num samples = %d', len(lengths))
        # A batch includes: input_id, masked_id, masked_position, masked_weight,
        #                   next_sentence_label, segment_id, valid_length
        batchify_fn = Tuple(Pad(), Pad(), Pad(), Pad(), Stack(), Pad(), Stack())
        if use_avg_len:
            # sharded data loader
            sampler = nlp.data.FixedBucketSampler(lengths=lengths,
                                                  # batch_size per shard
                                                  batch_size=batch_size,
                                                  num_buckets=num_buckets,
                                                  shuffle=shuffle,
                                                  use_average_length=True,
                                                  num_shards=num_ctxes)
            dataloader = nlp.data.ShardedDataLoader(dataset,
                                                    batch_sampler=sampler,
                                                    batchify_fn=batchify_fn,
                                                    num_parts=num_ctxes)
        else:
            sampler = nlp.data.FixedBucketSampler(lengths,
                                                  batch_size=batch_size * num_ctxes,
                                                  num_buckets=num_buckets,
                                                  ratio=0,
                                                  shuffle=shuffle)
            dataloader = DataLoader(dataset=dataset,
                                    batch_sampler=sampler,
                                    batchify_fn=batchify_fn,
                                    num_workers=1)
        logging.debug('Batch Sampler:\n%s', sampler.stats())
        t1 = time.time()
        logging.debug('Dataloader creation cost = %.2f s', t1 - t0)
        return dataloader

    stream = stream.transform(get_dataloader)
    return stream

def get_dummy_dataloader(dataloader, target_shape):
    """Return a dummy data loader which returns a fixed data batch of target shape"""
    data_iter = enumerate(dataloader)
    _, data_batch = next(data_iter)
    logging.debug('Searching target batch shape: %s', target_shape)
    while data_batch[0].shape != target_shape:
        logging.debug('Skip batch with shape %s', data_batch[0].shape)
        _, data_batch = next(data_iter)
    logging.debug('Found target dummy batch.')

    class DummyIter():
        def __init__(self, batch):
            self._batch = batch

        def __iter__(self):
            while True:
                yield self._batch

    return DummyIter(data_batch)
