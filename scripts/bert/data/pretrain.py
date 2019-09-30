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
"""Dataset for pre-training. """
import logging

import gluonnlp as nlp
from gluonnlp.data.batchify import Tuple, Stack, Pad
try:
    from .dataloader import SamplerFn, DataLoaderFn
except ImportError:
    from dataloader import SamplerFn, DataLoaderFn

__all__ = ['BERTSamplerFn', 'BERTDataLoaderFn']

class BERTSamplerFn(SamplerFn):
    """Callable object to create the sampler"""
    def __init__(self, batch_size, shuffle, num_ctxes, num_buckets):
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_ctxes = num_ctxes
        self._num_buckets = num_buckets

    def __call__(self, dataset):
        """Create data sampler based on the dataset"""
        if isinstance(dataset, nlp.data.NumpyDataset):
            lengths = dataset.get_field('valid_lengths')
        else:
            # dataset is a BERTPretrainDataset:
            lengths = dataset.transform(lambda input_ids, segment_ids, masked_lm_positions, \
                                               masked_lm_ids, masked_lm_weights, \
                                               next_sentence_labels, valid_lengths: \
                                               valid_lengths, lazy=False)
        # calculate total batch size for all GPUs
        batch_size = self._batch_size * self._num_ctxes
        sampler = nlp.data.FixedBucketSampler(lengths,
                                              batch_size=batch_size,
                                              num_buckets=self._num_buckets,
                                              ratio=0,
                                              shuffle=self._shuffle)
        logging.debug('Sampler created for a new dataset:\n%s', sampler.stats())
        return sampler

class BERTDataLoaderFn(DataLoaderFn):
    """Callable object to create the data loader"""
    def __init__(self, num_ctxes, vocab):
        self._num_ctxes = num_ctxes
        pad_val = vocab[vocab.padding_token]
        self._batchify_fn = Tuple(Pad(pad_val=pad_val, round_to=8), # input_id
                                  Pad(pad_val=pad_val),             # masked_id
                                  Pad(pad_val=0),                   # masked_position
                                  Pad(pad_val=0),                   # masked_weight
                                  Stack(),                          # next_sentence_label
                                  Pad(pad_val=0, round_to=8),       # segment_id
                                  Stack())                          # valid_length

    def __call__(self, dataset, sampler):
        from mxnet.gluon.data import DataLoader
        dataloader = DataLoader(dataset=dataset,
                                batch_sampler=sampler,
                                batchify_fn=self._batchify_fn,
                                num_workers=self._num_ctxes)
        return dataloader
