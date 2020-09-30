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

"""Utilities for pre-training."""
import time
import os
import sys
import logging
import random

import numpy as np
import mxnet as mx
from mxnet.gluon.metric import EvalMetric

import gluonnlp.data.batchify as bf
from gluonnlp.utils.misc import glob
from gluonnlp.data.loading import NumpyDataset, DatasetLoader
from gluonnlp.data.sampler import SplitSampler, FixedBucketSampler

from create_pretraining_data import create_training_instances


def prepare_pretrain_npz_dataset(filename, allow_pickle=False):
    """Create dataset based on the numpy npz file"""
    if isinstance(filename, (list, tuple)):
        assert len(filename) == 1, \
            'When .npy/.npz data file is loaded, len(filename) must be 1.' \
            ' Received len(filename)={}.'.format(len(filename))
        filename = filename[0]
    logging.debug('start to load file %s ...', filename)
    return NumpyDataset(filename, allow_pickle=allow_pickle)


def prepare_pretrain_text_dataset(filename, tokenizer, max_seq_length, short_seq_prob,
                                  masked_lm_prob, max_predictions_per_seq, whole_word_mask,
                                  random_next_sentence, vocab):
    """Create dataset based on the raw text files"""
    dupe_factor = 1
    if not isinstance(filename, (list, tuple)):
        filename = [filename]
    logging.debug('start to load files %s ...', filename)
    instances = create_training_instances((filename, tokenizer, max_seq_length,
                                           short_seq_prob, masked_lm_prob,
                                           max_predictions_per_seq,
                                           whole_word_mask, vocab,
                                           dupe_factor, 1, None, None, random_next_sentence))
    return mx.gluon.data.ArrayDataset(*instances)


def prepare_pretrain_bucket_sampler(dataset, batch_size, shuffle=False, num_buckets=1):
    """Create data sampler based on the dataset"""
    if isinstance(dataset, NumpyDataset):
        lengths = dataset.get_field('valid_lengths')
    else:
        lengths = dataset.transform(lambda input_ids, segment_ids, masked_lm_positions, \
                                           masked_lm_ids, masked_lm_weights, \
                                           next_sentence_labels, valid_lengths: \
                                        valid_lengths, lazy=False)
    sampler = FixedBucketSampler(lengths,
                                 batch_size=batch_size,
                                 num_buckets=num_buckets,
                                 ratio=0,
                                 shuffle=shuffle)
    logging.debug('Sampler created for a new dataset:\n%s', sampler)
    return sampler


def get_pretrain_data_text(data, batch_size, shuffle, num_buckets, vocab, tokenizer,
                           max_seq_length, short_seq_prob, masked_lm_prob,
                           max_predictions_per_seq, whole_word_mask, random_next_sentence,
                           num_parts=1, part_idx=0, num_dataset_workers=1, num_batch_workers=1,
                           circle_length=1, repeat=1,
                           dataset_cached=False, num_max_dataset_cached=0):
    """Get a data iterator from raw text documents.

    Parameters
    ----------
    batch_size : int
        The batch size per GPU.
    shuffle : bool
        Whether to shuffle the data.
    num_buckets : int
        The number of buckets for the FixedBucketSampler for training.
    vocab : Vocab
        The vocabulary.
    tokenizer : BaseTokenizer
        The tokenizer.
    max_seq_length : int
        The hard limit of maximum sequence length of sentence pairs.
    short_seq_prob : float
        The probability of sampling sequences shorter than the max_seq_length.
    masked_lm_prob : float
        The probability of replacing texts with masks/random words/original words.
    max_predictions_per_seq : int
        The hard limit of the number of predictions for masked words
    whole_word_mask : bool
        Whether to use whole word masking.
    num_parts : int
        The number of partitions for the dataset.
    part_idx : int
        The index of the partition to read.
    num_dataset_workers : int
        The number of worker processes for dataset construction.
    num_batch_workers : int
        The number of worker processes for batch construction.
    circle_length : int, default is 1
        The number of files to be read for a single worker at the same time.
        When circle_length is larger than 1, we merge circle_length files.
    repeat : int, default is 1
        The number of times that files are repeated.
    dataset_cached : bool, default is False
        Whether or not to cache last processed dataset.
        Each processed dataset can only be cached for once.
        When there is no new available processed dataset to be fetched,
        we pop a cached processed dataset.
    num_max_dataset_cached : int, default is 0
        Maximum number of cached datasets. It is valid only if dataset_cached is True
    """
    num_files = len(glob(data))
    logging.info('%d files are found.', num_files)
    assert num_files >= num_parts, \
        'The number of text files must be no less than the number of ' \
        'workers/partitions (%d). Only %d files at %s are found.'%(num_parts, num_files, data)
    dataset_params = {'tokenizer': tokenizer, 'max_seq_length': max_seq_length,
                      'short_seq_prob': short_seq_prob, 'masked_lm_prob': masked_lm_prob,
                      'max_predictions_per_seq': max_predictions_per_seq, 'vocab':vocab,
                      'whole_word_mask': whole_word_mask, 'random_next_sentence': random_next_sentence}
    sampler_params = {'batch_size': batch_size, 'shuffle': shuffle, 'num_buckets': num_buckets}
    dataset_fn = prepare_pretrain_text_dataset
    sampler_fn = prepare_pretrain_bucket_sampler
    pad_val = vocab.pad_id
    batchify_fn = bf.Tuple(
        bf.Pad(val=pad_val, round_to=8),  # input_id
        bf.Pad(val=pad_val),  # masked_id
        bf.Pad(val=0),  # masked_position
        bf.Pad(val=0),  # masked_weight
        bf.Stack(),  # next_sentence_label
        bf.Pad(val=0, round_to=8),  # segment_id
        bf.Stack())  # valid_lengths
    split_sampler = SplitSampler(num_files, num_parts=num_parts,
                                 part_index=part_idx, repeat=repeat)
    dataloader = DatasetLoader(data,
                               file_sampler=split_sampler,
                               dataset_fn=dataset_fn,
                               batch_sampler_fn=sampler_fn,
                               dataset_params=dataset_params,
                               batch_sampler_params=sampler_params,
                               batchify_fn=batchify_fn,
                               num_dataset_workers=num_dataset_workers,
                               num_batch_workers=num_batch_workers,
                               pin_memory=False,
                               circle_length=circle_length,
                               dataset_cached=dataset_cached,
                               num_max_dataset_cached=num_max_dataset_cached)
    return dataloader


def get_pretrain_data_npz(data, batch_size,
                          shuffle, num_buckets,
                          vocab, num_parts=1, part_idx=0,
                          num_dataset_workers=1, num_batch_workers=1,
                          circle_length=1, repeat=1,
                          dataset_cached=False, num_max_dataset_cached=0):
    """Get a data iterator from pre-processed npz files.

    Parameters
    ----------
    batch_size : int
        The batch size per GPU.
    shuffle : bool
        Whether to shuffle the data.
    num_buckets : int
        The number of buckets for the FixedBucketSampler for training.
    vocab : Vocab
        The vocabulary.
    num_parts : int
        The number of partitions for the dataset.
    part_idx : int
        The index of the partition to read.
    num_dataset_workers : int
        The number of worker processes for dataset construction.
    num_batch_workers : int
        The number of worker processes for batch contruction.
    circle_length : int, default is 1
        The number of files to be read for a single worker at the same time.
        When circle_length is larger than 1, we merge circle_length files.
    repeat : int, default is 1
        The number of times that files are repeated.
    dataset_cached : bool, default is False
        Whether or not to cache last processed dataset.
        Each processed dataset can only be cached for once.
        When there is no new available processed dataset to be fetched,
        we pop a cached processed dataset.
    num_max_dataset_cached : int, default is 0
        Maximum number of cached datasets. It is valid only if dataset_cached is True
    """
    num_files = len(glob(data))
    logging.info('%d files are found.', num_files)
    assert num_files >= num_parts, \
        'The number of text files must be no less than the number of ' \
        'workers/partitions (%d). Only %d files at %s are found.'%(num_parts, num_files, data)
    dataset_params = {'allow_pickle': True}
    sampler_params = {'batch_size': batch_size, 'shuffle': shuffle, 'num_buckets': num_buckets}
    dataset_fn = prepare_pretrain_npz_dataset
    sampler_fn = prepare_pretrain_bucket_sampler
    pad_val = vocab.pad_id
    batchify_fn = bf.Tuple(
        bf.Pad(val=pad_val, round_to=8),  # input_id
        bf.Pad(val=pad_val),  # masked_id
        bf.Pad(val=0),  # masked_position
        bf.Pad(val=0),  # masked_weight
        bf.Stack(),  # next_sentence_label
        bf.Pad(val=0, round_to=8),  # segment_id
        bf.Stack())  # valid_lengths
    split_sampler = SplitSampler(num_files, num_parts=num_parts,
                                 part_index=part_idx, repeat=repeat)
    dataloader = DatasetLoader(data,
                               file_sampler=split_sampler,
                               dataset_fn=dataset_fn,
                               batch_sampler_fn=sampler_fn,
                               dataset_params=dataset_params,
                               batch_sampler_params=sampler_params,
                               batchify_fn=batchify_fn,
                               num_dataset_workers=num_dataset_workers,
                               num_batch_workers=num_batch_workers,
                               pin_memory=False,
                               circle_length=circle_length,
                               dataset_cached=dataset_cached,
                               num_max_dataset_cached=num_max_dataset_cached)
    return dataloader



class MaskedAccuracy(EvalMetric):
    def __init__(self, axis=1, name='masked_accuracy',
                 output_names=None, label_names=None):
        super(MaskedAccuracy, self).__init__(name, axis=axis,
            output_names=output_names, label_names=label_names)
        self.axis = axis

    def update(self, labels, preds, masks=None):
        masks = [None] * len(labels) if masks is None else masks
        for label, pred_label, mask in zip(labels, preds, masks):
            if pred_label.shape != label.shape:
                #pred_label = pred_label.argmax(axis=self.axis)
                pred_label = mx.npx.topk(pred_label.astype('float32', copy=False),
                    k=1, ret_typ='indices', axis=self.axis, dtype=np.int32)
            pred_label = pred_label.astype('int32').reshape((-1,))
            label = label.astype('int32').reshape((-1,))
            if mask is not None:
                mask = mask.astype('int32').reshape((-1,))
                num_correct = ((pred_label == label) * mask).sum().asnumpy().item()
                self.sum_metric += num_correct
                self.num_inst += mask.sum().asnumpy().item()
            else:
                num_correct = (pred_label == label).sum().asnumpy().item()
                self.sum_metric += num_correct
                self.num_inst += len(pred_label)


