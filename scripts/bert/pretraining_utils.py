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

"""Utilities for pre-training."""
import glob
import time
import os
import functools
import logging
import argparse
import random
import multiprocessing

import numpy as np

import mxnet as mx
from mxnet.gluon.data import DataLoader
from create_pretraining_data import create_training_instances

import gluonnlp as nlp
from gluonnlp.data.batchify import Tuple, Stack, Pad
from gluonnlp.metric import MaskedAccuracy

__all__ = ['get_model_loss', 'get_pretrain_data_npz', 'get_dummy_dataloader',
           'save_parameters', 'save_states', 'evaluate', 'forward', 'split_and_load',
           'get_argparser', 'get_pretrain_data_text', 'generate_dev_set']

def get_model_loss(ctx, model, pretrained, dataset_name, vocab, dtype,
                   ckpt_dir=None, start_step=None):
    """Get model for pre-training.

    Parameters
    ----------
    ctx : Context or list of Context
        Contexts to initialize model
    model : str
        The name of the model, 'bert_12_768_12' or 'bert_24_1024_16'.
    pretrained : bool
        Whether to use pre-trained model weights as initialization.
    dataset_name : str
        The name of the dataset, which is used to retrieve the corresponding vocabulary file
        when the vocab argument is not provided. Options include 'book_corpus_wiki_en_uncased',
        'book_corpus_wiki_en_cased', 'wiki_multilingual_uncased', 'wiki_multilingual_cased',
        'wiki_cn_cased'.
    vocab : BERTVocab or None
        The vocabulary for the model. If not provided, The vocabulary will be constructed
        based on dataset_name.
    dtype : float
        Data type of the model for training.
    ckpt_dir : str
        The path to the checkpoint directory.
    start_step : int or None
        If provided, it loads the model from the corresponding checkpoint from the ckpt_dir.

    Returns
    -------
    BERTModel : the model for pre-training.
    Loss : the next sentence prediction loss.
    Loss : the masked langauge model loss.
    BERTVocab : the vocabulary.
    """
    # model
    model, vocabulary = nlp.model.get_model(model, dataset_name=dataset_name, vocab=vocab,
                                            pretrained=pretrained, ctx=ctx)

    if not pretrained:
        model.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    model.cast(dtype)

    if ckpt_dir and start_step:
        param_path = os.path.join(ckpt_dir, '%07d.params'%start_step)
        nlp.utils.load_parameters(model, param_path, ctx=ctx)
        logging.info('Loading step %d checkpoints from %s.', start_step, param_path)

    model.hybridize(static_alloc=True)

    # losses
    nsp_loss = mx.gluon.loss.SoftmaxCELoss()
    mlm_loss = mx.gluon.loss.SoftmaxCELoss()
    nsp_loss.hybridize(static_alloc=True)
    mlm_loss.hybridize(static_alloc=True)

    return model, nsp_loss, mlm_loss, vocabulary

class BERTPretrainDataset(mx.gluon.data.ArrayDataset):
    """Dataset for BERT pre-training.

    Each record contains the following numpy ndarrays: input_ids, masked_lm_ids,
    masked_lm_positions, masked_lm_weights, next_sentence_labels, segment_ids, valid_lengths.

    Parameters
    ----------
    filename : str
        Path to the input text file.
    tokenizer : BERTTokenizer
        The BERTTokenizer
    max_seq_length : int
        The hard limit of maximum sequence length of sentence pairs
    short_seq_prob : float
        The probability of sampling sequences shorter than the max_seq_length.
    masked_lm_prob : float
        The probability of replacing texts with masks/random words/original words.
    max_predictions_per_seq : int
        The hard limit of the number of predictions for masked words
    whole_word_mask : bool
        Whether to use whole word masking.
    vocab : BERTVocab
        The BERTVocab
    num_workers : int
        The number of worker processes for dataset contruction.
    worker_pool : multiprocessing.Pool
        The worker process pool. Must be provided if num_workers > 1.
    """
    def __init__(self, filename, tokenizer, max_seq_length, short_seq_prob,
                 masked_lm_prob, max_predictions_per_seq, whole_word_mask,
                 vocab, num_workers=1, worker_pool=None):
        logging.debug('start to load file %s ...', filename)
        dupe_factor = 1
        instances = create_training_instances(([filename], tokenizer, max_seq_length,
                                               short_seq_prob, masked_lm_prob,
                                               max_predictions_per_seq,
                                               whole_word_mask, vocab,
                                               dupe_factor, num_workers,
                                               worker_pool, None))
        super(BERTPretrainDataset, self).__init__(*instances)

def get_pretrain_data_text(data, batch_size, num_ctxes, shuffle, use_avg_len,
                           num_buckets, vocab, tokenizer, max_seq_length, short_seq_prob,
                           masked_lm_prob, max_predictions_per_seq, whole_word_mask,
                           num_parts=1, part_idx=0,
                           prefetch=True, num_workers=1):
    """Get data iterators from raw text documents.

    Parameters
    ----------
    batch_size : int
        The batch size. If use_avg_len is set to True, batch_size is roughly the number of
        (non-padded) tokens in a batch.
    num_buckets : int
        The number of buckets for the FixedBucketSampler for training.
    vocab : BERTVocab
        The vocabulary.
    tokenizer : BERTTokenizer or BERTSPTokenizer
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
    prefetch : bool
        If set to True, a separate thread helps prefetching the next mini-batch of data.
    num_workers : int
        The number of worker processes for dataset contruction.
    """
    # handle commas in the provided path
    num_files = sum([len(glob.glob(os.path.expanduser(d.strip()))) for d in data.split(',')])
    logging.info('%d files found.', num_files)
    assert num_files >= num_parts, \
        'Number of training files must be greater than the number of partitions. ' \
        'Only found %d files at %s'%(num_files, data)
    worker_pool = multiprocessing.Pool(num_workers)
    dataset_cls = functools.partial(BERTPretrainDataset, tokenizer=tokenizer,
                                    max_seq_length=max_seq_length,
                                    short_seq_prob=short_seq_prob,
                                    masked_lm_prob=masked_lm_prob,
                                    max_predictions_per_seq=max_predictions_per_seq,
                                    whole_word_mask=whole_word_mask,
                                    vocab=vocab, num_workers=num_workers, worker_pool=worker_pool)

    split_sampler = nlp.data.SplitSampler(num_files, num_parts=num_parts, part_index=part_idx)
    stream = nlp.data.SimpleDatasetStream(dataset_cls, data, split_sampler)
    if prefetch:
        stream = nlp.data.PrefetchingStream(stream)
    # create data loader based on the dataset
    dataloader_xform = BERTLoaderTransform(use_avg_len, batch_size,
                                           shuffle, num_ctxes, num_buckets)
    stream = stream.transform(dataloader_xform)
    return stream

class BERTLoaderTransform(object):
    """Create dataloader for a BERT dataset. """

    def __init__(self, use_avg_len, batch_size, shuffle, num_ctxes, num_buckets):
        self._use_avg_len = use_avg_len
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_ctxes = num_ctxes
        self._num_buckets = num_buckets

    def __call__(self, dataset):
        """create data loader based on the dataset chunk"""
        if isinstance(dataset, nlp.data.NumpyDataset):
            lengths = dataset.get_field('valid_lengths')
        elif isinstance(dataset, BERTPretrainDataset):
            lengths = dataset.transform(lambda input_ids, segment_ids, masked_lm_positions, \
                                               masked_lm_ids, masked_lm_weights, \
                                               next_sentence_labels, valid_lengths: \
                                               valid_lengths, lazy=False)
        else:
            raise ValueError('unexpected dataset type: %s'%str(dataset))

        # A batch includes: input_id, masked_id, masked_position, masked_weight,
        #                   next_sentence_label, segment_id, valid_length
        batchify_fn = Tuple(Pad(), Pad(), Pad(), Pad(), Stack(), Pad(), Stack())
        if self._use_avg_len:
            # sharded data loader
            sampler = nlp.data.FixedBucketSampler(lengths=lengths,
                                                  # batch_size per shard
                                                  batch_size=self._batch_size,
                                                  num_buckets=self._num_buckets,
                                                  shuffle=self._shuffle,
                                                  use_average_length=True,
                                                  num_shards=self._num_ctxes)
            dataloader = nlp.data.ShardedDataLoader(dataset,
                                                    batch_sampler=sampler,
                                                    batchify_fn=batchify_fn,
                                                    num_workers=self._num_ctxes)
        else:
            sampler = nlp.data.FixedBucketSampler(lengths,
                                                  batch_size=self._batch_size * self._num_ctxes,
                                                  num_buckets=self._num_buckets,
                                                  ratio=0,
                                                  shuffle=self._shuffle)
            dataloader = DataLoader(dataset=dataset,
                                    batch_sampler=sampler,
                                    batchify_fn=batchify_fn,
                                    num_workers=1)
        logging.debug('Sampler created for a new dataset:\n%s', sampler.stats())
        return dataloader

def get_pretrain_data_npz(data, batch_size, num_ctxes, shuffle, use_avg_len,
                          num_buckets, num_parts=1, part_idx=0, prefetch=True):
    """create dataset for pretraining based on pre-processed npz files."""
    # handle commas in the provided path
    num_files = sum([len(glob.glob(os.path.expanduser(d.strip()))) for d in data.split(',')])
    logging.info('%d files found.', num_files)
    assert num_files >= num_parts, \
        'Number of training files must be greater than the number of partitions. ' \
        'Only found %d files at %s'%(num_files, data)
    split_sampler = nlp.data.SplitSampler(num_files, num_parts=num_parts, part_index=part_idx)
    stream = nlp.data.SimpleDatasetStream(nlp.data.NumpyDataset, data, split_sampler)
    if prefetch:
        stream = nlp.data.PrefetchingStream(stream)

    # create data loader based on the dataset
    dataloader_xform = BERTLoaderTransform(use_avg_len, batch_size,
                                           shuffle, num_ctxes, num_buckets)
    stream = stream.transform(dataloader_xform)
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

def save_parameters(step_num, model, ckpt_dir):
    """Save the model parameter, marked by step_num."""
    param_path = os.path.join(ckpt_dir, '%07d.params'%step_num)
    logging.info('[step %d] Saving model params to %s.', step_num, param_path)
    nlp.utils.save_parameters(model, param_path)

def save_states(step_num, trainer, ckpt_dir, local_rank=0):
    """Save the trainer states, marked by step_num."""
    trainer_path = os.path.join(ckpt_dir, '%07d.states.%02d'%(step_num, local_rank))
    logging.info('[step %d] Saving trainer states to %s.', step_num, trainer_path)
    nlp.utils.save_states(trainer, trainer_path)

def log(begin_time, running_num_tks, running_mlm_loss, running_nsp_loss, step_num,
        mlm_metric, nsp_metric, trainer, log_interval):
    """Log training progress."""
    end_time = time.time()
    duration = end_time - begin_time
    throughput = running_num_tks / duration / 1000.0
    running_mlm_loss = running_mlm_loss / log_interval
    running_nsp_loss = running_nsp_loss / log_interval
    lr = trainer.learning_rate if trainer else 0
    # pylint: disable=line-too-long
    logging.info('[step {}]\tmlm_loss={:.5f}\tmlm_acc={:.5f}\tnsp_loss={:.5f}\tnsp_acc={:.3f}\tthroughput={:.1f}K tks/s\tlr={:.7f} time={:.2f}, latency={:.1f} ms/batch'
                 .format(step_num, running_mlm_loss.asscalar(), mlm_metric.get()[1] * 100, running_nsp_loss.asscalar(),
                         nsp_metric.get()[1] * 100, throughput.asscalar(), lr, duration, duration*1000/log_interval))
    # pylint: enable=line-too-long

def split_and_load(arrs, ctx):
    """split and load arrays to a list of contexts"""
    assert isinstance(arrs, (list, tuple))
    # split and load
    loaded_arrs = [mx.gluon.utils.split_and_load(arr, ctx, even_split=False) for arr in arrs]
    return zip(*loaded_arrs)


def forward(data, model, mlm_loss, nsp_loss, vocab_size, dtype):
    """forward computation for evaluation"""
    (input_id, masked_id, masked_position, masked_weight, \
     next_sentence_label, segment_id, valid_length) = data
    num_masks = masked_weight.sum() + 1e-8
    valid_length = valid_length.reshape(-1)
    masked_id = masked_id.reshape(-1)
    valid_length_typed = valid_length.astype(dtype, copy=False)
    _, _, classified, decoded = model(input_id, segment_id, valid_length_typed,
                                      masked_position)
    decoded = decoded.reshape((-1, vocab_size))
    ls1 = mlm_loss(decoded.astype('float32', copy=False),
                   masked_id, masked_weight.reshape((-1, 1)))
    ls2 = nsp_loss(classified.astype('float32', copy=False), next_sentence_label)
    ls1 = ls1.sum() / num_masks
    ls2 = ls2.mean()
    ls = ls1 + ls2
    return ls, next_sentence_label, classified, masked_id, decoded, \
           masked_weight, ls1, ls2, valid_length.astype('float32', copy=False)


def evaluate(data_eval, model, nsp_loss, mlm_loss, vocab_size, ctx, log_interval, dtype):
    """Evaluation function."""
    logging.info('Running evaluation ... ')
    mlm_metric = MaskedAccuracy()
    nsp_metric = MaskedAccuracy()
    mlm_metric.reset()
    nsp_metric.reset()

    eval_begin_time = time.time()
    begin_time = time.time()
    step_num = 0
    running_mlm_loss = running_nsp_loss = 0
    total_mlm_loss = total_nsp_loss = 0
    running_num_tks = 0
    for _, dataloader in enumerate(data_eval):
        for _, data_batch in enumerate(dataloader):
            step_num += 1

            data_list = split_and_load(data_batch, ctx)
            loss_list = []
            ns_label_list, ns_pred_list = [], []
            mask_label_list, mask_pred_list, mask_weight_list = [], [], []
            for data in data_list:
                out = forward(data, model, mlm_loss, nsp_loss, vocab_size, dtype)
                (ls, next_sentence_label, classified, masked_id,
                 decoded, masked_weight, ls1, ls2, valid_length) = out
                loss_list.append(ls)
                ns_label_list.append(next_sentence_label)
                ns_pred_list.append(classified)
                mask_label_list.append(masked_id)
                mask_pred_list.append(decoded)
                mask_weight_list.append(masked_weight)

                running_mlm_loss += ls1.as_in_context(mx.cpu())
                running_nsp_loss += ls2.as_in_context(mx.cpu())
                running_num_tks += valid_length.sum().as_in_context(mx.cpu())
            nsp_metric.update(ns_label_list, ns_pred_list)
            mlm_metric.update(mask_label_list, mask_pred_list, mask_weight_list)

            # logging
            if (step_num + 1) % (log_interval) == 0:
                total_mlm_loss += running_mlm_loss
                total_nsp_loss += running_nsp_loss
                log(begin_time, running_num_tks, running_mlm_loss, running_nsp_loss,
                    step_num, mlm_metric, nsp_metric, None, log_interval)
                begin_time = time.time()
                running_mlm_loss = running_nsp_loss = running_num_tks = 0
                mlm_metric.reset_local()
                nsp_metric.reset_local()

    mx.nd.waitall()
    eval_end_time = time.time()
    # accumulate losses from last few batches, too
    if running_mlm_loss != 0:
        total_mlm_loss += running_mlm_loss
        total_nsp_loss += running_nsp_loss
    total_mlm_loss /= step_num
    total_nsp_loss /= step_num
    logging.info('Eval mlm_loss={:.3f}\tmlm_acc={:.1f}\tnsp_loss={:.3f}\tnsp_acc={:.1f}\t'
                 .format(total_mlm_loss.asscalar(), mlm_metric.get_global()[1] * 100,
                         total_nsp_loss.asscalar(), nsp_metric.get_global()[1] * 100))
    logging.info('Eval cost={:.1f}s'.format(eval_end_time - eval_begin_time))

def get_argparser():
    """Argument parser"""
    parser = argparse.ArgumentParser(description='BERT pretraining example.')
    parser.add_argument('--num_steps', type=int, default=20, help='Number of optimization steps')
    parser.add_argument('--num_buckets', type=int, default=10,
                        help='Number of buckets for variable length sequence sampling')
    parser.add_argument('--dtype', type=str, default='float16', help='data dtype')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU.')
    parser.add_argument('--accumulate', type=int, default=1,
                        help='Number of batches for gradient accumulation. '
                             'The effective batch size = batch_size * accumulate.')
    parser.add_argument('--use_avg_len', action='store_true',
                        help='Use average length information for the bucket sampler. '
                             'The batch size is approximately the number of tokens in the batch')
    parser.add_argument('--batch_size_eval', type=int, default=8,
                        help='Batch size per GPU for evaluation.')
    parser.add_argument('--dataset_name', type=str, default='book_corpus_wiki_en_uncased',
                        choices=['book_corpus_wiki_en_uncased', 'book_corpus_wiki_en_cased',
                                 'wiki_multilingual_uncased', 'wiki_multilingual_cased',
                                 'wiki_cn_cased'],
                        help='The pre-defined dataset from which the vocabulary is created. '
                             'Default is book_corpus_wiki_en_uncased.')
    parser.add_argument('--pretrained', action='store_true',
                        help='Load the pretrained model released by Google.')
    parser.add_argument('--model', type=str, default='bert_12_768_12',
                        help='Model to run pre-training on. '
                             'Options are bert_12_768_12, bert_24_1024_16')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to training data. Training is skipped if not set.')
    parser.add_argument('--data_eval', type=str, required=True,
                        help='Path to evaluation data. Evaluation is skipped if not set.')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt_dir',
                        help='Path to checkpoint directory')
    parser.add_argument('--start_step', type=int, default=0,
                        help='Start optimization step from the checkpoint.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--warmup_ratio', type=float, default=0.01,
                        help='ratio of warmup steps used in NOAM\'s stepsize schedule')
    parser.add_argument('--log_interval', type=int, default=250, help='Report interval')
    parser.add_argument('--ckpt_interval', type=int, default=25000, help='Checkpoint interval')
    parser.add_argument('--dummy_data_len', type=int, default=None,
                        help='If provided, a data batch of target sequence length is '
                             'used. For benchmarking purpuse only.')
    parser.add_argument('--verbose', action='store_true', help='verbose logging')
    parser.add_argument('--profile', type=str, default=None,
                        help='output profiling result to the target file')
    return parser

def generate_dev_set(tokenizer, vocab, cache_file, args):
    """Generate validation set."""
    # set random seed to generate dev data deterministically
    np.random.seed(0)
    random.seed(0)
    mx.random.seed(0)

    worker_pool = multiprocessing.Pool()
    eval_files = glob.glob(os.path.expanduser(args.data_eval))
    num_files = len(eval_files)
    assert num_files > 0, 'Number of eval files must be greater than 0.' \
                          'Only found %d files at %s'%(num_files, args.data_eval)
    logging.info('Generating validation set from %d files on rank 0.', len(eval_files))
    create_training_instances((eval_files, tokenizer, args.max_seq_length,
                               args.short_seq_prob, args.masked_lm_prob,
                               args.max_predictions_per_seq, vocab,
                               1, args.num_data_workers,
                               worker_pool, cache_file))
    logging.info('Done generating validation set on rank 0.')
