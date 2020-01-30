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
import multiprocessing

import numpy as np
import mxnet as mx
import gluonnlp as nlp

from data.create_pretraining_data import create_training_instances


__all__ = ['get_model_loss', 'get_pretrain_data_npz', 'get_dummy_dataloader',
           'save_parameters', 'save_states', 'evaluate', 'split_and_load',
           'get_pretrain_data_text', 'generate_dev_set', 'profile']

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
    BERTForPretrain : the model for pre-training.
    BERTVocab : the vocabulary.
    """
    # model
    model, vocabulary = nlp.model.get_model(model, dataset_name=dataset_name, vocab=vocab,
                                            pretrained=pretrained, ctx=ctx,
                                            hparam_allow_override=True)

    if not pretrained:
        model.initialize(init=mx.init.Normal(0.02), ctx=ctx)
    model.cast(dtype)

    if ckpt_dir and start_step:
        param_path = os.path.join(ckpt_dir, '%07d.params'%start_step)
        nlp.utils.load_parameters(model, param_path, ctx=ctx, cast_dtype=True)
        logging.info('Loading step %d checkpoints from %s.', start_step, param_path)

    model.hybridize(static_alloc=True, static_shape=True)

    # losses
    nsp_loss = mx.gluon.loss.SoftmaxCELoss()
    mlm_loss = mx.gluon.loss.SoftmaxCELoss()
    nsp_loss.hybridize(static_alloc=True, static_shape=True)
    mlm_loss.hybridize(static_alloc=True, static_shape=True)

    model = BERTForPretrain(model, nsp_loss, mlm_loss, len(vocabulary))
    return model, vocabulary


def prepare_pretrain_npz_dataset(filename, allow_pickle=False):
    """Create dataset based on the numpy npz file"""
    if isinstance(filename, (list, tuple)):
        assert len(filename) == 1, \
            'When .npy/.npz data file is loaded, len(filename) must be 1.' \
            ' Received len(filename)={}.'.format(len(filename))
        filename = filename[0]
    logging.debug('start to load file %s ...', filename)
    return nlp.data.NumpyDataset(filename, allow_pickle=allow_pickle)


def prepare_pretrain_text_dataset(filename, tokenizer, max_seq_length, short_seq_prob,
                                  masked_lm_prob, max_predictions_per_seq, whole_word_mask,
                                  vocab, num_workers=1, worker_pool=None):
    """Create dataset based on the raw text files"""
    dupe_factor = 1
    if not isinstance(filename, (list, tuple)):
        filename = [filename]
    logging.debug('start to load files %s ...', filename)
    instances = create_training_instances((filename, tokenizer, max_seq_length,
                                           short_seq_prob, masked_lm_prob,
                                           max_predictions_per_seq,
                                           whole_word_mask, vocab,
                                           dupe_factor, num_workers,
                                           worker_pool, None))
    return mx.gluon.data.ArrayDataset(*instances)


def prepare_pretrain_bucket_sampler(dataset, batch_size, shuffle=False,
                                    num_ctxes=1, num_buckets=1):
    """Create data sampler based on the dataset"""
    if isinstance(dataset, nlp.data.NumpyDataset):
        lengths = dataset.get_field('valid_lengths')
    else:
        lengths = dataset.transform(lambda input_ids, segment_ids, masked_lm_positions, \
                                           masked_lm_ids, masked_lm_weights, \
                                           next_sentence_labels, valid_lengths: \
                                        valid_lengths, lazy=False)
    # calculate total batch size for all GPUs
    batch_size = batch_size * num_ctxes
    sampler = nlp.data.FixedBucketSampler(lengths,
                                          batch_size=batch_size,
                                          num_buckets=num_buckets,
                                          ratio=0,
                                          shuffle=shuffle)
    logging.debug('Sampler created for a new dataset:\n%s', sampler.stats())
    return sampler


def get_pretrain_data_text(data, batch_size, num_ctxes, shuffle,
                           num_buckets, vocab, tokenizer, max_seq_length, short_seq_prob,
                           masked_lm_prob, max_predictions_per_seq, whole_word_mask,
                           num_parts=1, part_idx=0, num_dataset_workers=1, num_batch_workers=1,
                           circle_length=1, repeat=1,
                           dataset_cached=False, num_max_dataset_cached=0):
    """Get a data iterator from raw text documents.

    Parameters
    ----------
    batch_size : int
        The batch size per GPU.
    num_ctxes : int
        The number of GPUs.
    shuffle : bool
        Whether to shuffle the data.
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
    num_files = len(nlp.utils.glob(data))
    logging.info('%d files are found.', num_files)
    assert num_files >= num_parts, \
        'The number of text files must be no less than the number of ' \
        'workers/partitions (%d). Only %d files at %s are found.'%(num_parts, num_files, data)
    dataset_params = {'tokenizer': tokenizer, 'max_seq_length': max_seq_length,
                      'short_seq_prob': short_seq_prob, 'masked_lm_prob': masked_lm_prob,
                      'max_predictions_per_seq': max_predictions_per_seq, 'vocab':vocab,
                      'whole_word_mask': whole_word_mask}
    sampler_params = {'batch_size': batch_size, 'shuffle': shuffle,
                      'num_ctxes': num_ctxes, 'num_buckets': num_buckets}
    dataset_fn = prepare_pretrain_text_dataset
    sampler_fn = prepare_pretrain_bucket_sampler
    pad_val = vocab[vocab.padding_token]
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(pad_val=pad_val, round_to=8),  # input_id
        nlp.data.batchify.Pad(pad_val=pad_val),  # masked_id
        nlp.data.batchify.Pad(pad_val=0),  # masked_position
        nlp.data.batchify.Pad(pad_val=0),  # masked_weight
        nlp.data.batchify.Stack(),  # next_sentence_label
        nlp.data.batchify.Pad(pad_val=0, round_to=8),  # segment_id
        nlp.data.batchify.Stack())
    split_sampler = nlp.data.SplitSampler(num_files, num_parts=num_parts,
                                          part_index=part_idx, repeat=repeat)
    dataloader = nlp.data.DatasetLoader(data,
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


def get_pretrain_data_npz(data, batch_size, num_ctxes,
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
    num_ctxes : int
        The number of GPUs.
    shuffle : bool
        Whether to shuffle the data.
    num_buckets : int
        The number of buckets for the FixedBucketSampler for training.
    vocab : BERTVocab
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
    num_files = len(nlp.utils.glob(data))
    logging.info('%d files are found.', num_files)
    assert num_files >= num_parts, \
        'The number of text files must be no less than the number of ' \
        'workers/partitions (%d). Only %d files at %s are found.'%(num_parts, num_files, data)
    dataset_params = {'allow_pickle': True}
    sampler_params = {'batch_size': batch_size, 'shuffle': shuffle,
                      'num_ctxes': num_ctxes, 'num_buckets': num_buckets}
    dataset_fn = prepare_pretrain_npz_dataset
    sampler_fn = prepare_pretrain_bucket_sampler
    pad_val = vocab[vocab.padding_token]
    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(pad_val=pad_val, round_to=8),  # input_id
        nlp.data.batchify.Pad(pad_val=pad_val),  # masked_id
        nlp.data.batchify.Pad(pad_val=0),  # masked_position
        nlp.data.batchify.Pad(pad_val=0),  # masked_weight
        nlp.data.batchify.Stack(),  # next_sentence_label
        nlp.data.batchify.Pad(pad_val=0, round_to=8),  # segment_id
        nlp.data.batchify.Stack())
    split_sampler = nlp.data.SplitSampler(num_files, num_parts=num_parts,
                                          part_index=part_idx, repeat=repeat)
    dataloader = nlp.data.DatasetLoader(data,
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


def get_dummy_dataloader(batch_size, seq_len, max_predict):
    """Return a dummy data loader which returns a fixed data batch of target shape"""
    class DummyIter():
        def __init__(self, batch):
            self._batch = batch

        def __iter__(self):
            while True:
                yield self._batch
    data_batch = ((mx.nd.zeros((batch_size, seq_len)),
                   mx.nd.zeros((batch_size, max_predict)),
                   mx.nd.zeros((batch_size, max_predict)),
                   mx.nd.zeros((batch_size, max_predict)),
                   mx.nd.ones((batch_size,)) * seq_len,
                   mx.nd.zeros((batch_size, seq_len)),
                   mx.nd.ones((batch_size,)) * seq_len))
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

def log_noacc(begin_time, running_num_tks, running_mlm_loss, running_nsp_loss, step_num,
              trainer, log_interval):
    """Log training progress."""
    end_time = time.time()
    duration = end_time - begin_time
    throughput = running_num_tks / duration / 1000.0
    running_mlm_loss = running_mlm_loss / log_interval
    running_nsp_loss = running_nsp_loss / log_interval
    lr = trainer.learning_rate if trainer else 0
    # pylint: disable=line-too-long
    logging.info('[step {}]\tmlm_loss={:7.5f}\tnsp_loss={:5.2f}\tthroughput={:.1f}K tks/s\tlr={:.7f} time={:.2f}, latency={:.1f} ms/step'
                 .format(step_num, running_mlm_loss.asscalar(), running_nsp_loss.asscalar(),
                         throughput.asscalar(), lr, duration, duration*1000/log_interval))
    # pylint: enable=line-too-long

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
    logging.info('[step {}]\tmlm_loss={:7.5f}\tmlm_acc={:4.2f}\tnsp_loss={:5.2f}\tnsp_acc={:5.2f}\tthroughput={:.1f}K tks/s\tlr={:.7f} time={:.2f}, latency={:.1f} ms/step'
                 .format(step_num, running_mlm_loss.asscalar(), mlm_metric.get()[1] * 100, running_nsp_loss.asscalar(),
                         nsp_metric.get()[1] * 100, throughput.asscalar(), lr, duration, duration*1000/log_interval))
    # pylint: enable=line-too-long


def split_and_load(arrs, ctx):
    """split and load arrays to a list of contexts"""
    assert isinstance(arrs, (list, tuple))
    # split and load
    loaded_arrs = [mx.gluon.utils.split_and_load(arr, ctx, even_split=False) for arr in arrs]
    return zip(*loaded_arrs)


class BERTForPretrain(mx.gluon.Block):
    """Model for pre-training MLM and NSP with BERT.

    Parameters
    ----------
    bert: BERTModel
        Bidirectional encoder with transformer.
    mlm_loss : Loss or None
    nsp_loss : Loss or None
    vocab_size : int
    prefix : str or None
        See document of `mx.gluon.Block`.
    params : ParameterDict or None
        See document of `mx.gluon.Block`.
    """

    def __init__(self, bert, mlm_loss, nsp_loss, vocab_size, prefix=None, params=None):
        super(BERTForPretrain, self).__init__(prefix=prefix, params=params)
        self.bert = bert
        self.mlm_loss = mlm_loss
        self.nsp_loss = nsp_loss
        self._vocab_size = vocab_size

    def forward(self, input_id, masked_id, masked_position, masked_weight,
                next_sentence_label=None, segment_id=None, valid_length=None):
        # pylint: disable=arguments-differ
        """Predict with BERT for MLM and NSP. """
        num_masks = masked_weight.sum() + 1e-8
        valid_length = valid_length.reshape(-1)
        masked_id = masked_id.reshape(-1)
        _, _, classified, decoded = self.bert(input_id, segment_id, valid_length, masked_position)
        decoded = decoded.reshape((-1, self._vocab_size))
        ls1 = self.mlm_loss(decoded.astype('float32', copy=False),
                            masked_id, masked_weight.reshape((-1, 1)))
        ls2 = self.nsp_loss(classified.astype('float32', copy=False), next_sentence_label)
        ls1 = ls1.sum() / num_masks
        ls2 = ls2.mean()
        return classified, decoded, ls1, ls2


def evaluate(data_eval, model, ctx, log_interval, dtype):
    """Evaluation function."""
    logging.info('Running evaluation ... ')
    mlm_metric = nlp.metric.MaskedAccuracy()
    nsp_metric = nlp.metric.MaskedAccuracy()
    mlm_metric.reset()
    nsp_metric.reset()

    eval_begin_time = time.time()
    begin_time = time.time()
    step_num = 0
    running_mlm_loss = running_nsp_loss = 0
    total_mlm_loss = total_nsp_loss = 0
    running_num_tks = 0
    for _, data_batch in enumerate(data_eval):
        step_num += 1

        data_list = split_and_load(data_batch, ctx)
        ns_label_list, ns_pred_list = [], []
        mask_label_list, mask_pred_list, mask_weight_list = [], [], []
        for data in data_list:
            (input_id, masked_id, masked_position, masked_weight, \
             next_sentence_label, segment_id, valid_length) = data
            valid_length = valid_length.astype(dtype, copy=False)
            out = model(input_id, masked_id, masked_position, masked_weight, \
                        next_sentence_label, segment_id, valid_length)
            classified, decoded, ls1, ls2 = out
            masked_id = masked_id.reshape(-1)
            ns_label_list.append(next_sentence_label)
            ns_pred_list.append(classified)
            mask_label_list.append(masked_id)
            mask_pred_list.append(decoded)
            mask_weight_list.append(masked_weight)

            valid_length = valid_length.astype('float32', copy=False)
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


def generate_dev_set(tokenizer, vocab, cache_file, args):
    """Generate validation set."""
    # set random seed to generate dev data deterministically
    np.random.seed(0)
    random.seed(0)
    mx.random.seed(0)
    worker_pool = multiprocessing.Pool()
    eval_files = nlp.utils.glob(args.data_eval)
    num_files = len(eval_files)
    assert num_files > 0, 'Number of eval files must be greater than 0.' \
                          'Only found %d files at %s'%(num_files, args.data_eval)
    logging.info('Generating validation set from %d files on rank 0.', len(eval_files))
    create_training_instances((eval_files, tokenizer, args.max_seq_length,
                               args.short_seq_prob, args.masked_lm_prob,
                               args.max_predictions_per_seq,
                               args.whole_word_mask, vocab,
                               1, args.num_dataset_workers,
                               worker_pool, cache_file))
    logging.info('Done generating validation set on rank 0.')

def profile(curr_step, start_step, end_step, profile_name='profile.json',
            early_exit=True):
    """profile the program between [start_step, end_step)."""
    if curr_step == start_step:
        mx.nd.waitall()
        mx.profiler.set_config(profile_memory=False, profile_symbolic=True,
                               profile_imperative=True, filename=profile_name,
                               aggregate_stats=True)
        mx.profiler.set_state('run')
    elif curr_step == end_step:
        mx.nd.waitall()
        mx.profiler.set_state('stop')
        logging.info(mx.profiler.dumps())
        mx.profiler.dump()
        if early_exit:
            sys.exit(0)
