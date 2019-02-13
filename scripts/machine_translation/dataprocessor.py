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
"""Data preprocessing for transformer."""

import os
import io
import time
import logging
import numpy as np
from mxnet import gluon
import gluonnlp as nlp
import gluonnlp.data.batchify as btf
import _constants
import dataset as _dataset


def _cache_dataset(dataset, prefix):
    """Cache the processed npy dataset the dataset into a npz

    Parameters
    ----------
    dataset : SimpleDataset
    file_path : str
    """
    if not os.path.exists(_constants.CACHE_PATH):
        os.makedirs(_constants.CACHE_PATH)
    src_data = np.concatenate([e[0] for e in dataset])
    tgt_data = np.concatenate([e[1] for e in dataset])
    src_cumlen = np.cumsum([0]+[len(e[0]) for e in dataset])
    tgt_cumlen = np.cumsum([0]+[len(e[1]) for e in dataset])
    np.savez(os.path.join(_constants.CACHE_PATH, prefix + '.npz'),
             src_data=src_data, tgt_data=tgt_data,
             src_cumlen=src_cumlen, tgt_cumlen=tgt_cumlen)


def _load_cached_dataset(prefix):
    cached_file_path = os.path.join(_constants.CACHE_PATH, prefix + '.npz')
    if os.path.exists(cached_file_path):
        print('Loading dataset...')
        npz_data = np.load(cached_file_path)
        src_data, tgt_data, src_cumlen, tgt_cumlen = \
                [npz_data[n] for n in ['src_data', 'tgt_data', 'src_cumlen', 'tgt_cumlen']]
        src_data = np.array([src_data[low:high] for low, high
                             in zip(src_cumlen[:-1], src_cumlen[1:])])
        tgt_data = np.array([tgt_data[low:high] for low, high
                             in zip(tgt_cumlen[:-1], tgt_cumlen[1:])])
        return gluon.data.ArrayDataset(np.array(src_data), np.array(tgt_data))
    else:
        return None


class TrainValDataTransform(object):
    """Transform the machine translation dataset.

    Clip source and the target sentences to the maximum length. For the source sentence, append the
    EOS. For the target sentence, append BOS and EOS.

    Parameters
    ----------
    src_vocab : Vocab
    tgt_vocab : Vocab
    src_max_len : int
    tgt_max_len : int
    """

    def __init__(self, src_vocab, tgt_vocab, src_max_len=None, tgt_max_len=None):
        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab
        self._src_max_len = src_max_len
        self._tgt_max_len = tgt_max_len

    def __call__(self, src, tgt):
        # For src_max_len < 0, we do not clip the sequence
        if self._src_max_len >= 0:
            src_sentence = self._src_vocab[src.split()[:self._src_max_len]]
        else:
            src_sentence = self._src_vocab[src.split()]
        # For tgt_max_len < 0, we do not clip the sequence
        if self._tgt_max_len >= 0:
            tgt_sentence = self._tgt_vocab[tgt.split()[:self._tgt_max_len]]
        else:
            tgt_sentence = self._tgt_vocab[tgt.split()]
        src_sentence.append(self._src_vocab[self._src_vocab.eos_token])
        tgt_sentence.insert(0, self._tgt_vocab[self._tgt_vocab.bos_token])
        tgt_sentence.append(self._tgt_vocab[self._tgt_vocab.eos_token])
        src_npy = np.array(src_sentence, dtype=np.int32)
        tgt_npy = np.array(tgt_sentence, dtype=np.int32)
        return src_npy, tgt_npy


def process_dataset(dataset, src_vocab, tgt_vocab, src_max_len=-1, tgt_max_len=-1):
    start = time.time()
    dataset_processed = dataset.transform(TrainValDataTransform(src_vocab, tgt_vocab,
                                                                src_max_len,
                                                                tgt_max_len), lazy=False)
    end = time.time()
    print('Processing Time spent: {}'.format(end - start))
    return dataset_processed


def load_translation_data(dataset, bleu, args):
    """Load translation dataset

    Parameters
    ----------
    dataset : str
    args : argparse result

    Returns
    -------

    """
    src_lang, tgt_lang = args.src_lang, args.tgt_lang
    if dataset == 'IWSLT2015':
        common_prefix = 'IWSLT2015_{}_{}_{}_{}'.format(src_lang, tgt_lang,
                                                       args.src_max_len, args.tgt_max_len)
        data_train = nlp.data.IWSLT2015('train', src_lang=src_lang, tgt_lang=tgt_lang)
        data_val = nlp.data.IWSLT2015('val', src_lang=src_lang, tgt_lang=tgt_lang)
        data_test = nlp.data.IWSLT2015('test', src_lang=src_lang, tgt_lang=tgt_lang)
    elif dataset == 'WMT2016BPE':
        common_prefix = 'WMT2016BPE_{}_{}_{}_{}'.format(src_lang, tgt_lang,
                                                        args.src_max_len, args.tgt_max_len)
        data_train = nlp.data.WMT2016BPE('train', src_lang=src_lang, tgt_lang=tgt_lang)
        data_val = nlp.data.WMT2016BPE('newstest2013', src_lang=src_lang, tgt_lang=tgt_lang)
        data_test = nlp.data.WMT2016BPE('newstest2014', src_lang=src_lang, tgt_lang=tgt_lang)
    elif dataset == 'WMT2014BPE':
        common_prefix = 'WMT2014BPE_{}_{}_{}_{}'.format(src_lang, tgt_lang,
                                                        args.src_max_len, args.tgt_max_len)
        data_train = nlp.data.WMT2014BPE('train', src_lang=src_lang, tgt_lang=tgt_lang)
        data_val = nlp.data.WMT2014BPE('newstest2013', src_lang=src_lang, tgt_lang=tgt_lang)
        data_test = nlp.data.WMT2014BPE('newstest2014', src_lang=src_lang, tgt_lang=tgt_lang,
                                        full=args.full)
    elif dataset == 'TOY':
        common_prefix = 'TOY_{}_{}_{}_{}'.format(src_lang, tgt_lang,
                                                 args.src_max_len, args.tgt_max_len)
        data_train = _dataset.TOY('train', src_lang=src_lang, tgt_lang=tgt_lang)
        data_val = _dataset.TOY('val', src_lang=src_lang, tgt_lang=tgt_lang)
        data_test = _dataset.TOY('test', src_lang=src_lang, tgt_lang=tgt_lang)
    else:
        raise NotImplementedError
    src_vocab, tgt_vocab = data_train.src_vocab, data_train.tgt_vocab
    data_train_processed = _load_cached_dataset(common_prefix + '_train')
    if not data_train_processed:
        data_train_processed = process_dataset(data_train, src_vocab, tgt_vocab,
                                               args.src_max_len, args.tgt_max_len)
        _cache_dataset(data_train_processed, common_prefix + '_train')
    data_val_processed = _load_cached_dataset(common_prefix + '_val')
    if not data_val_processed:
        data_val_processed = process_dataset(data_val, src_vocab, tgt_vocab)
        _cache_dataset(data_val_processed, common_prefix + '_val')
    if dataset == 'WMT2014BPE':
        filename = common_prefix + '_' + str(args.full) + '_test'
    else:
        filename = common_prefix + '_test'
    data_test_processed = _load_cached_dataset(filename)
    if not data_test_processed:
        data_test_processed = process_dataset(data_test, src_vocab, tgt_vocab)
        _cache_dataset(data_test_processed, filename)
    if bleu == 'tweaked':
        fetch_tgt_sentence = lambda src, tgt: tgt.split()
        val_tgt_sentences = list(data_val.transform(fetch_tgt_sentence))
        test_tgt_sentences = list(data_test.transform(fetch_tgt_sentence))
    elif bleu == '13a' or bleu == 'intl':
        fetch_tgt_sentence = lambda src, tgt: tgt
        if dataset == 'WMT2016BPE':
            val_text = nlp.data.WMT2016('newstest2013', src_lang=src_lang, tgt_lang=tgt_lang)
            test_text = nlp.data.WMT2016('newstest2014', src_lang=src_lang, tgt_lang=tgt_lang)
        elif dataset == 'WMT2014BPE':
            val_text = nlp.data.WMT2014('newstest2013', src_lang=src_lang, tgt_lang=tgt_lang)
            test_text = nlp.data.WMT2014('newstest2014', src_lang=src_lang, tgt_lang=tgt_lang,
                                         full=args.full)
        elif dataset == 'IWSLT2015' or dataset == 'TOY':
            val_text = data_val
            test_text = data_test
        else:
            raise NotImplementedError
        val_tgt_sentences = list(val_text.transform(fetch_tgt_sentence))
        test_tgt_sentences = list(test_text.transform(fetch_tgt_sentence))
    else:
        raise NotImplementedError
    return data_train_processed, data_val_processed, data_test_processed, \
           val_tgt_sentences, test_tgt_sentences, src_vocab, tgt_vocab


def get_data_lengths(dataset):
    get_lengths = lambda *args: (args[2], args[3])
    return list(dataset.transform(get_lengths))


def make_dataloader(data_train, data_val, data_test, args,
                    use_average_length=False, num_shards=0, num_workers=8):
    """Create data loaders for training/validation/test."""
    data_train_lengths = get_data_lengths(data_train)
    data_val_lengths = get_data_lengths(data_val)
    data_test_lengths = get_data_lengths(data_test)
    train_batchify_fn = btf.Tuple(btf.Pad(), btf.Pad(),
                                  btf.Stack(dtype='float32'), btf.Stack(dtype='float32'))
    test_batchify_fn = btf.Tuple(btf.Pad(), btf.Pad(),
                                 btf.Stack(dtype='float32'), btf.Stack(dtype='float32'),
                                 btf.Stack())
    target_val_lengths = list(map(lambda x: x[-1], data_val_lengths))
    target_test_lengths = list(map(lambda x: x[-1], data_test_lengths))
    if args.bucket_scheme == 'constant':
        bucket_scheme = nlp.data.ConstWidthBucket()
    elif args.bucket_scheme == 'linear':
        bucket_scheme = nlp.data.LinearWidthBucket()
    elif args.bucket_scheme == 'exp':
        bucket_scheme = nlp.data.ExpWidthBucket(bucket_len_step=1.2)
    else:
        raise NotImplementedError
    train_batch_sampler = nlp.data.FixedBucketSampler(lengths=data_train_lengths,
                                                      batch_size=args.batch_size,
                                                      num_buckets=args.num_buckets,
                                                      ratio=args.bucket_ratio,
                                                      shuffle=True,
                                                      use_average_length=use_average_length,
                                                      num_shards=num_shards,
                                                      bucket_scheme=bucket_scheme)
    logging.info('Train Batch Sampler:\n%s', train_batch_sampler.stats())
    train_data_loader = nlp.data.ShardedDataLoader(data_train,
                                                   batch_sampler=train_batch_sampler,
                                                   batchify_fn=train_batchify_fn,
                                                   num_workers=num_workers)

    val_batch_sampler = nlp.data.FixedBucketSampler(lengths=target_val_lengths,
                                                    batch_size=args.test_batch_size,
                                                    num_buckets=args.num_buckets,
                                                    ratio=args.bucket_ratio,
                                                    shuffle=False,
                                                    use_average_length=use_average_length,
                                                    bucket_scheme=bucket_scheme)
    logging.info('Valid Batch Sampler:\n%s', val_batch_sampler.stats())
    val_data_loader = gluon.data.DataLoader(data_val,
                                            batch_sampler=val_batch_sampler,
                                            batchify_fn=test_batchify_fn,
                                            num_workers=num_workers)
    test_batch_sampler = nlp.data.FixedBucketSampler(lengths=target_test_lengths,
                                                     batch_size=args.test_batch_size,
                                                     num_buckets=args.num_buckets,
                                                     ratio=args.bucket_ratio,
                                                     shuffle=False,
                                                     use_average_length=use_average_length,
                                                     bucket_scheme=bucket_scheme)
    logging.info('Test Batch Sampler:\n%s', test_batch_sampler.stats())
    test_data_loader = gluon.data.DataLoader(data_test,
                                             batch_sampler=test_batch_sampler,
                                             batchify_fn=test_batchify_fn,
                                             num_workers=num_workers)
    return train_data_loader, val_data_loader, test_data_loader


def write_sentences(sentences, file_path):
    with io.open(file_path, 'w', encoding='utf-8') as of:
        for sent in sentences:
            if isinstance(sent, (list, tuple)):
                of.write(u' '.join(sent) + u'\n')
            else:
                of.write(sent + u'\n')
