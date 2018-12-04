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
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
import nmt
import hyperparameters as hparams

def cache_dataset(dataset, prefix):
    """Cache the processed npy dataset the dataset into a npz

    Parameters
    ----------
    dataset : SimpleDataset
    file_path : str
    """
    if not os.path.exists(nmt._constants.CACHE_PATH):
        os.makedirs(nmt._constants.CACHE_PATH)
    src_data = np.concatenate([e[0] for e in dataset])
    tgt_data = np.concatenate([e[1] for e in dataset])
    src_cumlen = np.cumsum([0]+[len(e[0]) for e in dataset])
    tgt_cumlen = np.cumsum([0]+[len(e[1]) for e in dataset])
    np.savez(os.path.join(nmt._constants.CACHE_PATH, prefix + '.npz'),
             src_data=src_data, tgt_data=tgt_data,
             src_cumlen=src_cumlen, tgt_cumlen=tgt_cumlen)


def load_cached_dataset(prefix):
    cached_file_path = os.path.join(nmt._constants.CACHE_PATH, prefix + '.npz')
    if os.path.exists(cached_file_path):
        print('Loading dataset...')
        npz_data = np.load(cached_file_path)
        src_data, tgt_data, src_cumlen, tgt_cumlen = [npz_data[n] for n in
                ['src_data', 'tgt_data', 'src_cumlen', 'tgt_cumlen']]
        src_data = np.array([src_data[low:high] for low, high in zip(src_cumlen[:-1], src_cumlen[1:])])
        tgt_data = np.array([tgt_data[low:high] for low, high in zip(tgt_cumlen[:-1], tgt_cumlen[1:])])
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
        if self._src_max_len:
            src_sentence = self._src_vocab[src.split()[:self._src_max_len]]
        else:
            src_sentence = self._src_vocab[src.split()]
        if self._tgt_max_len:
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


def load_translation_data(dataset, src_lang='en', tgt_lang='de'):
    """Load translation dataset

    Parameters
    ----------
    dataset : str
    src_lang : str, default 'en'
    tgt_lang : str, default 'de'

    Returns
    -------

    """
    if dataset == 'WMT2014BPE':
        common_prefix = 'WMT2014BPE_{}_{}_{}_{}'.format(src_lang, tgt_lang,
                                                        hparams.src_max_len, hparams.tgt_max_len)
        data_train = nlp.data.WMT2014BPE('train', src_lang=src_lang, tgt_lang=tgt_lang)
        data_val = nlp.data.WMT2014BPE('newstest2013', src_lang=src_lang, tgt_lang=tgt_lang)
        data_test = nlp.data.WMT2014BPE('newstest2014', src_lang=src_lang, tgt_lang=tgt_lang,
                               full=False)
    elif dataset == 'TOY':
        common_prefix = 'TOY_{}_{}_{}_{}'.format(src_lang, tgt_lang,
                                                 hparams.src_max_len, hparams.tgt_max_len)
        data_train = nmt.dataset.TOY('train', src_lang=src_lang, tgt_lang=tgt_lang)
        data_val = nmt.dataset.TOY('val', src_lang=src_lang, tgt_lang=tgt_lang)
        data_test = nmt.dataset.TOY('test', src_lang=src_lang, tgt_lang=tgt_lang)
    else:
        raise NotImplementedError
    src_vocab, tgt_vocab = data_train.src_vocab, data_train.tgt_vocab
    data_train_processed = load_cached_dataset(common_prefix + '_train')
    if not data_train_processed:
        data_train_processed = process_dataset(data_train, src_vocab, tgt_vocab,
                                               hparams.src_max_len, hparams.tgt_max_len)
        cache_dataset(data_train_processed, common_prefix + '_train')
    data_val_processed = load_cached_dataset(common_prefix + '_val')
    if not data_val_processed:
        data_val_processed = process_dataset(data_val, src_vocab, tgt_vocab)
        cache_dataset(data_val_processed, common_prefix + '_val')
    data_test_processed = load_cached_dataset(common_prefix + '_' + str(False) + '_test')
    if not data_test_processed:
        data_test_processed = process_dataset(data_test, src_vocab, tgt_vocab)
        cache_dataset(data_test_processed, common_prefix + '_' + str(False) + '_test')
    fetch_tgt_sentence = lambda src, tgt: tgt
    if dataset == 'WMT2014BPE':
        val_text = nlp.data.WMT2014('newstest2013', src_lang=src_lang, tgt_lang=tgt_lang)
        test_text = nlp.data.WMT2014('newstest2014', src_lang=src_lang, tgt_lang=tgt_lang,
                            full=False)
    elif dataset == 'TOY':
        val_text = data_val
        test_text = data_test
    else:
        raise NotImplementedError
    val_tgt_sentences = list(val_text.transform(fetch_tgt_sentence))
    test_tgt_sentences = list(test_text.transform(fetch_tgt_sentence))
    return data_train_processed, data_val_processed, data_test_processed, val_tgt_sentences, test_tgt_sentences, src_vocab, tgt_vocab


def get_data_lengths(dataset):
    return list(dataset.transform(lambda srg, tgt: (len(srg), len(tgt))))
