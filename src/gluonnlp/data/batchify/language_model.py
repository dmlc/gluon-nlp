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

# pylint: disable=undefined-all-variable
"""Data transforms useful for language models."""

__all__ = ['CorpusBatchify', 'CorpusBPTTBatchify', 'StreamBPTTBatchify']

import itertools
import math

import numpy as np
import mxnet as mx
from mxnet.gluon.data import RandomSampler, SequentialSampler, SimpleDataset

from ..utils import slice_sequence, _slice_pad_length
from ..stream import DataStream

class CorpusBatchify(object):
    """Transform the dataset into N independent sequences, where N is the batch size.

    Parameters
    ----------
    vocab : gluonnlp.Vocab
        The vocabulary to use for numericalizing the dataset. Each token will be mapped to the
        index according to the vocabulary.
    batch_size : int
        The number of samples in each batch.
    """

    def __init__(self, vocab, batch_size):
        self._vocab = vocab
        self._batch_size = batch_size

    def __call__(self, data):
        """Batchify a dataset.

        Parameters
        ----------
        data : mxnet.gluon.data.Dataset
            A flat dataset to be batchified.

        Returns
        -------
        mxnet.gluon.data.Dataset
            NDArray of shape (len(data) // N, N) where N is the batch_size
            wrapped by a mxnet.gluon.data.SimpleDataset. Excessive tokens that
            don't align along the batches are discarded.
        """
        sample_len = len(data) // self._batch_size
        return SimpleDataset(
            mx.nd.array(
                self._vocab[data[:sample_len * self._batch_size]]).reshape(
                    self._batch_size, -1).T)


class CorpusBPTTBatchify(object):
    """Transform the dataset into batches of numericalized samples, in the way
    that the recurrent states from last batch connects with the current batch
    for each sample.

    Each sample is of shape `(seq_len, batch_size)`. When `last_batch='keep'`, the first
    dimension of last sample may be shorter than `seq_len`.

    Parameters
    ----------
    vocab : gluonnlp.Vocab
        The vocabulary to use for numericalizing the dataset. Each token will be mapped to the
        index according to the vocabulary.
    seq_len : int
        The length of each of the samples for truncated back-propagation-through-time (TBPTT).
    batch_size : int
        The number of samples in each batch.
    last_batch : {'keep', 'discard'}
        How to handle the last batch if the remaining length is less than `seq_len`.

        - keep: A batch with less samples than previous batches is returned. vocab.padding_token
          is used to pad the last batch based on batch size.

        - discard: The last batch is discarded if it's smaller than `(seq_len, batch_size)`.
    """

    def __init__(self,
                 vocab,
                 seq_len,
                 batch_size,
                 last_batch='keep'):
        self._vocab = vocab
        self._seq_len = seq_len
        self._batch_size = batch_size
        self._last_batch = last_batch
        self._padding_idx = vocab[vocab.padding_token]

        if last_batch not in ['keep', 'discard']:
            raise ValueError(
                'Got invalid last_batch: "{}". Must be "keep" or "discard".'.
                format(last_batch))

        if self._last_batch == 'keep':
            if not self._vocab.padding_token:
                raise ValueError('vocab.padding_token must be specified '
                                 'in vocab when last_batch="keep".')

    def __call__(self, corpus):
        """Batchify a dataset.

        Parameters
        ----------
        corpus : mxnet.gluon.data.Dataset
            A flat dataset to be batchified.

        Returns
        -------
        mxnet.gluon.data.Dataset
            Batches of numericalized samples such that the recurrent states
            from last batch connects with the current batch for each sample.
            Each element of the Dataset is a tuple of data and label arrays for
            BPTT. They are of shape (seq_len, batch_size) respectively.
        """
        if self._last_batch == 'keep':
            coded = self._vocab[list(corpus)]
            sample_len = math.ceil(float(len(coded)) / self._batch_size)
            padding_size = _slice_pad_length(sample_len, self._seq_len + 1, 1) * \
                self._batch_size + sample_len * self._batch_size - len(coded)
            coded.extend([self._vocab[self._vocab.padding_token]] * int(padding_size))
            assert len(coded) % self._batch_size == 0
            assert not _slice_pad_length(len(coded) / self._batch_size, self._seq_len + 1, 1)
        else:
            sample_len = len(corpus) // self._batch_size
            coded = self._vocab[corpus[:sample_len * self._batch_size]]
        data = mx.nd.array(coded).reshape((self._batch_size, -1)).T
        batches = slice_sequence(data, self._seq_len + 1, overlap=1)

        return SimpleDataset(batches).transform(_split_data_label, lazy=False)


def _split_data_label(x):
    return x[:-1, :], x[1:, :]


class StreamBPTTBatchify(object):
    """Transform a Stream of CorpusDataset to BPTT batches.

    The corpus is transformed into batches of numericalized samples, in the way that the
    recurrent states from last batch connects with the current batch for each sample.

    Each sample is of shape `(seq_len, batch_size)`.

    For example, the following 4 sequences::

        a b c d <eos>
        e f g h i j <eos>
        k l m n <eos>
        o <eos>

    will generate 2 batches with seq_len = 5, batch_size = 2 as follow (transposed):

    batch_0.data.T::

        a b c d <eos>
        e f g h i

    batch_0.target.T::

        b c d <eos> k
        f g h i j

    batch_1.data.T::

        k l m n <eos>
        j <eos> o <eos> <padding>

    batch_1.target.T::

        l m n <eos> <padding>
        <eos> o <eos> <padding> <padding>

    Parameters
    ----------
    vocab : gluonnlp.Vocab
        The vocabulary to use for numericalizing the dataset. Each token will be mapped to the
        index according to the vocabulary.
    seq_len : int
        The length of each of the samples for truncated back-propagation-through-time (TBPTT).
    batch_size : int
        The number of samples in each batch.
    sampler : str, {'sequential', 'random'}, defaults to 'random'
        The sampler used to sample texts within a file.

        - 'sequential': SequentialSampler
        - 'random': RandomSampler
    last_batch : {'keep', 'discard'}
        How to handle the last batch if the remaining length is less than `seq_len`.

        - keep: A batch with less samples than previous batches is returned.
        - discard: The last batch is discarded if it's smaller than `(seq_len, batch_size)`.
    """

    def __init__(self,
                 vocab,
                 seq_len,
                 batch_size,
                 sampler='random',
                 last_batch='keep'):
        self._vocab = vocab
        self._seq_len = seq_len
        self._batch_size = batch_size
        self._sampler = sampler
        self._last_batch = last_batch
        if not self._vocab.padding_token:
            raise ValueError('Padding token must be specified in vocab for BPTT.')
        self._padding_idx = vocab[vocab.padding_token]

        if last_batch not in ['keep', 'discard']:
            raise ValueError(
                'Got invalid last_batch: "{}". Must be "keep" or "discard".'.
                format(last_batch))

    def _get_sampler(self, sampler):
        assert isinstance(
            sampler,
            str), 'Expected sampler to be a str, but got %s' % type(sampler)
        if sampler == 'random':
            return RandomSampler
        if sampler == 'sequential':
            return SequentialSampler
        raise ValueError(
            'sampler must be either "random" or "sequential", but got %s' %
            (sampler))

    def __call__(self, corpus):
        """Batchify a stream.

        Parameters
        ----------
        corpus : nlp.data.DatasetStream
            A stream of un-flattened CorpusDataset.

        Returns
        -------
        nlp.data.DataStream
            Batches of numericalized samples such that the recurrent states
            from last batch connects with the current batch for each sample.
            Each element of the Dataset is a tuple of data and label arrays for
            BPTT. They are of shape (seq_len, batch_size) respectively.
        """
        return _StreamBPTTBatchify(
            corpus, self._vocab, self._seq_len, self._batch_size,
            self._get_sampler(self._sampler), self._last_batch)


class _StreamBPTTBatchify(DataStream):
    def __init__(self, corpus, vocab, seq_len, batch_size, sampler,
                 last_batch):
        self._corpus = corpus
        self._vocab = vocab
        self._seq_len = seq_len
        self._batch_size = batch_size
        self._sampler = sampler
        self._last_batch = last_batch
        self._padding_idx = vocab[vocab.padding_token]

    def __iter__(self):
        def _init(data, target, value):
            """Init the data and target with values."""
            data[:] = value
            target[:] = value

        def _read(buffers, i, vocab, corpus):
            """Read a sentence from the corpus into i-th buffer."""
            if len(buffers[i]) <= 1:
                buffers[i].extend(vocab[next(corpus)])

        def _write(data, target, buffers, seq_len, i, length):
            """Write a sentence from i-th buffer to data and target."""
            num_tokens = len(buffers[i]) - 1
            num_tokens = min(num_tokens, seq_len - length)
            # fill in data and target
            data[i, length:length+num_tokens] = buffers[i][:num_tokens]
            target[i, length:length+num_tokens] = buffers[i][1:num_tokens+1]
            # trim sentence in the buffer if too long. Used for the next batch
            buffers[i] = buffers[i][num_tokens:]
            return num_tokens

        # stream states
        buffers = [[] for _ in range(self._batch_size)]
        has_next = True
        has_token_buffered = False
        data = np.empty([self._batch_size, self._seq_len], dtype=np.float32)
        target = np.empty([self._batch_size, self._seq_len], dtype=np.float32)
        corpus = itertools.chain.from_iterable(
            (corpus_dataset[idx] for idx in self._sampler(len(corpus_dataset)))
            for corpus_dataset in self._corpus)

        while has_next or has_token_buffered:
            _init(data, target, self._padding_idx)
            has_token_buffered = False
            for i in range(self._batch_size):
                length = 0
                try:
                    while length < self._seq_len:
                        _read(buffers, i, self._vocab, corpus)
                        num_tokens = _write(data, target, buffers, self._seq_len, i, length)
                        if len(buffers[i]) > 0:
                            has_token_buffered = True
                        length += num_tokens
                except StopIteration:
                    has_next = False
            if has_token_buffered or self._last_batch == 'keep':
                yield mx.nd.array(data).T, mx.nd.array(target).T
