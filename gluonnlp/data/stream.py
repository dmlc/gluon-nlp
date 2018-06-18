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
"""NLP Toolkit Data Stream API. It allows easy and customizable streaming of
corpora and dataset files. Files can be streamed into formats that are
ready for training and evaluation."""
__all__ = ['DataStream', 'CorpusStream', 'LanguageModelStream', 'SimpleDataStream']

import os
import glob
import numpy as np

import mxnet as mx
from mxnet.gluon.data import RandomSampler, SequentialSampler
from .dataset import CorpusDataset

class DataStream(object):
    """Abstract Data Stream Interface."""
    def __iter__(self):
        raise NotImplementedError

    def transform(self, fn):
        """
        Returns
        -------
        DataStream
            The data stream that lazily transforms the data while streaming.
        """
        return _LazyTransformDataStream(self, fn)

class SimpleDataStream(DataStream):
    """Simple DataStream wrapper for a stream."""
    def __init__(self, stream):
        self._stream = stream

    def __iter__(self):
        return iter(self._stream)

class _LazyTransformDataStream(DataStream):
    """Data stream that lazily transforms the data."""
    def __init__(self, stream, fn):
        self._stream = stream
        self._fn = fn

    def __iter__(self):
        for item in iter(self._stream):
            yield self._fn(item)

class CorpusStream(DataStream):
    """Common text data stream that streams a corpus consisting of multiple text files
    that match provided `file_pattern`. One file is read at a time.

    The returned dataset includes samples, each of which can either be a list of tokens if tokenizer
    is specified, or otherwise a single string segment produced by the `sample_splitter`.

    Parameters
    ----------
    file_pattern: str
        Path to the input text files.
    encoding : str, default 'utf8'
        File encoding format.
    flatten : bool, default False
        Whether to return all samples as flattened tokens. If True, each sample is a token.
    skip_empty : bool, default True
        Whether to skip the empty samples produced from sample_splitters. If False, `bos` and `eos`
        will be added in empty samples.
    sample_splitter : function, default str.splitlines
        A function that splits the dataset string into samples.
    tokenizer : function or None, default str.split
        A function that splits each sample string into list of tokens. If None, raw samples are
        returned according to `sample_splitter`.
    bos : str or None, default None
        The token to add at the begining of each sequence. If None, or if tokenizer is not
        specified, then nothing is added.
    eos : str or None, default None
        The token to add at the end of each sequence. If None, or if tokenizer is not
        specified, then nothing is added.
    sampler : str, {'sequential', 'random'}, defaults to 'random'
        The sampler used to sample texts within a file.

        - 'sequential': SequentialSampler
        - 'random': RandomSampler
    file_sampler : str, {'sequential', 'random'}, defaults to 'random'
        The sampler used to sample a file.

        - 'sequential': SequentialSampler
        - 'random': RandomSampler
    """
    def __init__(self, file_pattern, encoding='utf8', flatten=False, skip_empty=True,
                 sample_splitter=lambda s: s.splitlines(), tokenizer=lambda s: s.split(),
                 bos=None, eos=None, sampler='random', file_sampler='random'):
        assert sample_splitter, 'sample_splitter must be specified.'
        if not isinstance(file_pattern, str):
            raise TypeError('file_pattern must be str, but got %s'%type(file_pattern))
        self._file_pattern = os.path.expanduser(file_pattern)
        self._encoding = encoding
        self._flatten = flatten
        self._skip_empty = skip_empty
        self._sample_splitter = sample_splitter
        self._tokenizer = tokenizer
        self._bos = bos
        self._eos = eos
        self._sampler = sampler
        self._file_sampler = file_sampler

    def _get_sampler(self, sampler):
        assert isinstance(sampler, str), 'Expected sampler to be a str, but got %s'%type(sampler)
        if sampler == 'random':
            return RandomSampler
        if sampler == 'sequential':
            return SequentialSampler
        raise ValueError('sampler must be either "random" or "sequential", but got %s'%(sampler))

    def __iter__(self):
        sampler = self._get_sampler(self._sampler)
        file_sampler = self._get_sampler(self._file_sampler)
        # generate file samples
        files = sorted(glob.glob(self._file_pattern))
        if len(files) == 0:
            raise ValueError('Cannot find any file with path "%s"'%self._file_pattern)
        for file_idx in iter(file_sampler(len(files))):
            filename = files[file_idx]
            corpus = CorpusDataset(filename, encoding=self._encoding,
                                   flatten=self._flatten, skip_empty=self._skip_empty,
                                   sample_splitter=self._sample_splitter,
                                   tokenizer=self._tokenizer, bos=self._bos, eos=self._eos)
            # generate samples
            num_samples = len(corpus)
            for idx in iter(sampler(num_samples)):
                yield corpus[idx]

class LanguageModelStream(CorpusStream):
    """Streams a corpus consisting of multiple text files that match provided
    `file_pattern`, and produces a language modeling stream given the provided
    sample splitter and word tokenizer.

    Parameters
    ----------
    file_pattern: str
        Path to the input text files.
    encoding : str, default 'utf8'
        File encoding format.
    skip_empty : bool, default True
        Whether to skip the empty samples produced from sample_splitters. If False, `bos` and `eos`
        will be added in empty samples.
    sample_splitter : function, default str.splitlines
        A function that splits the dataset string into samples.
    tokenizer : function or None, default str.split
        A function that splits each sample string into list of tokens. If None, raw samples are
        returned according to `sample_splitter`.
    bos : str or None, default None
        The token to add at the begining of each sequence. If None, or if tokenizer is not
        specified, then nothing is added.
    eos : str or None, default None
        The token to add at the end of each sequence. If None, or if tokenizer is not
        specified, then nothing is added.
    sampler : str, {'sequential', 'random'}, defaults to 'random'
        The sampler used to sample texts within a file.

        - 'sequential': SequentialSampler
        - 'random': RandomSampler
    file_sampler : str, {'sequential', 'random'}, defaults to 'random'
        The sampler used to sample a file.

        - 'sequential': SequentialSampler
        - 'random': RandomSampler
    """
    def __init__(self, file_pattern, encoding='utf8', skip_empty=True,
                 sample_splitter=lambda s: s.splitlines(), tokenizer=lambda s: s.split(),
                 bos=None, eos=None, sampler='random', file_sampler='random'):
        self._file_pattern = file_pattern
        self._encoding = encoding
        self._skip_empty = skip_empty
        self._sample_splitter = sample_splitter
        self._tokenizer = tokenizer
        self._bos = bos
        self._eos = eos
        self._sampler = sampler
        self._file_sampler = file_sampler
        super(LanguageModelStream, self).__init__(self._file_pattern, flatten=True,
                                                  encoding=self._encoding,
                                                  skip_empty=self._skip_empty,
                                                  sample_splitter=self._sample_splitter,
                                                  tokenizer=self._tokenizer, bos=self._bos,
                                                  eos=self._eos, sampler=self._sampler,
                                                  file_sampler=self._file_sampler)

    def bptt_batchify(self, vocab, seq_len, batch_size, last_batch='keep'):
        """The corpus is transformed into batches of numericalized samples, in the way that the
        recurrent states from last batch connects with the current batch for each sample.

        Each sample is of shape `(seq_len, batch_size)`.

        For example, the following 4 sequences::

            <bos> a b c d <eos>
            <bos> e f g h i j <eos>
            <bos> k l m n <eos>
            <bos> o <eos>

        will generate 2 batches with seq_len = 5, batch_size = 2 as follow (transposed):

        batch_0.data.T::

            <bos> a b c d
            <bos> e f g h

        batch_0.target.T::

            a b c d <eos>
            e f g h i

        batch_0.mask.T::

            1 1 1 1 1
            1 1 1 1 1

        batch_1.data.T::

            <bos> k l m n
            i j <bos> o <padding>

        batch_1.target.T::

            k l m n <eos>
            j <bos> o <eos> <padding>

        batch_1.mask.T::

            1 1 1 1 1
            1 1 1 1 0

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

            - keep: A batch with less samples than previous batches is returned.
            - discard: The last batch is discarded if it's smaller than `(seq_len, batch_size)`.
        """
        corpus = CorpusStream(self._file_pattern, flatten=False, encoding=self._encoding,
                              skip_empty=self._skip_empty, sample_splitter=self._sample_splitter,
                              tokenizer=self._tokenizer, bos=self._bos, eos=self._eos,
                              sampler=self._sampler, file_sampler=self._file_sampler)
        return _LanguageModelBPTTStream(corpus, vocab, seq_len, batch_size, last_batch=last_batch)

class _LanguageModelBPTTStream(DataStream):
    """Streams a corpus and produces a language modeling data stream.

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

        - keep: A batch with less samples than previous batches is returned.
        - discard: The last batch is discarded if it's smaller than `(seq_len, batch_size)`.
    """
    def __init__(self, corpus, vocab, seq_len, batch_size, last_batch='keep'):
        if corpus._flatten:
            raise ValueError('_LanguageModelBPTTStream does not support flatten corpus. '\
                             'Please create a CorpusStream with flatten=False.')
        self._corpus = corpus
        self._vocab = vocab
        self._seq_len = seq_len
        self._batch_size = batch_size
        self._last_batch = last_batch
        self._padding_idx = 0
        if last_batch == 'keep':
            assert vocab.padding_token, 'Padding token must be specified in vocab when '\
                                        'last_batch="keep".'
            self._padding_idx = vocab[vocab.padding_token]

    def __iter__(self):
        def _init(data, target, mask, value):
            """Init the data and target with values."""
            data[:] = value
            target[:] = value
            mask[:] = 0

        def _read(buffers, i, vocab, corpus):
            """Read a sentence from the corpus into i-th buffer."""
            if buffers[i] is None:
                buffers[i] = vocab[next(corpus)]
            if len(buffers[i]) <= 1:
                buffers[i].extend(vocab[next(corpus)])

        def _write(data, target, buffers, seq_len, i, length):
            """Write a sentence from i-th buffer to data and target."""
            num_tokens = len(buffers[i]) - 1
            num_tokens = min(num_tokens, seq_len - length)
            # fill in data and target
            data[i, length:length+num_tokens] = buffers[i][:num_tokens]
            target[i, length:length+num_tokens] = buffers[i][1:num_tokens+1]
            mask[i, length:length+num_tokens] = 1
            # trim sentence in the buffer if too long. Used for the next batch
            buffers[i] = buffers[i][num_tokens:]
            return num_tokens

        # stream states
        buffers = [None] * self._batch_size
        has_next = True
        has_token_buffered = False
        data = np.empty([self._batch_size, self._seq_len], dtype=np.float32)
        target = np.empty([self._batch_size, self._seq_len], dtype=np.float32)
        mask = np.empty([self._batch_size, self._seq_len], dtype=np.float32)
        corpus = iter(self._corpus)

        while has_next or has_token_buffered:
            _init(data, target, mask, self._padding_idx)
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
                yield mx.nd.array(data).T, mx.nd.array(target).T, mx.nd.array(mask).T
        return
