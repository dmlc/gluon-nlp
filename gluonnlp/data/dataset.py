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
"""NLP Toolkit Dataset API. It allows easy and customizable loading of corpora and dataset files.
Files can be loaded into formats that are immediately ready for training and evaluation."""
__all__ = ['TextLineDataset', 'CorpusDataset', 'LanguageModelDataset',\
           'CorpusIter', 'LanguageModelIter']

import io
import os
import glob
import numpy as np

import mxnet as mx
from mxnet.gluon.data import SimpleDataset, RandomSampler, SequentialSampler
from .utils import concat_sequence, slice_sequence, _slice_pad_length


class TextLineDataset(SimpleDataset):
    """Dataset that comprises lines in a file. Each line will be stripped.

    Parameters
    ----------
    filename : str
        Path to the input text file.
    encoding : str, default 'utf8'
        File encoding format.
    """
    def __init__(self, filename, encoding='utf8'):
        lines = []
        with io.open(filename, 'r', encoding=encoding) as in_file:
            for line in in_file:
                lines.append(line.strip())
        super(TextLineDataset, self).__init__(lines)


class CorpusDataset(SimpleDataset):
    """Common text dataset that reads a whole corpus based on provided sample splitter
    and word tokenizer.

    The returned dataset includes samples, each of which can either be a list of tokens if tokenizer
    is specified, or otherwise a single string segment produced by the sample_splitter.

    Parameters
    ----------
    filename : str or list of str
        Path to the input text file or list of paths to the input text files.
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
    """
    def __init__(self, filename, encoding='utf8', flatten=False, skip_empty=True,
                 sample_splitter=lambda s: s.splitlines(), tokenizer=lambda s: s.split(),
                 bos=None, eos=None):
        assert sample_splitter, 'sample_splitter must be specified.'

        if not isinstance(filename, (tuple, list)):
            filename = (filename, )

        self._filenames = [os.path.expanduser(f) for f in filename]
        self._encoding = encoding
        self._flatten = flatten
        self._skip_empty = skip_empty
        self._sample_splitter = sample_splitter
        self._tokenizer = tokenizer
        def process(s):
            tokens = [bos] if bos else []
            tokens.extend(s)
            if eos:
                tokens.append(eos)
            return tokens
        self._process = process
        super(CorpusDataset, self).__init__(self._read())

    def _read(self):
        all_samples = []
        for filename in self._filenames:
            with io.open(filename, 'r', encoding=self._encoding) as fin:
                content = fin.read()
            samples = (s.strip() for s in self._sample_splitter(content))
            if self._tokenizer:
                samples = [self._process(self._tokenizer(s)) for s in samples
                           if s or not self._skip_empty]
                if self._flatten:
                    samples = concat_sequence(samples)
            elif self._skip_empty:
                samples = [s for s in samples if s]

            all_samples += samples
        return all_samples


class LanguageModelDataset(CorpusDataset):
    """Reads a whole corpus and produces a language modeling dataset given the provided
    sample splitter and word tokenizer.

    Parameters
    ----------
    filename : str
        Path to the input text file.
    encoding : str, default 'utf8'
        File encoding format.
    skip_empty : bool, default True
        Whether to skip the empty samples produced from sample_splitters. If False, `bos` and `eos`
        will be added in empty samples.
    sample_splitter : function, default str.splitlines
        A function that splits the dataset string into samples.
    tokenizer : function, default str.split
        A function that splits each sample string into list of tokens.
    bos : str or None, default None
        The token to add at the begining of each sentence. If None, nothing is added.
    eos : str or None, default None
        The token to add at the end of each sentence. If None, nothing is added.
    """
    def __init__(self, filename, encoding='utf8', skip_empty=True,
                 sample_splitter=lambda s: s.splitlines(),
                 tokenizer=lambda s: s.split(), bos=None, eos=None):
        assert tokenizer, 'Tokenizer must be specified for reading language model corpus.'
        super(LanguageModelDataset, self).__init__(filename, encoding, True, skip_empty,
                                                   sample_splitter, tokenizer, bos, eos)

    def _read(self):
        return [super(LanguageModelDataset, self)._read()]

    def batchify(self, vocab, batch_size):
        """Transform the dataset into N independent sequences, where N is the batch size.

        Parameters
        ----------
        vocab : gluonnlp.Vocab
            The vocabulary to use for numericalizing the dataset. Each token will be mapped to the
            index according to the vocabulary.
        batch_size : int
            The number of samples in each batch.

        Returns
        -------
        NDArray of shape (num_tokens // N, N). Excessive tokens that don't align along
        the batches are discarded.
        """
        data = self._data[0]
        sample_len = len(data) // batch_size
        return mx.nd.array(vocab[data[:sample_len*batch_size]]).reshape(batch_size, -1).T

    def bptt_batchify(self, vocab, seq_len, batch_size, last_batch='keep'):
        """Transform the dataset into batches of numericalized samples, in the way that the
        recurrent states from last batch connects with the current batch for each sample.

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

            keep - A batch with less samples than previous batches is returned.
            discard - The last batch is discarded if its incomplete.
        """
        data = self.batchify(vocab, batch_size)
        batches = slice_sequence(data, seq_len+1, overlap=1)
        if last_batch == 'keep':
            sample_len = len(self._data[0]) // batch_size
            has_short_batch = _slice_pad_length(sample_len*batch_size, seq_len+1, 1) > 0
            if has_short_batch:
                batches.append(data[seq_len*len(batches):, :])
        return SimpleDataset(batches).transform(lambda x: (x[:min(len(x)-1, seq_len), :], x[1:, :]))

class DataIter(object):
    """Abstract Data Iterator Interface."""
    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def reset(self):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError

class CorpusIter(DataIter):
    """Common text data iterator that streams a corpus consisting of multiple text files
    that match provided file_pattern. One file is read at a time.

    The returned dataset includes samples, each of which can either be a list of tokens if tokenizer
    is specified, or otherwise a single string segment produced by the sample_splitter.

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

        self._file_pattern = glob.glob(os.path.expanduser(file_pattern))
        self._encoding = encoding
        self._flatten = flatten
        self._skip_empty = skip_empty
        self._sample_splitter = sample_splitter
        self._tokenizer = tokenizer
        self._bos = bos
        self._eos = eos
        self._sampler = self._get_sampler(sampler)
        self._file_sampler = self._get_sampler(file_sampler)

        # iterator states
        self._idx = None
        self._file_idx = None
        self._idx_samples = None
        self._file_idx_samples = None
        self._sample = None
        self._corpus = None

    def _get_sampler(self, sampler):
        assert isinstance(sampler, str), 'Expected sampler to be a str, but got %s'%type(sampler)
        if sampler == 'random':
            return RandomSampler
        if sampler == 'sequential':
            return SequentialSampler
        raise ValueError('sampler must be either "random" or "sequential", but got %s'%(sampler))

    def _reset_idx(self):
        self._idx_samples = None
        self._idx = None
        self._corpus = None

    def _reset_file_idx(self):
        self._file_idx_samples = None
        self._file_idx = None
        self._corpus = None

    def _read_corpus(self):
        assert self._corpus is None
        num_files = len(self._file_pattern)
        # generate samples
        if self._file_idx_samples is None:
            self._file_idx_samples = [i for i in iter(self._file_sampler(num_files))]
            self._file_idx = 0
        # no more files to read
        if self._file_idx >= num_files:
            raise StopIteration
        # next file
        file_idx = self._file_idx_samples[self._file_idx]
        filename = self._file_pattern[file_idx]
        self._corpus = CorpusDataset(filename, encoding=self._encoding,
                                     flatten=self._flatten, skip_empty=self._skip_empty,
                                     sample_splitter=self._sample_splitter,
                                     tokenizer=self._tokenizer,
                                     bos=self._bos, eos=self._eos)
        self._file_idx += 1

    def _read_sample(self):
        assert self._corpus
        num_samples = len(self._corpus)
        # generate samples
        if self._idx_samples is None:
            self._idx_samples = [i for i in iter(self._sampler(num_samples))]
            self._idx = 0
        # no more samples for the current file
        if self._idx >= num_samples:
            self._reset_idx()
            raise StopIteration
        # next sample
        idx = self._idx_samples[self._idx]
        self._sample = self._corpus[idx]
        self._idx += 1

    def reset(self):
        self._reset_file_idx()
        self._reset_idx()
        self._sample = None

    def next(self):
        while True:
            if self._corpus is None:
                self._read_corpus()
            try:
                self._read_sample()
                return self._sample
            except StopIteration:
                # try to open a new file
                pass

class LanguageModelIter(DataIter):
    """Streams a corpus and produces a language modeling data iterable.

    The corpus is transformedinto batches of numericalized samples, in the way that the
    recurrent states from last batch connects with the current batch for each sample.

    Each sample is of shape `(seq_len, batch_size)`.

    Parameters
    ----------
    corpus: CorpusIter
        The corpus to stream.
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
            raise ValueError('LanguageModelIter does not support flatten corpus. '\
                             'Please create a CorpusIter with flatten=False.')
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

        # iterator states
        self._buffers = [None] * self._batch_size
        self._has_next = None
        self._has_token_buffered = None
        self._data = np.empty([self._batch_size, self._seq_len], dtype=np.float32)
        self._target = np.empty([self._batch_size, self._seq_len], dtype=np.float32)
        self.reset()

    def _read(self, i):
        """Read a sentence from the corpus into i-th buffer."""
        if self._buffers[i] is None:
            self._buffers[i] = self._vocab[next(self._corpus)]
        if len(self._buffers[i]) <= 1:
            self._buffers[i].extend(self._vocab[next(self._corpus)])

    def _write(self, i, length):
        """Write a sentence from i-th buffer to data and target."""
        num_tokens = len(self._buffers[i]) - 1
        num_tokens = min(num_tokens, self._seq_len - length)
        # fill in data and target
        self._data[i, length:length+num_tokens] = self._buffers[i][:num_tokens]
        self._target[i, length:length+num_tokens] = self._buffers[i][1:num_tokens+1]
        # trim sentence in the buffer if too long. Used for the next batch
        self._buffers[i] = self._buffers[i][num_tokens:]
        return num_tokens

    def _init(self):
        """Initialize the data and target with padding indices."""
        self._data[:] = self._padding_idx
        self._target[:] = self._padding_idx

    def reset(self):
        """Reset iterator states."""
        self._corpus.reset()
        self._buffers = [None] * self._batch_size
        self._has_next = True
        self._has_token_buffered = False
        self._init()

    def next(self):
        # No more sentences
        if not self._has_next and not self._has_token_buffered:
            raise StopIteration

        self._init()
        self._has_token_buffered = False
        for i in range(self._batch_size):
            length = 0
            try:
                while length < self._seq_len:
                    self._read(i)
                    num_tokens = self._write(i, length)
                    if len(self._buffers[i]) > 0:
                        self._has_token_buffered = True
                    length += num_tokens
            except StopIteration:
                self._has_next = False
        if self._has_token_buffered or self._last_batch == 'keep':
            return mx.nd.array(self._data).T, mx.nd.array(self._target).T
        else:
            raise StopIteration
