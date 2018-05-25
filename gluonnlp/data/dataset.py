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
           'StreamingCorpus', 'StreamingLanguageModel']

import io, os, glob

import mxnet as mx
import numpy as np
from mxnet.gluon.data import SimpleDataset
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

class StreamingCorpus(object):
    """Common text dataset that streams a corpus consisting of multiple text files
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
    """

    def __init__(self, file_pattern, encoding='utf8', flatten=False, skip_empty=True,
                 sample_splitter=lambda s: s.splitlines(), tokenizer=lambda s: s.split(),
                 bos=None, eos=None):
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

    def __iter__(self):
        for filename in self._next_filename():
            corpus = CorpusDataset(filename, encoding=self._encoding,
                                   flatten=self._flatten, skip_empty=self._skip_empty,
                                   sample_splitter=self._sample_splitter,
                                   tokenizer=self._tokenizer,
                                   bos=self._bos, eos=self._eos)
            for token in corpus:
                yield token

    def _next_filename(self):
        file_patterns = glob.glob(self._file_pattern)
        for filename in file_patterns:
            yield filename


class StreamingLanguageModel(object):
    """Streams a corpus consisting of multiple text files that match provided
    file_pattern, and produces a language modeling dataset given the provided
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
    tokenizer : function, default str.split
        A function that splits each sample string into list of tokens.
    bos : str or None, default None
        The token to add at the begining of each sentence. If None, nothing is added.
    eos : str or None, default None
        The token to add at the end of each sentence. If None, nothing is added.
    """
    def __init__(self, file_pattern, encoding='utf8', skip_empty=True,
               sample_splitter=lambda s: s.splitlines(), tokenizer=lambda s: s.split(),
               bos=None, eos=None):
        # read one sentence at a time for streaming
        flatten = False
        self._corpus = StreamingCorpus(file_pattern, encoding=encoding, flatten=False,
                                       skip_empty=skip_empty,
                                       sample_splitter=sample_splitter, tokenizer=tokenizer,
                                       bos=bos, eos=eos)
        pass

    def bptt_batchify(self, vocab, seq_len, batch_size, last_batch='keep'):
        """Transform the dataset into batches of numericalized samples, in the way that the
        recurrent states from last batch connects with the current batch for each sample.

        Each sample is of shape `(seq_len, batch_size)`.

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
        padding_val = 0
        if last_batch == 'keep':
            assert vocab.padding_token, 'Padding token must be specified in vocab when '\
                                        'last_batch="keep".'
            padding_val = vocab[vocab.padding_token]
        # buffers for the data and labels before converted to NDArrays
        data = np.full([batch_size, seq_len], padding_val, dtype=np.float32)
        label = np.full([batch_size, seq_len], padding_val, dtype=np.float32)
        corpus = iter(self._corpus)
        # buffers for the next sentence
        sentences = [None] * batch_size
        has_next = True
        while has_next:
            data[:] = padding_val
            label[:] = padding_val
            for i in range(batch_size):
                start = 0
                try:
                    while start < seq_len:
                        if sentences[i] is None or len(sentences[i]) <= 1:
                            sentences[i] = vocab[next(corpus)]
                        num_tokens = len(sentences[i]) - 1
                        num_tokens = min(num_tokens, seq_len - start)
                        end = start + num_tokens
                        # fill in data and label
                        data[i, start:end] = sentences[i][:num_tokens]
                        label[i, start:end] = sentences[i][1:num_tokens+1]
                        # trim sentence for the next batch
                        sentences[i] = sentences[i][num_tokens:]
                        start += num_tokens
                except StopIteration:
                    has_next = False
            if has_next or last_batch == 'keep':
                yield mx.nd.array(data).T, mx.nd.array(label).T
        return
