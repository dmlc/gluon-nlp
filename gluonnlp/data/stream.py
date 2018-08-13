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
__all__ = ['DataStream', 'CorpusStream', 'LanguageModelStream', 'SimpleDataStream',
           'PrefetchingStream']

import glob
import itertools
import multiprocessing
import os
import random
import threading

import numpy as np

import mxnet as mx
from mxnet.gluon.data import RandomSampler, SequentialSampler

from .dataset import CorpusDataset
from .utils import line_splitter, whitespace_splitter

try:
    import Queue as queue
except ImportError:
    import queue

class DataStream(object):
    """Abstract Data Stream Interface.

    DataStreams are useful to avoid loading big datasets to memory. A
    DataStream is a iterable object (it implements the __iter__ function).
    Whenever an iteration over the DataStream is requested (e.g. in a for loop
    or by calling iter(datastream)), a new iterator over all samples in the
    DataStream is returned. DataStreams can be lazily transformed by calling
    `transform()` which returns a DataStream over the transformed samples.

    """

    def __iter__(self):
        """Return an iterator over all elements of the DataStream.

        This method returns a new iterator object that can iterate over
        all the objects in the DataStream.

        Returns
        -------
        iterator
            An object implementing the Python *iterator protocol*.

        """
        raise NotImplementedError

    def transform(self, fn):
        """Transform a DataStream lazily.

        Returns
        -------
        DataStream
            The data stream that lazily transforms the data while streaming.
        """

        return _LazyTransformDataStream(self, fn)

class DatasetStream(DataStream):
    """Abstract Dataset Stream Interface.

    A DatasetStream is a DataStream where each sample is a
    `mxnet.gluon.data.Dataset`. An iteration over a DatasetStream iterates over
    `mxnet.gluon.data.Dataset` objects, representing a chunk or shards of some
    large datasets.

    Iterating over sizeable chunks of a dataset can be helpful to speed up
    preprocessing as the overhead of preprocessing each sample individually is
    reduced (this is similar to the idea of using batches for training a
    model).

    """

    def __iter__(self):
        raise NotImplementedError


class SimpleDataStream(DataStream):
    """SimpleDataStream wraps iterables to expose the DataStream API.

    Unlike the iterable itself, the SimpleDataStream exposes the DataStream API
    and allows lazy transformation of the iterable.

    """
    def __init__(self, iterable):
        self._stream = iterable

    def __iter__(self):
        return iter(self._stream)


class _LazyTransformDataStream(DataStream):
    """Data stream that lazily transforms the data."""
    def __init__(self, stream, fn):
        self._stream = stream
        self._fn = fn

    def __iter__(self):
        stream_iter = iter(self._stream)
        try:
            item = next(stream_iter)
        except StopIteration:
            return
        istuple = isinstance(item, tuple)
        if istuple:
            yield self._fn(*item)
            for item in stream_iter:
                yield self._fn(*item)
        else:
            yield self._fn(item)
            for item in stream_iter:
                yield self._fn(item)


class CorpusStream(DatasetStream):
    """CorpusStream streams a number of CorpusDatasets.

    The CorpusDatasets are created from multiple text files that match provided
    `file_pattern`. One file is read at a time and the corresponding
    CorpusDataset is returned.

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
    file_sampler : str, {'sequential', 'random'}, defaults to 'random'
        The sampler used to sample a file.

        - 'sequential': SequentialSampler
        - 'random': RandomSampler
    """
    def __init__(self, file_pattern, encoding='utf8', flatten=False, skip_empty=True,
                 sample_splitter=line_splitter, tokenizer=whitespace_splitter,
                 bos=None, eos=None, file_sampler='random'):
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
        self._file_sampler = file_sampler

    def _get_sampler(self, sampler):
        assert isinstance(sampler, str), 'Expected sampler to be a str, but got %s'%type(sampler)
        if sampler == 'random':
            return RandomSampler
        if sampler == 'sequential':
            return SequentialSampler
        raise ValueError('sampler must be either "random" or "sequential", but got %s'%(sampler))

    def __iter__(self):
        file_sampler = self._get_sampler(self._file_sampler)
        # generate file samples
        files = sorted(glob.glob(self._file_pattern))
        if len(files) == 0:
            raise ValueError('Cannot find any file with path "%s"'%self._file_pattern)
        for file_idx in iter(file_sampler(len(files))):
            filename = files[file_idx]
            yield CorpusDataset(filename, encoding=self._encoding,
                                flatten=self._flatten, skip_empty=self._skip_empty,
                                sample_splitter=self._sample_splitter,
                                tokenizer=self._tokenizer, bos=self._bos, eos=self._eos)

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
                 sample_splitter=line_splitter, tokenizer=whitespace_splitter,
                 bos=None, eos=None, sampler='random', file_sampler='random'):
        super(LanguageModelStream, self).__init__(file_pattern, flatten=True,
                                                  encoding=encoding,
                                                  skip_empty=skip_empty,
                                                  sample_splitter=sample_splitter,
                                                  tokenizer=tokenizer, bos=bos,
                                                  eos=eos, file_sampler=file_sampler)
        self._sampler = sampler

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

        batch_1.data.T::

            <bos> k l m n
            i j <bos> o <padding>

        batch_1.target.T::

            k l m n <eos>
            j <bos> o <eos> <padding>

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
                              file_sampler=self._file_sampler)
        return _LanguageModelBPTTStream(
            corpus, vocab, seq_len, batch_size, sampler=self._get_sampler(
                self._sampler), last_batch=last_batch)

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
    sampler : mx.gluon.data.Sampler
        The sampler used to sample texts within a file.
    last_batch : {'keep', 'discard'}
        How to handle the last batch if the remaining length is less than `seq_len`.

        - keep: A batch with less samples than previous batches is returned.
        - discard: The last batch is discarded if it's smaller than `(seq_len, batch_size)`.
    """
    def __init__(self, corpus, vocab, seq_len, batch_size, sampler, last_batch='keep'):
        if corpus._flatten:
            raise ValueError('_LanguageModelBPTTStream does not support flatten corpus. '\
                             'Please create a CorpusStream with flatten=False.')
        self._corpus = corpus
        self._vocab = vocab
        self._seq_len = seq_len
        self._batch_size = batch_size
        self._sampler = sampler
        self._last_batch = last_batch
        self._padding_idx = 0
        if last_batch == 'keep':
            assert vocab.padding_token, 'Padding token must be specified in vocab when '\
                                        'last_batch="keep".'
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
        buffers = [None] * self._batch_size
        for i in range(self._batch_size):
            buffers[i] = []
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
        return


class _Prefetcher(object):
    """Internal shared prefetcher logic."""
    data_queue = None
    control_queue = None

    def __init__(self, stream, num_prefetch, seed, np_seed, mx_seed):
        super(_Prefetcher, self).__init__()
        self.stream = stream
        assert num_prefetch > 0, 'Unbounded Prefetcher is unsupported.'
        self.num_prefetch = num_prefetch
        self.seed = seed
        self.np_seed = np_seed
        self.mx_seed = mx_seed

    def run(self):
        """Method representing the process’s activity."""
        random.seed(self.seed)
        np.random.seed(self.np_seed)
        mx.random.seed(self.mx_seed)

        stream_iter = iter(self.stream)
        while True:
            try:  # Check control queue
                c = self.control_queue.get(False)
                if c is None:
                    break
            except queue.Empty:
                pass

            try:
                data = next(stream_iter)
                self.data_queue.put(data)
            except StopIteration:
                self.data_queue.put(None)

    def __next__(self):
        next_item = self.data_queue.get()
        if next_item is None:
            self.control_queue.put(None)
            raise StopIteration
        return next_item

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self


class _ProcessPrefetcher(_Prefetcher, multiprocessing.Process):
    """Internal multi-processing prefetcher."""

    def __init__(self, *args, **kwargs):
        super(_ProcessPrefetcher, self).__init__(*args, **kwargs)
        self.data_queue = multiprocessing.Queue(self.num_prefetch)
        self.control_queue = multiprocessing.Queue()
        self.daemon = True
        self.start()


class _ThreadPrefetcher(_Prefetcher, threading.Thread):
    """Internal threaded prefetcher."""

    def __init__(self, *args, **kwargs):
        super(_ThreadPrefetcher, self).__init__(*args, **kwargs)
        self.data_queue = queue.Queue(self.num_prefetch)
        self.control_queue = queue.Queue()
        self.daemon = True
        self.start()


class PrefetchingStream(DataStream):
    """Prefetch a DataStream in a separate Thread or Process.

    This iterator will create another thread or process to perform
    ``iter_next`` and then store the data in memory. It potentially accelerates
    the data read, at the cost of more memory usage.

    The python, numpy and mxnet random states in the launched Thread or Process
    will be initialized randomly based on the next 32 bit integer in the
    python, numpy and mxnet random generator of the caller respectively
    (random.getrandbits(32), numpy.random.randint(0, 2**32),
    int(mx.nd.random.uniform(0, 2**32).asscalar())).

    Parameters
    ----------
    stream : DataStream
        Source stream.
    num_prefetch : int, default 1
        Number of elements to prefetch from the stream. Must be greater 0.
    worker_type : 'thread' or 'process', default 'thread'
        Use a separate Python Thread or Process to prefetch.
    """

    def __init__(self, stream, num_prefetch=1, worker_type='thread'):
        self._stream = stream
        self._num_prefetch = num_prefetch
        if num_prefetch < 1:
            raise ValueError('num_prefetch must be greater 0.')
        assert worker_type.lower() in ['thread', 'process']
        self._multiprocessing = worker_type.lower() == 'process'

    def __iter__(self):
        seed = random.getrandbits(32)
        np_seed = np.random.randint(0, 2**32)
        mx_seed = int(mx.nd.random.uniform(0, 2**32).asscalar())
        if self._multiprocessing:
            return _ProcessPrefetcher(self._stream, self._num_prefetch,
                                      seed=seed, np_seed=np_seed,
                                      mx_seed=mx_seed)
        else:
            return _ThreadPrefetcher(self._stream, self._num_prefetch,
                                     seed=seed, np_seed=np_seed,
                                     mx_seed=mx_seed)
