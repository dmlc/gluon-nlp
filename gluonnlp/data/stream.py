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
"""NLP Toolkit Data Stream API. It allows easy and customizable streaming of
corpora and dataset files. Files can be streamed into formats that are
ready for training and evaluation."""
__all__ = [
    'DataStream', 'CorpusStream', 'LanguageModelStream', 'SimpleDataStream',
    'StreamDataLoader'
]

import glob
import multiprocessing
import os
import sys
import threading

import mxnet as mx
import mxnet.gluon.data as gdata
from mxnet.gluon.data import RandomSampler, SequentialSampler
import numpy as np

from .dataset import CorpusDataset

try:  # Python 2
    import Queue as queue
    from itertools import izip as zip  # pylint: disable=redefined-builtin
    from itertools import izip_longest as zip_longest
except ImportError:  # Python 3
    from itertools import zip_longest as zip_longest
    import queue

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


def worker_loop(batched_stream, control_queue, data_queue, batchify_fn):
    """Worker loop for multiprocessing DataLoader."""
    batched_stream_iter = iter(batched_stream)
    while True:
        idx, get_next = control_queue.get()
        if not get_next:
            break
        try:
            batch = batchify_fn(next(batched_stream_iter))
        except StopIteration:
            batch = None
        data_queue.put((idx, batch))


class _MultiWorkerIter(object):
    """Interal multi-worker iterator for DataLoader."""

    def __init__(self, prefetch, batched_stream, batchify_fn):
        self._prefetch = prefetch
        self._batched_stream = batched_stream
        self._batchify_fn = batchify_fn
        self._control_queue = gdata.dataloader.Queue()
        self._data_queue = gdata.dataloader.Queue() \
                            if sys.version_info[0] <= 2 \
                            else gdata.dataloader.SimpleQueue()
        self._data_buffer = {}
        self._rcvd_idx = 0
        self._sent_idx = 0
        self._shutdown = False

        worker = multiprocessing.Process(
            target=worker_loop, args=(self._batched_stream,
                                      self._control_queue, self._data_queue,
                                      self._batchify_fn))
        worker.daemon = True
        worker.start()

        # pre-fetch
        for _ in range(self._prefetch):
            self._push_next()

    def __del__(self):
        self.shutdown()

    def _push_next(self):
        """Assign next batch workload to workers."""
        self._control_queue.put((self._sent_idx, True))
        self._sent_idx += 1

    def __next__(self):
        assert not self._shutdown, 'call __next__ after shutdown is forbidden'

        while True:
            self._push_next()
            if self._rcvd_idx in self._data_buffer:
                batch = self._data_buffer.pop(self._rcvd_idx)
                self._rcvd_idx += 1
                return batch
            idx, batch = self._data_queue.get()
            if batch is None and not self._data_buffer:
                self.shutdown()
                raise StopIteration
            self._data_buffer[idx] = batch

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

    def shutdown(self):
        """Shutdown internal workers by pushing terminate signals."""
        if not self._shutdown:
            self._control_queue.put((None, None))
            try:
                while not self._data_queue.empty():
                    self._data_queue.get()
            except IOError:
                pass
            self._shutdown = True


class _ThreadedIter(threading.Thread):
    def __init__(self, prefetch, batched_stream, batchify_fn):
        super(_ThreadedIter, self).__init__()
        self.queue = queue.Queue(prefetch)
        self.batched_stream = batched_stream
        self.batchify_fn = batchify_fn
        self.daemon = True
        self.start()

    def run(self):
        for data in self.batched_stream:
            self.queue.put(self.batchify_fn(data))
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self


class StreamDataLoader(object):
    """Loads data from a stream and returns mini-batches of data.

    Parameters
    ----------
    stream : DataStream
        Source stream.
    batch_size : int
        Size of mini-batch.
    last_batch : {'discard', 'keep'}, default 'keep'
        How to handle the last batch if batch_size does not evenly divide
        `len(dataset)`.
        discard - The last batch is discarded if its incomplete.
        keep - A batch with less samples than previous batches is returned.
    batchify_fn : callable
        Callback function to allow users to specify how to merge samples
        into a batch. Defaults to `default_batchify_fn`::

            def default_batchify_fn(data):
                if isinstance(data[0], nd.NDArray):
                    return nd.stack(*data)
                elif isinstance(data[0], tuple):
                    data = zip(*data)
                    return [default_batchify_fn(i) for i in data]
                else:
                    data = np.asarray(data)
                    return nd.array(data, dtype=data.dtype)

    prefetch : bool, default False
        If `prefetch > 0`, use a separate thread or process for fetching
        `prefetch` elements from the stream. Not supported on Windows yet.
    multiprocessing : bool, default False
        Use a multiprocessing worker for prefetching. Only taken into account
        if `prefetch > 0`.
    """

    def __init__(self, stream, batch_size, last_batch='keep', batchify_fn=None,
                 prefetch=0, use_multiprocessing=False):
        self._stream = stream
        self._batch_size = batch_size
        self._prefetch = prefetch
        self._multiprocessing = use_multiprocessing

        if batchify_fn is None:
            if self._prefetch and self._multiprocessing:
                batchify_fn = gdata.dataloader.default_mp_batchify_fn
            else:
                batchify_fn = gdata.dataloader.default_batchify_fn
        assert last_batch in ('discard', 'keep')
        self._last_batch = last_batch
        if self._last_batch == 'keep':
            self._last_batch_pad = object()

            def unpad_batches(data):
                data = [d for d in data if d is not self._last_batch_pad]
                return batchify_fn(data)

            self._batchify_fn = unpad_batches
        else:
            self._batchify_fn = batchify_fn

    def __iter__(self):
        # Duplicate the stream iterator batch_size times so zip can retrieve
        # batch_size elements
        zip_args = [iter(self._stream)] * self._batch_size
        if self._last_batch == 'discard':
            batch_data_generator = zip(*zip_args)
        else:
            batch_data_generator = zip_longest(*zip_args,
                                               fillvalue=self._last_batch_pad)

        if not self._prefetch:
            return (self._batchify_fn(batch_data)
                    for batch_data in batch_data_generator)
        elif self._multiprocessing:
            return _MultiWorkerIter(self._prefetch, batch_data_generator,
                                    self._batchify_fn)
        else:
            return _ThreadedIter(self._prefetch, batch_data_generator,
                                 self._batchify_fn)
