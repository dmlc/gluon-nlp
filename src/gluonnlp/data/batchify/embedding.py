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
"""Batchify helpers for embedding training."""

__all__ = ['EmbeddingCenterContextBatchify']

import logging
import random

import numpy as np

from ...base import _str_types
from ..stream import DataStream

try:
    from numba import njit, prange
    numba_njit = njit(nogil=True)
except ImportError:
    # Define numba shims
    prange = range

    def numba_njit(func):
        return func


class EmbeddingCenterContextBatchify(object):
    """Helper to create batches of center and contexts words.

    Batches are created lazily on a optionally shuffled version of the Dataset.
    To create batches from some corpus, first create a
    EmbeddingCenterContextBatchify object and then call it with the corpus.
    Please see the documentation of __call__ for more details.

    Parameters
    ----------
    batch_size : int
        Maximum size of batches returned. Actual batch returned can be smaller
        when running out of samples.
    window_size : int, default 5
        The maximum number of context elements to consider left and right of
        each center element. Less elements may be considered if there are not
        sufficient elements left / right of the center element or if a reduced
        window size was drawn.
    reduce_window_size_randomly : bool, default True
       If True, randomly draw a reduced window size for every center element
       uniformly from [1, window].
    shuffle : bool, default True
       If True, shuffle the sentences before lazily generating batches.
    cbow : bool, default False
       Enable CBOW mode. In CBOW mode the returned context contains multiple
       entries per row. One for each context. If CBOW is False (default), there
       is a separate row for each context. The context_data array always
       contains weights for the context words equal to 1 over the number of
       context words in the given row of the context array.
    weight_dtype : numpy.dtype, default numpy.float32
        Data type for data array of sparse COO context representation.
    index_dtype : numpy.dtype, default numpy.int64

    """

    def __init__(self, batch_size, window_size=5,
                 reduce_window_size_randomly=True, shuffle=True, cbow=False,
                 weight_dtype='float32', index_dtype='int64'):
        self._batch_size = batch_size
        self._window_size = window_size
        self._reduce_window_size_randomly = reduce_window_size_randomly
        self._shuffle = shuffle
        self._cbow = cbow
        self._weight_dtype = weight_dtype
        self._index_dtype = index_dtype

    def __call__(self, corpus):
        """Batchify a dataset.

        Parameters
        ----------
        corpus : list of sentences
            List of sentences. Any list containing for example integers or
            strings can be a sentence. Context samples do not cross sentence
            boundaries.

         Returns
         -------
         DataStream
             Each element of the DataStream is a tuple of 2 elements (center,
             context). center is a numpy.ndarray of shape (batch_size, ).
             context is a tuple of 3 numpy.ndarray, representing a sparse COO
             array (data, row, col). The center and context arrays contain the
             center and correpsonding context words respectively. A sparse
             representation is used for context as the number of context words
             for one center word varies based on the randomly chosen context
             window size and sentence boundaries. The returned center and col
             arrays are of the same dtype as the sentence elements.

        """
        return _EmbeddingCenterContextBatchify(
            corpus, self._batch_size, self._window_size,
            self._reduce_window_size_randomly, self._shuffle, cbow=self._cbow,
            weight_dtype=self._weight_dtype, index_dtype=self._index_dtype)


class _EmbeddingCenterContextBatchify(DataStream):
    def __init__(self, sentences, batch_size, window_size,
                 reduce_window_size_randomly, shuffle, cbow, weight_dtype,
                 index_dtype):
        self._sentences = sentences
        self._batch_size = batch_size
        self._window_size = window_size
        self._reduce_window_size_randomly = reduce_window_size_randomly
        self._shuffle = shuffle
        self._cbow = cbow
        self._weight_dtype = weight_dtype
        self._index_dtype = index_dtype

    def __iter__(self):
        if prange is range:
            logging.warning(
                'EmbeddingCenterContextBatchify supports just in time compilation '
                'with numba, but numba is not installed. '
                'Consider "pip install numba" for significant speed-ups.')

        if isinstance(self._sentences[0][0], _str_types):
            sentences = [np.asarray(s, dtype='O') for s in self._sentences]
        else:
            sentences = [np.asarray(s) for s in self._sentences]

        if self._shuffle:
            random.shuffle(sentences)

        sentence_boundaries = np.cumsum([len(c) for c in sentences])
        sentences = np.concatenate(sentences)

        it = iter(
            _context_generator(
                sentence_boundaries, self._window_size, self._batch_size,
                random_window_size=self._reduce_window_size_randomly,
                cbow=self._cbow, seed=random.getrandbits(32)))

        def _closure():
            while True:
                try:
                    (center, context_data, context_row, context_col) = next(it)
                    context_data = np.asarray(context_data, dtype=self._weight_dtype)
                    context_row = np.asarray(context_row, dtype=self._index_dtype)
                    context_col = sentences[context_col]
                    context_coo = (context_data, context_row, context_col)
                    yield sentences[center], context_coo
                except StopIteration:
                    return

        return _closure()


@numba_njit
def _get_sentence_start_end(sentence_boundaries, sentence_pointer):
    end = sentence_boundaries[sentence_pointer]
    if sentence_pointer == 0:
        start = 0
    else:
        start = sentence_boundaries[sentence_pointer - 1]
    return start, end


@numba_njit
def _context_generator(sentence_boundaries, window, batch_size,
                       random_window_size, cbow, seed):
    num_rows = batch_size
    word_pointer = 0
    num_context_skip = 0
    while True:
        center_batch = []
        # Prepare arrays for COO sparse matrix format
        context_data = []
        context_row = []
        context_col = []
        i = 0
        while i < num_rows:
            if word_pointer >= sentence_boundaries[-1]:
                # There is no data left
                break

            contexts = _get_context(word_pointer, sentence_boundaries, window,
                                    random_window_size, seed)
            if contexts is None:
                word_pointer += 1
                continue
            center = word_pointer
            for j, context in enumerate(contexts):
                if num_context_skip > j:
                    # In SkipGram mode, there may be some leftover contexts
                    # form the last batch
                    continue
                elif i < num_rows:
                    num_context_skip = 0
                    context_row.append(i)
                    context_col.append(context)
                    if cbow:
                        context_data.append(1.0 / len(contexts))
                    else:
                        center_batch.append(center)
                        context_data.append(1)
                        i += 1
                else:
                    num_context_skip = j
                    assert not cbow
                    break

            if cbow:
                center_batch.append(center)
                i += 1

            if num_context_skip == 0:
                word_pointer += 1
            else:
                assert i == num_rows
                break

        if len(center_batch) == num_rows:
            center_batch_np = np.array(center_batch, dtype=np.int64)
            context_data_np = np.array(context_data, dtype=np.float32)
            context_row_np = np.array(context_row, dtype=np.int64)
            context_col_np = np.array(context_col, dtype=np.int64)
            yield center_batch_np, context_data_np, context_row_np, context_col_np
        else:
            assert word_pointer >= sentence_boundaries[-1]
            break


@numba_njit
def _get_context(center_idx, sentence_boundaries, window_size,
                 random_window_size, seed):
    """Compute the context with respect to a center word in a sentence.

    Takes an numpy array of sentences boundaries.

    """
    random.seed(seed + center_idx)

    sentence_index = np.searchsorted(sentence_boundaries, center_idx)
    sentence_start, sentence_end = _get_sentence_start_end(
        sentence_boundaries, sentence_index)

    if random_window_size:
        window_size = random.randint(1, window_size)
    start_idx = max(sentence_start, center_idx - window_size)
    end_idx = min(sentence_end, center_idx + window_size + 1)

    if start_idx != center_idx and center_idx + 1 != end_idx:
        context = np.concatenate((np.arange(start_idx, center_idx),
                                  np.arange(center_idx + 1, end_idx)))
    elif start_idx != center_idx:
        context = np.arange(start_idx, center_idx)
    elif center_idx + 1 != end_idx:
        context = np.arange(center_idx + 1, end_idx)
    else:
        context = None

    return context
