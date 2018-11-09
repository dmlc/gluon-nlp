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

import random

from mxnet import nd
import numpy as np

from ..stream import DataStream

from libc.math cimport floor
from libc.stdint cimport int8_t, int64_t, uint32_t, uint64_t
from libcpp.algorithm cimport binary_search
from libcpp cimport bool
from libcpp.utility cimport pair
from libcpp.vector cimport vector

cdef extern from "<random>" namespace "std" nogil:
    cdef cppclass mt19937:
        mt19937()
        mt19937(unsigned int seed)

    cdef cppclass uniform_int_distribution[T]:
        uniform_int_distribution()
        uniform_int_distribution(T a, T b)
        T operator()(mt19937 gen)


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
    dtype : numpy.dtype, default numpy.float32
        Data type for data elements.
    index_dtype : numpy.dtype, default numpy.int64

    """

    def __init__(self, batch_size, window_size=5,
                 reduce_window_size_randomly=True, shuffle=True, cbow=False,
                 dtype='float32', index_dtype='int64'):
        self._batch_size = batch_size
        self._window_size = window_size
        self._reduce_window_size_randomly = reduce_window_size_randomly
        self._shuffle = shuffle
        self._cbow = cbow
        self._dtype = dtype
        self._index_dtype = index_dtype

    def __call__(self, corpus):
        """Batchify a dataset.

        Parameters
        ----------
        corpus : list of lists of int
            List of coded sentences. A coded sentence itself is a list of token
            indices. Context samples do not cross sentence boundaries.

         Returns
         -------
         DataStream
             Each element of the DataStream is a tuple of 2 elements (center,
             context). center is a NDArray of shape (batch_size, ). context is
             a tuple of 3 NDArrays, representing a sparse COO array (data, row,
             col). The center and context arrays contain the center and
             correpsonding context works respectively. A sparse representation
             is used for context as the number of context words for one center
             word varies based on the randomly chosen context window size and
             sentence boundaries.

        """
        return _EmbeddingCenterContextBatchify(
            corpus, self._batch_size, self._window_size,
            self._reduce_window_size_randomly, self._shuffle, cbow=self._cbow,
            dtype=self._dtype, index_dtype=self._index_dtype)


class _EmbeddingCenterContextBatchify(DataStream):
    def __init__(self, coded, batch_size, window_size,
                 reduce_window_size_randomly, shuffle, cbow, dtype,
                 index_dtype):
        self._coded = coded
        self._batch_size = batch_size
        self._window_size = window_size
        self._reduce_window_size_randomly = reduce_window_size_randomly
        self._shuffle = shuffle
        self._cbow = cbow
        self._dtype = dtype
        self._index_dtype = index_dtype

    def __iter__(self):
        coded = [c for c in self._coded if len(c) > 1]
        if self._shuffle:
            random.shuffle(coded)

        # Represent variable length data as continous data and pointer arrays
        sentence_boundaries = np.cumsum([len(c) for c in coded], dtype=np.int64)
        coded = np.concatenate(coded)

        iterator = _CenterContextIterator(coded, sentence_boundaries, self._window_size,
                                         self._batch_size, random_window_size=self._reduce_window_size_randomly,
                                         cbow=self._cbow, seed=random.getrandbits(32))
        for (center, context_data, context_row,
             context_col) in iterator:

            context_data = nd.array(context_data, dtype=self._dtype)
            context_row = nd.array(context_row, dtype=self._index_dtype)
            context_col = nd.array(context_col, dtype=self._index_dtype)
            context_coo = (context_data, context_row, context_col)
            yield nd.array(center, dtype=self._index_dtype), context_coo


cdef class _CenterContextIterator:
    cdef int64_t[:] sentences  # TODO memory view of python objects
    cdef int64_t[:] sentence_boundaries
    cdef int64_t window
    cdef int64_t batch_size
    cdef bool random_window_size
    cdef bool cbow

    cdef mt19937 gen
    cdef uniform_int_distribution[int64_t] dist

    # Iteration state
    cdef int64_t word_pointer
    cdef int64_t num_context_skip

    def __init__(self, sentences, sentence_boundaries, window, batch_size,
                 random_window_size, cbow, seed):
        self.sentences = sentences
        self.sentence_boundaries = sentence_boundaries
        self.window = window
        self.batch_size = batch_size
        self.random_window_size = random_window_size
        self.cbow = cbow

        self.gen = mt19937(seed)
        self.dist = uniform_int_distribution[int64_t](1, window)

        self.word_pointer = 0
        self.num_context_skip = 0

    def __iter__(self):
        return self

    def __next__(self):
        cdef vector[int64_t] center_batch
        cdef vector[double] context_data
        cdef vector[int64_t] context_row
        cdef vector[int64_t] context_col

        cdef int64_t i = 0
        cdef int64_t center = 0
        cdef vector[int64_t] contexts
        with nogil:
            while i < self.batch_size:
                if self.word_pointer >= self.sentence_boundaries[-1]:
                    # There is no data left
                    break

                center = self.sentences[self.word_pointer]
                contexts = get_context(self.word_pointer, self.sentences,
                                       self.sentence_boundaries, self.window,
                                       self.random_window_size, self.gen, self.dist)
                for j in range(contexts.size()):
                    if self.num_context_skip > j:
                        # In SkipGram mode, there may be some leftover contexts
                        # form the last batch
                        continue
                    elif i < self.batch_size:
                        self.num_context_skip = 0
                        context_row.push_back(i)
                        context_col.push_back(contexts[j])
                        if self.cbow:
                            context_data.push_back(1.0 / contexts.size())
                        else:
                            center_batch.push_back(center)
                            context_data.push_back(1)
                            i += 1
                    else:
                        self.num_context_skip = j
                        if self.cbow:
                            with gil:
                                raise RuntimeError
                        break

                if self.cbow:
                    center_batch.push_back(center)
                    i += 1

                if self.num_context_skip == 0:
                    self.word_pointer += 1
                else:
                    if i != self.batch_size:
                        with gil:
                            raise RuntimeError
                    break

        if len(center_batch) == self.batch_size:
            return center_batch, context_data, context_row, context_col
        else:
            assert self.word_pointer >= self.sentence_boundaries[-1]
            raise StopIteration


cdef int64_t searchsorted(const int64_t[:] a, const int64_t v) nogil:
    cdef int64_t left = 0
    cdef int64_t right = a.shape[0] - 1
    cdef int64_t ptr
    while left <= right:
        ptr = (left + right) // 2
        if a[ptr] < v:
            left = ptr + 1
        elif a[ptr] > v:
            right = ptr - 1
        else:
            break
    return left


cdef vector[int64_t] get_context(const int64_t center_idx, const int64_t[:] sentences,
                                 const int64_t[:] sentence_boundaries,
                                 const int64_t window_size_, const bool random_window_size,
                                 mt19937 gen, uniform_int_distribution[int64_t] dist) nogil:
    """Compute the context with respect to a center word in a sentence.

    Takes an numpy array of flattened sentences and their boundaries.

    """
    cdef int64_t sentence_index = searchsorted(sentence_boundaries, center_idx)
    cdef int64_t sentence_end = sentence_boundaries[sentence_index]
    cdef int64_t sentence_start
    if sentence_index == 0:
        sentence_start = 0
    else:
        sentence_start = sentence_boundaries[sentence_index - 1]

    cdef int64_t window_size
    if random_window_size:
        window_size = dist(gen)
    else:
        window_size = window_size_
    start_idx = max(sentence_start, center_idx - window_size)
    end_idx = min(sentence_end, center_idx + window_size + 1)

    cdef vector[int64_t] context
    for i in range(start_idx, center_idx):
        context.push_back(sentences[i])
    for i in range(center_idx + 1, end_idx):
        context.push_back(sentences[i])
    return context
