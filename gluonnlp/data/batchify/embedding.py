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

from mxnet import nd
import numpy as np

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
    """Batches of center and contexts words (and their masks).

    The context size is choosen uniformly at random for every sample from [1,
    `window`] if reduce_window_size_randomly is True. The mask is used to mask
    entries that lie outside of the randomly chosen context size. Contexts do
    not cross sentence boundaries.

    Batches are created lazily on a optionally shuffled version of the Dataset.

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

    """

    def __init__(self,
                 batch_size,
                 window_size=5,
                 reduce_window_size_randomly=True,
                 shuffle=True):
        self._batch_size = batch_size
        self._window_size = window_size
        self._reduce_window_size_randomly = reduce_window_size_randomly
        self._shuffle = shuffle

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
             Each element of the DataStream is a tuple of 3 NDArrays (center,
             context, mask). The center array has shape (batch_size, 1). The context
             and mask arrays have shape (batch_size, 2*window). The center and
             context arrays contain the center and correpsonding context works
             respectively. The mask array masks invalid elements in the context
             array. Elements in the context array can be invalid due to insufficient
             context elements at a certain position in a sentence or a randomly
             reduced context size.

        """
        return _EmbeddingCenterContextBatchify(
            corpus, self._batch_size, self._window_size,
            self._reduce_window_size_randomly, self._shuffle)


class _EmbeddingCenterContextBatchify(DataStream):
    def __init__(self, coded, batch_size, window_size, reduce_window_size_randomly,
                 shuffle):
        self._coded = coded
        self._batch_size = batch_size
        self._window_size = window_size
        self._reduce_window_size_randomly = reduce_window_size_randomly
        self._shuffle = shuffle

    def __iter__(self):
        if prange is range:
            logging.warning(
                'EmbeddingCenterContextBatchify supports just in time compilation '
                'with numba, but numba is not installed. '
                'Consider "pip install numba" for significant speed-ups.')

        coded = [c for c in self._coded if len(c) > 1]
        if self._shuffle:
            random.shuffle(coded)

        sentence_boundaries = np.cumsum([len(c) for c in coded])
        coded = np.concatenate(coded)  # numpy array for numba

        for center, context, mask in _context_generator(
                coded,
                sentence_boundaries,
                self._window_size,
                self._batch_size,
                random_window_size=self._reduce_window_size_randomly,
                seed=random.getrandbits(32)):
            yield nd.array(center), nd.array(context), nd.array(mask)


@numba_njit
def _get_sentence_start_end(sentence_boundaries, sentence_pointer):
    end = sentence_boundaries[sentence_pointer]
    if sentence_pointer == 0:
        start = 0
    else:
        start = sentence_boundaries[sentence_pointer - 1]
    return start, end


@numba_njit
def _context_generator(sentences, sentence_boundaries, window, batch_size,
                       random_window_size, seed):
    word_pointer = 0
    max_length = 2 * window
    while True:
        batch_size = min(batch_size, len(sentences) - word_pointer)
        center = np.expand_dims(
            sentences[word_pointer:word_pointer + batch_size],
            -1).astype(np.float32)
        context = np.zeros((batch_size, max_length), dtype=np.int_)
        mask = np.zeros((batch_size, max_length), dtype=np.int_)

        for i in prange(batch_size):
            context_ = _get_context(word_pointer + i, sentences,
                                    sentence_boundaries, window,
                                    random_window_size, seed)
            context[i, :len(context_)] = context_
            mask[i, :len(context_)] = 1

        word_pointer += batch_size

        yield center, context, mask

        if word_pointer >= sentence_boundaries[-1]:
            break


@numba_njit
def _get_context(center_index, sentences, sentence_boundaries, window_size,
                 random_window_size, seed):
    """Compute the context with respect to a center word in a sentence.

    Takes an numpy array of flattened sentences and their boundaries.

    """
    random.seed(seed + center_index)

    sentence_index = np.searchsorted(sentence_boundaries, center_index)
    sentence_start, sentence_end = _get_sentence_start_end(
        sentence_boundaries, sentence_index)

    if random_window_size:
        window_size = random.randint(1, window_size)
    start_idx = max(sentence_start, center_index - window_size)
    end_idx = min(sentence_end, center_index + window_size + 1)

    if start_idx != center_index and center_index + 1 != end_idx:
        context = np.concatenate((sentences[start_idx:center_index],
                                  sentences[center_index + 1:end_idx]))
    elif start_idx != center_index:
        context = sentences[start_idx:center_index]
    elif center_index + 1 != end_idx:
        context = sentences[center_index + 1:end_idx]
    else:
        raise RuntimeError('Too short sentence passed to _one_center_context')

    return context
