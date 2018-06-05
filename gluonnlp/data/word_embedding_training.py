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

# pylint: disable=
"""Word embedding training.

Datasets and samplers for training of embedding models.

"""

__all__ = ['Text8', 'ContextSampler', 'NegativeSampler']

import os
import random
import shutil
import zipfile

import numpy as np
from mxnet.gluon.utils import check_sha1, download

from .dataset import CorpusDataset
from .utils import _get_home_dir

try:
    from numba import njit, prange
    numba_njit = njit(nogil=True)
except ImportError:
    # Define numba shims
    prange = range

    def numba_njit(func):
        return func


###############################################################################
# Datasets
###############################################################################
class Text8(CorpusDataset):
    """Text8 corpus

    http://mattmahoney.net/dc/textdata.html

    Part of the test data for the Large Text Compression Benchmark
    http://mattmahoney.net/dc/text.html. The first 10**8 bytes of the English
    Wikipedia dump on Mar. 3, 2006.

    License: https://en.wikipedia.org/wiki/Wikipedia:Copyrights

    Parameters
    ----------
    root : str, default '$MXNET_HOME/datasets/text8'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

    """

    archive_file = ('text8.zip', '6c70299b93b7e1f927b42cd8f6ac1a31547c7a2e')
    data_file = {
        'train': ('text8', '0dc3edebc970dcc96137e7deda4d9995af9d93de')
    }
    url = 'http://mattmahoney.net/dc/'

    def __init__(self, root=os.path.join(_get_home_dir(), 'datasets', 'text8'),
                 segment='train'):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        self._segment = segment
        super(Text8, self).__init__(self._get_data())

    def _get_data(self):
        archive_file_name, archive_hash = self.archive_file
        data_file_name, data_hash = self.data_file[self._segment]
        root = self._root
        path = os.path.join(root, data_file_name)
        if not os.path.exists(path) or not check_sha1(path, data_hash):
            downloaded_file_path = download(self.url + archive_file_name,
                                            path=root, sha1_hash=archive_hash)

            with zipfile.ZipFile(downloaded_file_path, 'r') as zf:
                for member in zf.namelist():
                    filename = os.path.basename(member)
                    if filename:
                        dest = os.path.join(root, filename)
                        with zf.open(member) as source:
                            with open(dest, 'wb') as target:
                                shutil.copyfileobj(source, target)
        return path


###############################################################################
# Samplers
###############################################################################
class NegativeSampler(object):
    """Sampler for drawing negatives from a smoothed unigram distribution.

    Obtain an instance of NegativeSampler and call it with the batch_size to
    sample.

    Parameters
    ----------
    vocab : Vocab
        The vocabulary specifies the set of tokens and their frequencies from
        which the smoothed unigram distribution is computed.
    negative : int, default 5
        The number of negative samples to draw.
    power : float, default 0.75
        Smoothing factor.

    """

    def __init__(self, vocab, negative=5, power=0.75):
        self.vocab = vocab
        self.negative = negative
        self.power = power

        # Smoothed unigram counts for negative sampling. Negatives can be drawn
        # by sampling a number in [0, self._smoothed_cumsum[-1]) and finding
        # the respective index with np.searchsorted.
        self._smoothed_token_freq_cumsum = np.cumsum((np.array(
            vocab.idx_to_counts)**self.power).astype(np.int))

    def __call__(self, size):
        return np.searchsorted(
            self._smoothed_token_freq_cumsum,
            np.random.randint(self._smoothed_token_freq_cumsum[-1],
                              size=(size, self.negative)))


class ContextSampler(object):
    """Sample batches of contexts (and their masks) from a corpus.

    The context size is choosen uniformly at random for every sample from [1,
    `window`]. The mask is used to mask entries that lie outside of the
    randomly chosen context size.

    Batches are created lazily, to avoid generating all batches for shuffling
    before training, simply shuffle the dataset before passing it to the
    ContextSampler.

    Instantiate a ContextSampler and call it with the dataset to be sampled
    from. The call returns an iterable over batches.

    Parameters
    ----------
    batch_size : int
        Maximum size of batches. Actual batch returned can be smaller when
        running out of samples.
    window : int, default 5
        The maximum context size.

    """

    def __init__(self, batch_size, window=5):
        self.batch_size = batch_size
        self.window = window

    def __call__(self, coded):
        coded = [c for c in coded if len(c) > 1]
        sentence_boundaries = np.cumsum([len(s) for s in coded])
        coded = np.concatenate(coded)

        return _context_generator(coded, sentence_boundaries, self.window,
                                  self.batch_size)


@numba_njit
def _get_sentence_start_end(sentence_boundaries, sentence_pointer):
    end = sentence_boundaries[sentence_pointer]
    if sentence_pointer == 0:
        start = 0
    else:
        start = sentence_boundaries[sentence_pointer - 1]
    return start, end


@numba_njit
def _context_generator(coded_sentences, sentence_boundaries, window,
                       batch_size):
    word_pointer = 0
    while True:
        batch_size = min(batch_size, len(coded_sentences) - word_pointer)
        center = coded_sentences[word_pointer:
                                 word_pointer + batch_size].astype(np.float32)
        context = np.zeros((batch_size, window * 2), dtype=np.float32)
        mask = np.zeros((batch_size, window * 2), dtype=np.float32)

        for i in prange(batch_size):
            context_, mask_ = _get_context(word_pointer + i, coded_sentences,
                                           sentence_boundaries, window)
            context[i] = context_
            mask[i] = mask_

        word_pointer += batch_size

        yield center, context, mask

        if word_pointer >= sentence_boundaries[-1]:
            break


@numba_njit
def _get_context(word_pointer, coded_sentences, sentence_boundaries, window):
    sentence_pointer = np.searchsorted(sentence_boundaries, word_pointer)
    sentence_start, sentence_end = _get_sentence_start_end(
        sentence_boundaries, sentence_pointer)

    random_window_size = random.randint(1, window)

    start_idx = max(sentence_start, word_pointer - random_window_size)
    # First index outside of the window
    end_idx = min(sentence_end, word_pointer + random_window_size + 1)

    # A random reduced window size is drawn. The mask masks entries
    # that fall inside the window, but outside random the reduced
    # window size.
    mask = np.ones(window * 2, dtype=np.float32)
    mask[end_idx - start_idx:] = 0
    context = np.zeros(window * 2, dtype=np.float32)

    # Get contexts
    next_context_idx = 0
    context[:word_pointer - start_idx] = \
        coded_sentences[start_idx:word_pointer]
    next_context_idx += word_pointer - start_idx
    context[next_context_idx:next_context_idx + end_idx -
            (word_pointer + 1)] = coded_sentences[word_pointer + 1:end_idx]
    next_context_idx += end_idx - (word_pointer + 1)

    # Set mask
    mask[next_context_idx:] = 0

    return context, mask
