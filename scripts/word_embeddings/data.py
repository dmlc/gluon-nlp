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
"""Word embedding training datasets."""

__all__ = [
    'WikiDumpStream', 'preprocess_dataset', 'wiki', 'transform_data_fasttext',
    'transform_data_word2vec', 'skipgram_lookup', 'cbow_lookup',
    'skipgram_fasttext_batch', 'cbow_fasttext_batch', 'skipgram_batch',
    'cbow_batch']

import functools
import io
import itertools
import json
import math
import os
import warnings

import mxnet as mx
import numpy as np

import gluonnlp as nlp
from gluonnlp import Vocab
from gluonnlp.base import numba_njit
from gluonnlp.data import CorpusDataset, SimpleDatasetStream
from utils import print_time


def preprocess_dataset(data, min_freq=5, max_vocab_size=None):
    """Dataset preprocessing helper.

    Parameters
    ----------
    data : mx.data.Dataset
        Input Dataset. For example gluonnlp.data.Text8 or gluonnlp.data.Fil9
    min_freq : int, default 5
        Minimum token frequency for a token to be included in the vocabulary
        and returned DataStream.
    max_vocab_size : int, optional
        Specifies a maximum size for the vocabulary.

    Returns
    -------
    gluonnlp.data.DataStream
        Each sample is a valid input to
        gluonnlp.data.EmbeddingCenterContextBatchify.
    gluonnlp.Vocab
        Vocabulary of all tokens in Text8 that occur at least min_freq times of
        maximum size max_vocab_size.
    idx_to_counts : list of int
        Mapping from token indices to their occurrence-counts in the Text8
        dataset.

    """
    with print_time('count and construct vocabulary'):
        counter = nlp.data.count_tokens(itertools.chain.from_iterable(data))
        vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                          bos_token=None, eos_token=None, min_freq=min_freq,
                          max_size=max_vocab_size)
        idx_to_counts = [counter[w] for w in vocab.idx_to_token]

    def code(sentence):
        return [vocab[token] for token in sentence if token in vocab]

    with print_time('code data'):
        data = data.transform(code, lazy=False)
    data = nlp.data.SimpleDataStream([data])
    return data, vocab, idx_to_counts


def wiki(wiki_root, wiki_date, wiki_language, max_vocab_size=None):
    """Wikipedia dump helper.

    Parameters
    ----------
    wiki_root : str
        Parameter for WikiDumpStream
    wiki_date : str
        Parameter for WikiDumpStream
    wiki_language : str
        Parameter for WikiDumpStream
    max_vocab_size : int, optional
        Specifies a maximum size for the vocabulary.

    Returns
    -------
    gluonnlp.data.DataStream
        Each sample is a valid input to
        gluonnlp.data.EmbeddingCenterContextBatchify.
    gluonnlp.Vocab
        Vocabulary of all tokens in the Wikipedia corpus as provided by
        WikiDumpStream but with maximum size max_vocab_size.
    idx_to_counts : list of int
        Mapping from token indices to their occurrence-counts in the Wikipedia
        corpus.

    """
    data = WikiDumpStream(
        root=os.path.expanduser(wiki_root), language=wiki_language,
        date=wiki_date)
    vocab = data.vocab
    if max_vocab_size:
        for token in vocab.idx_to_token[max_vocab_size:]:
            vocab.token_to_idx.pop(token)
        vocab.idx_to_token = vocab.idx_to_token[:max_vocab_size]
    idx_to_counts = data.idx_to_counts

    def code(shard):
        return [[vocab[token] for token in sentence if token in vocab]
                for sentence in shard]

    data = data.transform(code)
    return data, vocab, idx_to_counts


def transform_data_fasttext(data, vocab, idx_to_counts, cbow, ngram_buckets,
                            ngrams, batch_size, window_size,
                            frequent_token_subsampling=1E-4, dtype='float32',
                            index_dtype='int64'):
    """Transform a DataStream of coded DataSets to a DataStream of batches.

    Parameters
    ----------
    data : gluonnlp.data.DataStream
        DataStream where each sample is a valid input to
        gluonnlp.data.EmbeddingCenterContextBatchify.
    vocab : gluonnlp.Vocab
        Vocabulary containing all tokens whose indices occur in data. For each
        token, it's associated subwords will be computed and used for
        constructing the batches. No subwords are used if ngram_buckets is 0.
    idx_to_counts : list of int
        List of integers such that idx_to_counts[idx] represents the count of
        vocab.idx_to_token[idx] in the underlying dataset. The count
        information is used to subsample frequent words in the dataset.
        Each token is independently dropped with probability 1 - sqrt(t /
        (count / sum_counts)) where t is the hyperparameter
        frequent_token_subsampling.
    cbow : boolean
        If True, batches for CBOW are returned.
    ngram_buckets : int
        Number of hash buckets to consider for the fastText
        nlp.vocab.NGramHashes subword function.
    ngrams : list of int
        For each integer n in the list, all ngrams of length n will be
        considered by the nlp.vocab.NGramHashes subword function.
    batch_size : int
        The returned data stream iterates over batches of batch_size.
    window_size : int
        The context window size for
        gluonnlp.data.EmbeddingCenterContextBatchify.
    frequent_token_subsampling : float
        Hyperparameter for subsampling. See idx_to_counts above for more
        information.
    dtype : str or np.dtype, default 'float32'
        Data type of data array.
    index_dtype : str or np.dtype, default 'int64'
        Data type of index arrays.

    Returns
    -------
    gluonnlp.data.DataStream
        Stream over batches. Each returned element is a list corresponding to
        the arguments for the forward pass of model.SG or model.CBOW
        respectively based on if cbow is False or True. If ngarm_buckets > 0,
        the returned sample will contain ngrams. Both model.SG or model.CBOW
        will handle them correctly as long as they are initialized with the
        subword_function returned as second argument by this function (see
        below).
    gluonnlp.vocab.NGramHashes
        The subword_function used for obtaining the subwords in the returned
        batches.

    """
    if ngram_buckets <= 0:
        raise ValueError('Invalid ngram_buckets. Use Word2Vec training '
                         'pipeline if not interested in ngrams.')

    sum_counts = float(sum(idx_to_counts))
    idx_to_pdiscard = [
        1 - math.sqrt(frequent_token_subsampling / (count / sum_counts))
        for count in idx_to_counts]

    def subsample(shard):
        return [[
            t for t, r in zip(sentence,
                              np.random.uniform(0, 1, size=len(sentence)))
            if r > idx_to_pdiscard[t]] for sentence in shard]

    data = data.transform(subsample)

    batchify = nlp.data.batchify.EmbeddingCenterContextBatchify(
        batch_size=batch_size, window_size=window_size, cbow=cbow,
        weight_dtype=dtype, index_dtype=index_dtype)
    data = data.transform(batchify)

    with print_time('prepare subwords'):
        subword_function = nlp.vocab.create_subword_function(
            'NGramHashes', ngrams=ngrams, num_subwords=ngram_buckets)

        # Store subword indices for all words in vocabulary
        idx_to_subwordidxs = list(subword_function(vocab.idx_to_token))
        subwordidxs = np.concatenate(idx_to_subwordidxs)
        subwordidxsptr = np.cumsum([
            len(subwordidxs) for subwordidxs in idx_to_subwordidxs])
        subwordidxsptr = np.concatenate([
            np.zeros(1, dtype=np.int64), subwordidxsptr])
        if cbow:
            subword_lookup = functools.partial(
                cbow_lookup, subwordidxs=subwordidxs,
                subwordidxsptr=subwordidxsptr, offset=len(vocab))
        else:
            subword_lookup = functools.partial(
                skipgram_lookup, subwordidxs=subwordidxs,
                subwordidxsptr=subwordidxsptr, offset=len(vocab))
        max_subwordidxs_len = max(len(s) for s in idx_to_subwordidxs)
        if max_subwordidxs_len > 500:
            warnings.warn(
                'The word with largest number of subwords '
                'has {} subwords, suggesting there are '
                'some noisy words in your vocabulary. '
                'You should filter out very long words '
                'to avoid memory issues.'.format(max_subwordidxs_len))

    data = UnchainStream(data)

    if cbow:
        batchify_fn = cbow_fasttext_batch
    else:
        batchify_fn = skipgram_fasttext_batch
    batchify_fn = functools.partial(
        batchify_fn, num_tokens=len(vocab) + len(subword_function),
        subword_lookup=subword_lookup, dtype=dtype, index_dtype=index_dtype)

    return data, batchify_fn, subword_function


def transform_data_word2vec(data, vocab, idx_to_counts, cbow, batch_size,
                            window_size, frequent_token_subsampling=1E-4,
                            dtype='float32', index_dtype='int64'):
    """Transform a DataStream of coded DataSets to a DataStream of batches.

    Parameters
    ----------
    data : gluonnlp.data.DataStream
        DataStream where each sample is a valid input to
        gluonnlp.data.EmbeddingCenterContextBatchify.
    vocab : gluonnlp.Vocab
        Vocabulary containing all tokens whose indices occur in data.
    idx_to_counts : list of int
        List of integers such that idx_to_counts[idx] represents the count of
        vocab.idx_to_token[idx] in the underlying dataset. The count
        information is used to subsample frequent words in the dataset.
        Each token is independently dropped with probability 1 - sqrt(t /
        (count / sum_counts)) where t is the hyperparameter
        frequent_token_subsampling.
    batch_size : int
        The returned data stream iterates over batches of batch_size.
    window_size : int
        The context window size for
        gluonnlp.data.EmbeddingCenterContextBatchify.
    frequent_token_subsampling : float
        Hyperparameter for subsampling. See idx_to_counts above for more
        information.
    dtype : str or np.dtype, default 'float32'
        Data type of data array.
    index_dtype : str or np.dtype, default 'int64'
        Data type of index arrays.

    Returns
    -------
    gluonnlp.data.DataStream
        Stream over batches.
    """

    sum_counts = float(sum(idx_to_counts))
    idx_to_pdiscard = [
        1 - math.sqrt(frequent_token_subsampling / (count / sum_counts))
        for count in idx_to_counts]

    def subsample(shard):
        return [[
            t for t, r in zip(sentence,
                              np.random.uniform(0, 1, size=len(sentence)))
            if r > idx_to_pdiscard[t]] for sentence in shard]

    data = data.transform(subsample)

    batchify = nlp.data.batchify.EmbeddingCenterContextBatchify(
        batch_size=batch_size, window_size=window_size, cbow=cbow,
        weight_dtype=dtype, index_dtype=index_dtype)
    data = data.transform(batchify)
    data = UnchainStream(data)

    if cbow:
        batchify_fn = cbow_batch
    else:
        batchify_fn = skipgram_batch
    batchify_fn = functools.partial(batchify_fn, num_tokens=len(vocab),
                                    dtype=dtype, index_dtype=index_dtype)

    return data, batchify_fn,


def cbow_fasttext_batch(centers, contexts, num_tokens, subword_lookup, dtype,
                        index_dtype):
    """Create a batch for CBOW training objective with subwords."""
    _, contexts_row, contexts_col = contexts
    data, row, col = subword_lookup(contexts_row, contexts_col)
    centers = mx.nd.array(centers, dtype=index_dtype)
    contexts = mx.nd.sparse.csr_matrix(
        (data, (row, col)), dtype=dtype,
        shape=(len(centers), num_tokens))  # yapf: disable
    return centers, contexts


def skipgram_fasttext_batch(centers, contexts, num_tokens, subword_lookup,
                            dtype, index_dtype):
    """Create a batch for SG training objective with subwords."""
    contexts = mx.nd.array(contexts[2], dtype=index_dtype)
    data, row, col = subword_lookup(centers)
    centers = mx.nd.array(centers, dtype=index_dtype)
    centers_csr = mx.nd.sparse.csr_matrix(
        (data, (row, col)), dtype=dtype,
        shape=(len(centers), num_tokens))  # yapf: disable
    return centers_csr, contexts, centers


def cbow_batch(centers, contexts, num_tokens, dtype, index_dtype):
    """Create a batch for CBOW training objective."""
    contexts_data, contexts_row, contexts_col = contexts
    centers = mx.nd.array(centers, dtype=index_dtype)
    contexts = mx.nd.sparse.csr_matrix(
        (contexts_data, (contexts_row, contexts_col)),
        dtype=dtype, shape=(len(centers), num_tokens))  # yapf: disable
    return centers, contexts


def skipgram_batch(centers, contexts, num_tokens, dtype, index_dtype):
    """Create a batch for SG training objective."""
    contexts = mx.nd.array(contexts[2], dtype=index_dtype)
    indptr = mx.nd.arange(len(centers) + 1)
    centers = mx.nd.array(centers, dtype=index_dtype)
    centers_csr = mx.nd.sparse.csr_matrix(
        (mx.nd.ones(centers.shape), centers, indptr), dtype=dtype,
        shape=(len(centers), num_tokens))
    return centers_csr, contexts, centers


class UnchainStream(nlp.data.DataStream):
    def __init__(self, iterable):
        self._stream = iterable

    def __iter__(self):
        return iter(itertools.chain.from_iterable(self._stream))


@numba_njit
def skipgram_lookup(indices, subwordidxs, subwordidxsptr, offset=0):
    """Get a sparse COO array of words and subwords for SkipGram.

    Parameters
    ----------
    indices : numpy.ndarray
        Array containing numbers in [0, vocabulary_size). The element at
        position idx is taken to be the word that occurs at row idx in the
        SkipGram batch.
    offset : int
        Offset to add to each subword index.
    subwordidxs : numpy.ndarray
        Array containing concatenation of all subwords of all tokens in the
        vocabulary, in order of their occurrence in the vocabulary.
        For example np.concatenate(idx_to_subwordidxs)
    subwordidxsptr
        Array containing pointers into subwordidxs array such that
        subwordidxs[subwordidxsptr[i]:subwordidxsptr[i+1]] returns all subwords
        of of token i. For example subwordidxsptr = np.cumsum([
        len(subwordidxs) for subwordidxs in idx_to_subwordidxs])
    offset : int, default 0
        Offset to add to each subword index.

    Returns
    -------
    numpy.ndarray of dtype float32
        Array containing weights such that for each row, all weights sum to
        1. In particular, all elements in a row have weight 1 /
        num_elements_in_the_row
    numpy.ndarray of dtype int64
        This array is the row array of a sparse array of COO format.
    numpy.ndarray of dtype int64
        This array is the col array of a sparse array of COO format.

    """
    row = []
    col = []
    data = []
    for i, idx in enumerate(indices):
        start = subwordidxsptr[idx]
        end = subwordidxsptr[idx + 1]

        row.append(i)
        col.append(idx)
        data.append(1 / (1 + end - start))
        for subword in subwordidxs[start:end]:
            row.append(i)
            col.append(subword + offset)
            data.append(1 / (1 + end - start))

    return (np.array(data, dtype=np.float32), np.array(row, dtype=np.int64),
            np.array(col, dtype=np.int64))


@numba_njit
def cbow_lookup(context_row, context_col, subwordidxs, subwordidxsptr,
                offset=0):
    """Get a sparse COO array of words and subwords for CBOW.

    Parameters
    ----------
    context_row : numpy.ndarray of dtype int64
        Array of same length as context_col containing numbers in [0,
        batch_size). For each idx, context_row[idx] specifies the row that
        context_col[idx] occurs in a sparse matrix.
    context_col : numpy.ndarray of dtype int64
        Array of same length as context_row containing numbers in [0,
        vocabulary_size). For each idx, context_col[idx] is one of the
        context words in the context_row[idx] row of the batch.
    subwordidxs : numpy.ndarray
        Array containing concatenation of all subwords of all tokens in the
        vocabulary, in order of their occurrence in the vocabulary.
        For example np.concatenate(idx_to_subwordidxs)
    subwordidxsptr
        Array containing pointers into subwordidxs array such that
        subwordidxs[subwordidxsptr[i]:subwordidxsptr[i+1]] returns all subwords
        of of token i. For example subwordidxsptr = np.cumsum([
        len(subwordidxs) for subwordidxs in idx_to_subwordidxs])
    offset : int, default 0
        Offset to add to each subword index.

    Returns
    -------
    numpy.ndarray of dtype float32
        Array containing weights summing to 1. The weights are chosen such
        that the sum of weights for all subwords and word units of a given
        context word is equal to 1 / number_of_context_words_in_the_row.
        This array is the data array of a sparse array of COO format.
    numpy.ndarray of dtype int64
        This array is the row array of a sparse array of COO format.
    numpy.ndarray of dtype int64
        This array is the col array of a sparse array of COO format.
        Array containing weights such that for each row, all weights sum to
        1. In particular, all elements in a row have weight 1 /
        num_elements_in_the_row

    """
    row = []
    col = []
    data = []

    num_rows = np.max(context_row) + 1
    row_to_numwords = np.zeros(num_rows)

    for i, idx in enumerate(context_col):
        start = subwordidxsptr[idx]
        end = subwordidxsptr[idx + 1]

        row_ = context_row[i]
        row_to_numwords[row_] += 1

        row.append(row_)
        col.append(idx)
        data.append(1 / (1 + end - start))
        for subword in subwordidxs[start:end]:
            row.append(row_)
            col.append(subword + offset)
            data.append(1 / (1 + end - start))

    # Normalize by number of words
    for i, row_ in enumerate(row):
        assert 0 <= row_ <= num_rows
        data[i] /= row_to_numwords[row_]

    return (np.array(data, dtype=np.float32), np.array(row, dtype=np.int64),
            np.array(col, dtype=np.int64))


class WikiDumpStream(SimpleDatasetStream):
    """Stream for preprocessed Wikipedia Dumps.

    Expects data in format
    - root/date/wiki.language/*.txt
    - root/date/wiki.language/vocab.json
    - root/date/wiki.language/counts.json

    Parameters
    ----------
    path : str
        Path to a folder storing the dataset and preprocessed vocabulary.
    skip_empty : bool, default True
        Whether to skip the empty samples produced from sample_splitters. If
        False, `bos` and `eos` will be added in empty samples.
    bos : str or None, default None
        The token to add at the beginning of each sentence. If None, nothing is
        added.
    eos : str or None, default None
        The token to add at the end of each sentence. If None, nothing is
        added.

    Attributes
    ----------
    vocab : gluonnlp.Vocab
        Vocabulary object constructed from vocab.json.
    idx_to_counts : list[int]
        Mapping from vocabulary word indices to word counts.

    """

    def __init__(self, root, language, date, skip_empty=True, bos=None,
                 eos=None):
        self._root = root
        self._language = language
        self._date = date
        self._path = os.path.join(root, date, 'wiki.' + language)

        if not os.path.isdir(self._path):
            raise ValueError('{} is not valid. '
                             'Please make sure that the path exists and '
                             'contains the preprocessed files.'.format(
                                 self._path))

        self._file_pattern = os.path.join(self._path, '*.txt')
        super(WikiDumpStream, self).__init__(
            dataset=CorpusDataset, file_pattern=self._file_pattern,
            skip_empty=skip_empty, bos=bos, eos=eos)

    @property
    def vocab(self):
        path = os.path.join(self._path, 'vocab.json')
        with io.open(path, 'r', encoding='utf-8') as in_file:
            return Vocab.from_json(in_file.read())

    @property
    def idx_to_counts(self):
        path = os.path.join(self._path, 'counts.json')
        with io.open(path, 'r', encoding='utf-8') as in_file:
            return json.load(in_file)
