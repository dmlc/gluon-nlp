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
__all__ = ['TextLineDataset', 'CorpusDataset', 'ConcatDataset', 'TSVDataset', 'NumpyDataset',
           'Filter', 'RangeFilter', 'SplitFilter']

import io
import os
import bisect
import numpy as np

from mxnet.gluon.data import SimpleDataset, Dataset, ArrayDataset

from .utils import concat_sequence, line_splitter, whitespace_splitter, Splitter


class ConcatDataset(Dataset):
    """Dataset that concatenates a list of datasets.

    Parameters
    ----------
    datasets : list
        List of datasets.
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.cum_sizes = np.cumsum([0] + [len(d) for d in datasets])

    def __getitem__(self, i):
        dataset_id = bisect.bisect_right(self.cum_sizes, i)
        sample_id = i - self.cum_sizes[dataset_id - 1]
        return self.datasets[dataset_id - 1][sample_id]

    def __len__(self):
        return self.cum_sizes[-1]

class Filter(object):
    """Abstract Filter class for Dataset creation.

    A Filter can be used during Dataset construction time to reduce the number
    of samples to hold in memory (and subsequent transformations).
    """
    def __call__(self, index, data):
        """Check if the data sample passes the filter.

        Parameters
        ----------
        index : int
            The original dataset index before filtering is applied.
        sample : object
            The original data sample object at the provided index.
        """
        raise NotImplementedError()

class RangeFilter(Filter):
    """RangeFilter filters the data samples based on the range [start_idx, end_idx)
    from the dataset. Only data samples within the range passes the filter.

    Parameters
    ----------
    start_idx : int
        The start index (included).
    end_idx : int or None
        The end index (excluded). If set to None, it is set to infinity.

    Example
    -------
    >>> data =  "a,b,c\n"
    >>> data += "d,e,f\n"
    >>> data += "g,h,i\n"
    >>> data += "j,k,l\n"
    >>> data += "m,n,o\n"
    >>> with open('test_range_filter.txt', 'w') as fout:
    >>>     fout.write(data)
    >>>
    >>> # create 2 partitions, and read partition 0 only
    >>> filter_fn = nlp.data.RangeFilter(1, 3)
    >>> dataset = nlp.data.TextLineDataset('test_range_filter.txt', filter_fn=filter_fn)
    >>> len(dataset)
    2
    >>> dataset[0]
    "d,e,f"
    >>> dataset[1]
    "g,h,i"
    """
    def __init__(self, start_idx, end_idx):
        self.start = start_idx
        self.end = end_idx
        if end_idx is not None:
            assert self.start < self.end, 'end_idx must be greater than start_idx'

    def __call__(self, index, data):
        """Check if the data sample passes the filter.

        Parameters
        ----------
        index : int
            The original dataset index before filtering is applied.
        sample : object
            The original data sample object at the provided index.
        """
        if self.end is not None:
            return index >= self.start and index < self.end
        else:
            return index >= self.start

class SplitFilter(Filter):
    """SplitFilter filters the data samples based on the number of partitions
    and partition index of the dataset. Only data samples for the target
    partition index passes the filter.

    Parameters
    ----------
    num_parts : int
        The number of partitions.
    part_idx : int
        The target partition index that will pass the filter.

    Example
    -------
    >>> data =  "a,b,c\n"
    >>> data += "d,e,f\n"
    >>> data += "g,h,i\n"
    >>> data += "j,k,l\n"
    >>> data += "m,n,o\n"
    >>> with open('test_split_filter.txt', 'w') as fout:
    >>>     fout.write(data)
    >>>
    >>> # create 2 partitions, and read partition 0 only
    >>> filter_fn = nlp.data.SplitFilter(2, 0)
    >>> dataset = nlp.data.TextLineDataset('test_split_filter.txt', filter_fn=filter_fn)
    >>> len(dataset)
    3
    >>> dataset[0]
    "a,b,c"
    >>> dataset[1]
    "g,h,i"
    >>> dataset[2]
    "m,n,o"
    """
    def __init__(self, num_parts, part_idx):
        self.num_parts = num_parts
        self.part_idx = part_idx
        assert self.part_idx < self.num_parts, 'part_idx should be less than num_parts'

    def __call__(self, index, data):
        """Check if the data sample passes the filter.

        Parameters
        ----------
        index : int
            The original dataset index before filtering is applied.
        sample : object
            The original data sample object at the provided index.
        """
        return index % self.num_parts == self.part_idx

class TextLineDataset(SimpleDataset):
    """Dataset that comprises lines in a file. Each line will be stripped.

    Parameters
    ----------
    filename : str
        Path to the input text file.
    encoding : str, default 'utf8'
        File encoding format.
    filter_fn : Filter, default None
        Filter function to decide if the line should be kept or discarded.
    """
    def __init__(self, filename, encoding='utf8', filter_fn=None):
        lines = []
        with io.open(filename, 'r', encoding=encoding) as in_file:
            for idx, line in enumerate(in_file):
                line = line.strip()
                # check if the line should be filtered out
                if filter_fn is not None and not filter_fn(idx, line):
                    continue
                lines.append(line)
        super(TextLineDataset, self).__init__(lines)


def _corpus_dataset_process(s, bos, eos):
    tokens = [bos] if bos else []
    tokens.extend(s)
    if eos:
        tokens.append(eos)
    return tokens

class TSVDataset(SimpleDataset):
    """Common tab separated text dataset that reads text fields based on provided sample splitter
    and field separator.

    The returned dataset includes samples, each of which can either be a list of text fields
    if field_separator is specified, or otherwise a single string segment produced by the
    sample_splitter.

    Example::

        # assume `test.tsv` contains the following content:
        # Id\tFirstName\tLastName
        # a\tJiheng\tJiang
        # b\tLaoban\tZha
        # discard the first line and select the 0th and 2nd fields
        dataset = data.TSVDataset('test.tsv', num_discard_samples=1, field_indices=[0, 2])
        assert dataset[0] == [u'a', u'Jiang']
        assert dataset[1] == [u'b', u'Zha']

    Parameters
    ----------
    filename : str or list of str
        Path to the input text file or list of paths to the input text files.
    encoding : str, default 'utf8'
        File encoding format.
    sample_splitter : function, default str.splitlines
        A function that splits the dataset string into samples.
    field_separator : function or None, default Splitter('\t')
        A function that splits each sample string into list of text fields.
        If None, raw samples are returned according to `sample_splitter`.
    num_discard_samples : int, default 0
        Number of samples discarded at the head of the first file.
    field_indices : list of int or None, default None
        If set, for each sample, only fields with provided indices are selected as the output.
        Otherwise all fields are returned.
    """
    def __init__(self, filename, encoding='utf8',
                 sample_splitter=line_splitter, field_separator=Splitter('\t'),
                 num_discard_samples=0, field_indices=None):
        assert sample_splitter, 'sample_splitter must be specified.'

        if not isinstance(filename, (tuple, list)):
            filename = (filename, )

        self._filenames = [os.path.expanduser(f) for f in filename]
        self._encoding = encoding
        self._sample_splitter = sample_splitter
        self._field_separator = field_separator
        self._num_discard_samples = num_discard_samples
        self._field_indices = field_indices
        super(TSVDataset, self).__init__(self._read())

    def _should_discard(self):
        discard = self._num_discard_samples > 0
        self._num_discard_samples -= 1
        return discard

    def _field_selector(self, fields):
        if not self._field_indices:
            return fields
        return [fields[i] for i in self._field_indices]

    def _read(self):
        all_samples = []
        for filename in self._filenames:
            with io.open(filename, 'r', encoding=self._encoding) as fin:
                content = fin.read()
            samples = (s for s in self._sample_splitter(content) if not self._should_discard())
            if self._field_separator:
                samples = [self._field_selector(self._field_separator(s)) for s in samples]
            all_samples += samples
        return all_samples

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
        The token to add at the beginning of each sequence. If None, or if tokenizer is not
        specified, then nothing is added.
    eos : str or None, default None
        The token to add at the end of each sequence. If None, or if tokenizer is not
        specified, then nothing is added.
    """
    def __init__(self, filename, encoding='utf8', flatten=False, skip_empty=True,
                 sample_splitter=line_splitter, tokenizer=whitespace_splitter,
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
        self._bos = bos
        self._eos = eos
        super(CorpusDataset, self).__init__(self._read())

    def _read(self):
        all_samples = []
        for filename in self._filenames:
            with io.open(filename, 'r', encoding=self._encoding) as fin:
                content = fin.read()
            samples = (s.strip() for s in self._sample_splitter(content))
            if self._tokenizer:
                samples = [
                    _corpus_dataset_process(self._tokenizer(s), self._bos, self._eos)
                    for s in samples if s or not self._skip_empty
                ]
                if self._flatten:
                    samples = concat_sequence(samples)
            elif self._skip_empty:
                samples = [s for s in samples if s]

            all_samples += samples
        return all_samples

class NumpyDataset(ArrayDataset):
    """A dataset wrapping over a Numpy binary (.npy, .npz) file.

    If the file is a .npy file, then a single numpy array is loaded.
    If the file is a .npz file with multiple arrays, then a list of
    numpy arrays are loaded, ordered by their key in the archive.

    Sparse matrix is not yet supported.

    Parameters
    ----------
    filename : str
        Path to the .npy or .npz file.

    Properties
    ----------
    keys: list of str or None
        The list of keys loaded from the .npz file.
    """
    def __init__(self, filename):
        arrs = np.load(filename)
        keys = None
        data = []
        if filename.endswith('.npy'):
            data.append(arrs)
        elif filename.endswith('.npz'):
            keys = sorted(arrs.keys())
            for key in keys:
                data.append(arrs[key])
        else:
            raise ValueError('Unsupported extension: %s'%filename)
        self._keys = keys
        super(NumpyDataset, self).__init__(*data)

    @property
    def keys(self):
        return self._keys

    def get_field(self, field):
        """Return the dataset corresponds to the provided key.

        Example::
            a = np.ones((2,2))
            b = np.zeros((2,2))
            np.savez('data.npz', a=a, b=b)
            dataset = NumpyDataset('data.npz')
            data_a = dataset.get_field('a')
            data_b = dataset.get_field('b')

        Parameters
        ----------
        field : str
            The name of the field to retrieve.
        """
        idx = self._keys.index(field)
        return self._data[idx]
