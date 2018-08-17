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
__all__ = ['TextLineDataset', 'CorpusDataset']

import io
import os

from mxnet.gluon.data import SimpleDataset

from .utils import concat_sequence, line_splitter, whitespace_splitter


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


def _corpus_dataset_process(s, bos, eos):
    tokens = [bos] if bos else []
    tokens.extend(s)
    if eos:
        tokens.append(eos)
    return tokens


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
