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

# pylint: disable=invalid-encoded-data
"""Transformer API. It provides tools for common transformation on samples in text dataset, such as
clipping, padding, and tokenization."""
from __future__ import absolute_import
from __future__ import print_function

__all__ = ['ClipSequence', 'PadSequence', 'NLTKMosesTokenizer', 'SpacyTokenizer']

import numpy as np
import mxnet as mx


class ClipSequence(object):
    """Clip the sequence to have length no more than `length`.

    Parameters
    ----------
    length : int
        Maximum length of the sequence

    Examples
    --------
    >>> from mxnet.gluon.data import SimpleDataset
    >>> datasets = SimpleDataset([[1, 3, 5, 7], [1, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8]])
    >>> list(datasets.transform(ClipSequence(4)))
    [[1, 3, 5, 7], [1, 2, 3], [1, 2, 3, 4]]
    >>> datasets = SimpleDataset([np.array([[1, 3], [5, 7], [7, 5], [3, 1]]),
    ...                           np.array([[1, 2], [3, 4], [5, 6], [6, 5], [4, 3], [2, 1]]),
    ...                           np.array([[2, 4], [4, 2]])])
    >>> list(datasets.transform(ClipSequence(3)))
    [array([[1, 3],
            [5, 7],
            [7, 5]]), array([[1, 2],
            [3, 4],
            [5, 6]]), array([[2, 4],
            [4, 2]])]
    """
    def __init__(self, length):
        self._length = length

    def __call__(self, sample):
        return sample[:min(len(sample), self._length)]


class PadSequence(object):
    """Pad the sequence.

    Pad the sequence to the given `length` by inserting `pad_val`. If `clip` is set,
    sequence that has length larger than `length` will be clipped.

    Parameters
    ----------
    length : int
        The maximum length to pad/clip the sequence
    pad_val : number
        The pad value. Default 0
    clip : bool

    Examples
    --------
    >>> from mxnet.gluon.data import SimpleDataset
    >>> datasets = SimpleDataset([[1, 3, 5, 7], [1, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8]])
    >>> list(datasets.transform(PadSequence(6)))
    [[1, 3, 5, 7, 0, 0], [1, 2, 3, 0, 0, 0], [1, 2, 3, 4, 5, 6]]
    >>> list(datasets.transform(PadSequence(6, clip=False)))
    [[1, 3, 5, 7, 0, 0], [1, 2, 3, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8]]
    >>> list(datasets.transform(PadSequence(6, pad_val=-1, clip=False)))
    [[1, 3, 5, 7, -1, -1], [1, 2, 3, -1, -1, -1], [1, 2, 3, 4, 5, 6, 7, 8]]
    """
    def __init__(self, length, pad_val=0, clip=True):
        self._length = length
        self._pad_val = pad_val
        self._clip = clip

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample : list of number or mx.nd.NDArray or np.ndarray

        Returns
        -------
        ret : list of number or mx.nd.NDArray or np.ndarray
        """
        sample_length = len(sample)
        if sample_length >= self._length:
            if self._clip and sample_length > self._length:
                return sample[:self._length]
            else:
                return sample
        else:
            if isinstance(sample, mx.nd.NDArray):
                # TODO(sxjscience) Use this trick for padding because mx.pad currently only supports
                # 4D/5D inputs
                new_sample_shape = (self._length,) + sample.shape[1:]
                ret = mx.nd.full(shape=new_sample_shape, val=self._pad_val, ctx=sample.context,
                                 dtype=sample.dtype)
                ret[:sample_length] = sample
                return ret
            elif isinstance(sample, np.ndarray):
                pad_width = [(0, self._length - sample_length)] +\
                            [(0, 0) for _ in range(sample.ndim - 1)]
                return np.pad(sample, mode='constant', constant_values=self._pad_val,
                              pad_width=pad_width)
            elif isinstance(sample, list):
                return sample + [self._pad_val for _ in range(self._length - sample_length)]
            else:
                raise NotImplementedError('The input must be 1) list or 2) numpy.ndarray or 3) '
                                          'mxnet.NDArray, received type=%s' % str(type(sample)))


class NLTKMosesTokenizer(object):
    r"""Apply the Moses Tokenizer implemented in NLTK.

    Users are required to [install NLTK](https://www.nltk.org/install.html) to use this tokenizer.

    Examples
    --------
    >>> tokenizer = NLTKMosesTokenizer()
    >>> tokenizer("Gluon NLP toolkit provides a suite of text processing tools.")
    ['Gluon',
     'NLP',
     'toolkit',
     'provides',
     'a',
     'suite',
     'of',
     'text',
     'processing',
     'tools',
     '.']
    >>> tokenizer("Das Gluon NLP-Toolkit stellt eine Reihe von Textverarbeitungstools "
    ...           "zur Verfügung.")
    ['Das',
     'Gluon',
     'NLP-Toolkit',
     'stellt',
     'eine',
     'Reihe',
     'von',
     'Textverarbeitungstools',
     'zur',
     'Verfügung',
     '.']
    """
    def __init__(self):
        try:
            from nltk.tokenize.moses import MosesTokenizer
        except ImportError:
            raise ImportError('NLTK is not installed. You must install NLTK in order to use the '
                              'NLTKMosesTokenizer. You can refer to the official installation '
                              'guide in https://www.nltk.org/install.html .')
        self._tokenizer = MosesTokenizer()

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample: str
            The sentence to tokenize

        Returns
        -------
        ret : list of strs
            List of tokens
        """
        return self._tokenizer.tokenize(sample)


class SpacyTokenizer(object):
    r"""Apply the Spacy Tokenizer.

    Users are required to [install spaCy](https://spacy.io/usage/) to use this tokenizer and
    download the corresponding NLP models. We only support spacy>=2.0.0.

    Parameters
    ----------
    lang : str
        The language to tokenize. Default is "en", i.e, English.
        You may refer to https://spacy.io/usage/models for supported languages.

    Examples
    --------
    >>> tokenizer = SpacyTokenizer()
    >>> tokenizer(u"Gluon NLP toolkit provides a suite of text processing tools.")
    ['Gluon',
     'NLP',
     'toolkit',
     'provides',
     'a',
     'suite',
     'of',
     'text',
     'processing',
     'tools',
     '.']
    >>> tokenizer = SpacyTokenizer('de')
    >>> tokenizer(u"Das Gluon NLP-Toolkit stellt eine Reihe von Textverarbeitungstools"
    ...            " zur Verfügung.")
    ['Das',
     'Gluon',
     'NLP-Toolkit',
     'stellt',
     'eine',
     'Reihe',
     'von',
     'Textverarbeitungstools',
     'zur',
     'Verfügung',
     '.']
    """
    def __init__(self, lang='en'):
        try:
            import spacy
            from pkg_resources import parse_version
            assert parse_version(spacy.__version__) >= parse_version('2.0.0'),\
                'We only support spacy>=2.0.0'
        except ImportError:
            raise ImportError('spaCy is not installed. You must install spaCy in order to use the '
                              'SpacyTokenizer. You can refer to the official installation guide '
                              'in https://spacy.io/usage/.')
        try:
            self._nlp = spacy.load(lang, disable=['parser', 'tagger', 'ner'])
        except IOError:
            raise IOError('SpaCy Model for the specified language="{lang}" has not been '
                          'downloaded. You need to check the installation guide in '
                          'https://spacy.io/usage/models. Usually, the installation command '
                          'should be `python -m spacy download {lang}`.'.format(lang=lang))

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample: str
            The sentence to tokenize

        Returns
        -------
        ret : list of strs
            List of tokens
        """
        return [tok.text for tok in self._nlp(sample)]
