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

__all__ = ['ClipSequence', 'PadSequence', 'SacreMosesTokenizer', 'NLTKMosesTokenizer',
           'SpacyTokenizer', 'SacreMosesDetokenizer', 'NLTKMosesDetokenizer', 'JiebaTokenizer',
           'NLTKStanfordSegmenter', 'SentencepieceTokenizer', 'SentencepieceDetokenizer']

import os
import warnings

import numpy as np
import mxnet as mx
from mxnet.gluon.utils import download, check_sha1
from .utils import _get_home_dir, _extract_archive


class ClipSequence(object):
    """Clip the sequence to have length no more than `length`.

    Parameters
    ----------
    length : int
        Maximum length of the sequence

    Examples
    --------
    >>> datasets = gluon.data.SimpleDataset([[1, 3, 5, 7], [1, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8]])
    >>> list(datasets.transform(gluonnlp.data.ClipSequence(4)))
    [[1, 3, 5, 7], [1, 2, 3], [1, 2, 3, 4]]
    >>> datasets = gluon.data.SimpleDataset([np.array([[1, 3], [5, 7], [7, 5], [3, 1]]),
    ...                                      np.array([[1, 2], [3, 4], [5, 6],
    ...                                                [6, 5], [4, 3], [2, 1]]),
    ...                                      np.array([[2, 4], [4, 2]])])
    >>> list(datasets.transform(gluonnlp.data.ClipSequence(3)))
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
    >>> datasets = gluon.data.SimpleDataset([[1, 3, 5, 7], [1, 2, 3], [1, 2, 3, 4, 5, 6, 7, 8]])
    >>> list(datasets.transform(gluonnlp.data.PadSequence(6)))
    [[1, 3, 5, 7, 0, 0], [1, 2, 3, 0, 0, 0], [1, 2, 3, 4, 5, 6]]
    >>> list(datasets.transform(gluonnlp.data.PadSequence(6, clip=False)))
    [[1, 3, 5, 7, 0, 0], [1, 2, 3, 0, 0, 0], [1, 2, 3, 4, 5, 6, 7, 8]]
    >>> list(datasets.transform(gluonnlp.data.PadSequence(6, pad_val=-1, clip=False)))
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
    """Apply the Moses Tokenizer implemented in NLTK.

    Users of this class are required to install `NLTK <https://www.nltk.org/install.html>`_
    and install relevant NLTK packages, such as
    :samp:`python -m nltk.downloader perluniprops nonbreaking_prefixes`.

    Examples
    --------
    >>> tokenizer = gluonnlp.data.NLTKMosesTokenizer()
    >>> tokenizer("Gluon NLP toolkit provides a suite of text processing tools.")
    ['Gluon', 'NLP', 'toolkit', 'provides', 'a', 'suite', 'of', 'text', 'processing', 'tools', '.']
    >>> tokenizer("Das Gluon NLP-Toolkit stellt eine Reihe von Textverarbeitungstools "
    ...           "zur Verfügung.")
    ['Das', 'Gluon', 'NLP-Toolkit', 'stellt', 'eine', 'Reihe', 'von', 'Textverarbeitungstools', \
'zur', 'Verf\xfcgung', '.']
    """
    def __init__(self):
        try:
            from nltk.tokenize.moses import MosesTokenizer
        except ImportError:
            warnings.warn('NLTK or relevant packages are not installed. '
                          'Due to the LGPL 2.1+, moses has been deprecated in NLTK since 3.3.0. '
                          'You must install NLTK <= 3.2.5 in order to use the '
                          'NLTKMosesTokenizer. You can refer to the official '
                          'installation guide in https://www.nltk.org/install.html .'
                          ' Now try SacreMosesTokenizer using sacremoses ...')
            try:
                from sacremoses import MosesTokenizer
            except ImportError:
                raise ImportError('sacremoses is also not installed. '
                                  'Please use sacremoses or older nltk version, e.g. 3.2.5. '
                                  'To install sacremoses, use pip install -U sacremoses')
        try:
            self._tokenizer = MosesTokenizer()
        except ValueError:
            raise ValueError('The instantiation of MosesTokenizer in sacremoses is'
                             ' currently only supported in python3.')

    def __call__(self, sample, return_str=False):
        """

        Parameters
        ----------
        sample: str
            The sentence to tokenize
        return_str: bool, default False
            True: return a single string
            False: return a list of tokens

        Returns
        -------
        ret : list of strs or str
            List of tokens or tokenized text
        """
        return self._tokenizer.tokenize(sample, return_str=return_str)


class SacreMosesTokenizer(object):
    """Apply the Moses Tokenizer implemented in sacremoses.

    Users of this class are required to install
    `sacremoses <https://github.com/alvations/sacremoses>`_.
    For example, one can use :samp:`pip install sacremoses`.

    .. note::
        sacremoses carries an LGPL 2.1+ license.

    Examples
    --------
    >>> tokenizer = gluonnlp.data.SacreMosesTokenizer()
    >>> tokenizer("Gluon NLP toolkit provides a suite of text processing tools.")
    ['Gluon', 'NLP', 'toolkit', 'provides', 'a', 'suite', 'of', 'text', 'processing', 'tools', '.']
    >>> tokenizer("Das Gluon NLP-Toolkit stellt eine Reihe von Textverarbeitungstools "
    ...           "zur Verfügung.")
    ['Das', 'Gluon', 'NLP-Toolkit', 'stellt', 'eine', 'Reihe', 'von', 'Textverarbeitungstools', \
'zur', 'Verf\xfcgung', '.']
    """
    def __init__(self):
        try:
            from sacremoses import MosesTokenizer
            self._tokenizer = MosesTokenizer()
        except (ImportError, TypeError) as err:
            if isinstance(err, TypeError):
                warnings.warn('The instantiation of MosesTokenizer in sacremoses is'
                              ' currently only supported in python3.'
                              ' Now try NLTKMosesTokenizer using NLTK ...')
            else:
                warnings.warn('sacremoses is not installed. '
                              'To install sacremoses, use pip install -U sacremoses'
                              ' Now try NLTKMosesTokenizer using NLTK ...')
            try:
                from nltk.tokenize.moses import MosesTokenizer
                self._tokenizer = MosesTokenizer()
            except ImportError:
                raise ImportError('NLTK is also not installed. '
                                  'You must install NLTK <= 3.2.5 in order to use the '
                                  'NLTKMosesTokenizer. You can refer to the official '
                                  'installation guide in https://www.nltk.org/install.html .')

    def __call__(self, sample, return_str=False):
        """

        Parameters
        ----------
        sample: str
            The sentence to tokenize
        return_str: bool, default False
            True: return a single string
            False: return a list of tokens

        Returns
        -------
        ret : list of strs or str
            List of tokens or tokenized text
        """
        return self._tokenizer.tokenize(sample, return_str=return_str)


class SpacyTokenizer(object):
    """Apply the Spacy Tokenizer.

    Users of this class are required to install `spaCy <https://spacy.io/usage/>`_
    and download corresponding NLP models, such as :samp:`python -m spacy download en`.

    Only spacy>=2.0.0 is supported.

    Parameters
    ----------
    lang : str
        The language to tokenize. Default is "en", i.e, English.
        You may refer to https://spacy.io/usage/models for supported languages.

    Examples
    --------
    >>> tokenizer = gluonnlp.data.SpacyTokenizer()
    >>> tokenizer(u"Gluon NLP toolkit provides a suite of text processing tools.")
    ['Gluon', 'NLP', 'toolkit', 'provides', 'a', 'suite', 'of', 'text', 'processing', 'tools', '.']
    >>> tokenizer = gluonnlp.data.SpacyTokenizer('de')
    >>> tokenizer(u"Das Gluon NLP-Toolkit stellt eine Reihe von Textverarbeitungstools"
    ...            " zur Verfügung.")
    ['Das', 'Gluon', 'NLP-Toolkit', 'stellt', 'eine', 'Reihe', 'von', 'Textverarbeitungstools', \
'zur', 'Verf\xfcgung', '.']
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


class NLTKMosesDetokenizer(object):
    r"""Apply the Moses Detokenizer implemented in NLTK.

    Users of this class are required to `install NLTK <https://www.nltk.org/install.html>`_
    and install relevant NLTK packages, such as
    :samp:`python -m nltk.downloader perluniprops nonbreaking_prefixes`

    Examples
    --------
    >>> detokenizer = gluonnlp.data.NLTKMosesDetokenizer()
    >>> detokenizer(['Gluon', 'NLP', 'toolkit', 'provides', 'a', 'suite', 'of',
    ...              'text', 'processing', 'tools', '.'], return_str=True)
    'Gluon NLP toolkit provides a suite of text processing tools.'
    >>> detokenizer(['Das', 'Gluon','NLP-Toolkit','stellt','eine','Reihe','von',
    ...              'Textverarbeitungstools','zur','Verfügung','.'], return_str=True)
    'Das Gluon NLP-Toolkit stellt eine Reihe von Textverarbeitungstools zur Verfügung.'
    """
    def __init__(self):
        try:
            from nltk.tokenize.moses import MosesDetokenizer
        except ImportError:
            warnings.warn('NLTK or relevant packages are not installed. '
                          'Due to the LGPL 2.1+, moses has been deprecated in NLTK since 3.3.0. '
                          'You must install NLTK <= 3.2.5 in order to use the '
                          'NLTKMosesDetokenizer. You can refer to the official '
                          'installation guide in https://www.nltk.org/install.html .'
                          ' Now try SacreMosesDetokenizer using sacremoses ...')
            try:
                from sacremoses import MosesDetokenizer
            except ImportError:
                raise ImportError('sacremoses is also not installed. '
                                  'Please use sacremoses or older nltk version, e.g. 3.2.5. '
                                  'To install sacremoses, use pip install -U sacremoses')
        try:
            self._detokenizer = MosesDetokenizer()
        except ValueError:
            raise ValueError('The instantiation of MosesDetokenizer in sacremoses is'
                             ' currently only supported in python3.')

    def __call__(self, sample, return_str=False):
        """

        Parameters
        ----------
        sample: list(str)
            The sentence to detokenize
        return_str: bool, default False
            True: return a single string
            False: return a list of words

        Returns
        -------
        ret : list of strs or str
            List of words or detokenized text
        """
        return self._detokenizer.detokenize(sample, return_str=return_str)


class SacreMosesDetokenizer(object):
    r"""Apply the Moses Detokenizer implemented in sacremoses.

    Users of this class are required to `install sacremoses
    <https://github.com/alvations/sacremoses>`_. For example, one can use
    :samp:`pip install sacremoses`.

    .. note::
        sacremoses carries an LGPL 2.1+ license.

    Examples
    --------
    >>> detokenizer = gluonnlp.data.SacreMosesDetokenizer()
    >>> detokenizer(['Gluon', 'NLP', 'toolkit', 'provides', 'a', 'suite', 'of',
    ...              'text', 'processing', 'tools', '.'], return_str=True)
    'Gluon NLP toolkit provides a suite of text processing tools.'
    >>> detokenizer(['Das', 'Gluon','NLP-Toolkit','stellt','eine','Reihe','von',
    ...              'Textverarbeitungstools','zur','Verfügung','.'], return_str=True)
    'Das Gluon NLP-Toolkit stellt eine Reihe von Textverarbeitungstools zur Verfügung.'
    """
    def __init__(self):
        try:
            from sacremoses import MosesDetokenizer
            self._detokenizer = MosesDetokenizer()
        except (ImportError, TypeError) as err:
            if isinstance(err, TypeError):
                warnings.warn('The instantiation of MosesDetokenizer in sacremoses is'
                              ' currently only supported in python3.'
                              ' Now try NLTKMosesDetokenizer using NLTK ...')
            else:
                warnings.warn('sacremoses is not installed. '
                              'To install sacremoses, use pip install -U sacremoses'
                              ' Now try NLTKMosesDetokenizer using NLTK ...')
            try:
                import nltk
                try:
                    nltk.data.find('perluniprops')
                except LookupError:
                    nltk.download('perluniprops')
                from nltk.tokenize.moses import MosesDetokenizer
                self._detokenizer = MosesDetokenizer()
            except ImportError:
                raise ImportError('NLTK is not installed. '
                                  'You must install NLTK <= 3.2.5 in order to use the '
                                  'NLTKMosesDetokenizer. You can refer to the official '
                                  'installation guide in https://www.nltk.org/install.html .')

    def __call__(self, sample, return_str=False):
        """

        Parameters
        ----------
        sample: list(str)
            The sentence to detokenize
        return_str: bool, default False
            True: return a single string
            False: return a list of words

        Returns
        -------
        ret : list of strs or str
            List of words or detokenized text
        """
        return self._detokenizer.detokenize(sample, return_str=return_str)


class JiebaTokenizer(object):
    r"""Apply the jieba Tokenizer.

    Users of this class are required to `install jieba <https://github.com/fxsjy/jieba>`_

    Parameters
    ----------
    lang : str
        The language to tokenize. Default is "zh", i.e, Chinese.

    Examples
    --------
    >>> tokenizer = gluonnlp.data.JiebaTokenizer()
    >>> tokenizer(u"我来到北京清华大学")
    ['我', '来到', '北京', '清华大学']
    >>> tokenizer(u"小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
    ['小明', '硕士', '毕业', '于', '中国科学院', '计算所', '，', '后', '在', '日本京都大学', '深造']

    """
    def __init__(self):
        try:
            import jieba
        except ImportError:
            raise ImportError('jieba is not installed. You must install jieba in order to use the '
                              'JiebaTokenizer. You can refer to the official installation guide '
                              'in https://github.com/fxsjy/jieba')
        self._tokenizer = jieba

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample: str
            The Chinese sentence to tokenize. Better not to input sentence in other languages
            since this class is mainly used for Chinese Word Segmentation.

        Returns
        -------
        ret : list of strs
            List of tokens
        """
        # we use default cutting mode provided by jieba, i.e., accurate mode
        return [tok for tok in self._tokenizer.cut(sample) if tok != ' ' and tok != '']


class NLTKStanfordSegmenter(object):
    r"""Apply the Stanford Chinese Word Segmenter implemented in NLTK.

    Users of this class are required to install Java, NLTK and download Stanford Word Segmenter

    Parameters
    ----------
    segmenter_root : str, default '$MXNET_HOME/stanford-segmenter'
        Path to folder for storing stanford segmenter.
        MXNET_HOME defaults to '~/.mxnet'.

    slf4j_root : str, default '$MXNET_HOME/slf4j'
        Path to foler for storing slf4j.
        MXNET_HOME defaults to '~/.mxnet'

    java_class : str, default 'edu.stanford.nlp.ie.crf.CRFClassifier'
        The learning algorithm used for segmentation

    Examples
    --------
    >>> tokenizer = gluonnlp.data.NLTKStanfordSegmenter() #doctest:+SKIP
    >>> tokenizer(u"我来到北京清华大学")
    ['我', '来到', '北京', '清华大学']
    >>> tokenizer(u"小明硕士毕业于中国科学院计算所，后在日本京都大学深造")
    ['小明', '硕士', '毕业', '于', '中国科学院', '计算所', '，', '后', '在', '日本京都大学', '深造']

    """
    def __init__(self, segmenter_root=os.path.join(_get_home_dir(), 'stanford-segmenter'),
                 slf4j_root=os.path.join(_get_home_dir(), 'slf4j'),
                 java_class='edu.stanford.nlp.ie.crf.CRFClassifier'):
        is_java_exist = os.system('java -version')
        assert is_java_exist == 0, 'Java is not installed. You must install Java 8.0' \
                                   'in order to use the NLTKStanfordSegmenter'
        try:
            from nltk.tokenize import StanfordSegmenter
        except ImportError:
            raise ImportError('NLTK or relevant packages are not installed. You must install NLTK '
                              'in order to use the NLTKStanfordSegmenter. You can refer to the '
                              'official installation guide in https://www.nltk.org/install.html.')
        path_to_jar = os.path.join(segmenter_root, 'stanford-segmenter-2018-02-27',
                                   'stanford-segmenter-3.9.1.jar')
        path_to_model = os.path.join(segmenter_root, 'stanford-segmenter-2018-02-27',
                                     'data', 'pku.gz')
        path_to_dict = os.path.join(segmenter_root, 'stanford-segmenter-2018-02-27',
                                    'data', 'dict-chris6.ser.gz')
        path_to_sihan_corpora_dict = os.path.join(segmenter_root,
                                                  'stanford-segmenter-2018-02-27', 'data')
        segmenter_url = 'https://nlp.stanford.edu/software/stanford-segmenter-2018-02-27.zip'
        segmenter_sha1 = 'aa27a6433704b7b4c6a14be1c126cb4b14b8f57b'
        stanford_segmenter = os.path.join(segmenter_root, 'stanford-segmenter-2018-02-27.zip')
        if not os.path.exists(path_to_jar) or \
                not os.path.exists(path_to_model) or \
                not os.path.exists(path_to_dict) or \
                not os.path.exists(path_to_sihan_corpora_dict) or \
                not check_sha1(filename=stanford_segmenter, sha1_hash=segmenter_sha1):
            # automatically download the files from the website and place them to stanford_root
            if not os.path.exists(segmenter_root):
                os.mkdir(segmenter_root)
            download(url=segmenter_url, path=segmenter_root, sha1_hash=segmenter_sha1)
            _extract_archive(file=stanford_segmenter, target_dir=segmenter_root)

        path_to_slf4j = os.path.join(slf4j_root, 'slf4j-1.7.25', 'slf4j-api-1.7.25.jar')
        slf4j_url = 'https://www.slf4j.org/dist/slf4j-1.7.25.zip'
        slf4j_sha1 = '89ea41ad6ebe1b190139421bb7c8d981e9df1625'
        slf4j = os.path.join(slf4j_root, 'slf4j-1.7.25.zip')
        if not os.path.exists(path_to_slf4j) or \
                not check_sha1(filename=slf4j, sha1_hash=slf4j_sha1):
            # automatically download the files from the website and place them to slf4j_root
            if not os.path.exists(slf4j_root):
                os.mkdir(slf4j_root)
            download(url=slf4j_url, path=slf4j_root, sha1_hash=slf4j_sha1)
            _extract_archive(file=slf4j, target_dir=slf4j_root)
        self._tokenizer = StanfordSegmenter(java_class=java_class, path_to_jar=path_to_jar,
                                            path_to_slf4j=path_to_slf4j, path_to_dict=path_to_dict,
                                            path_to_sihan_corpora_dict=path_to_sihan_corpora_dict,
                                            path_to_model=path_to_model)

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample: str
            The Chinese sentence to tokenize. Better not to input sentence in other languages
            since this class is mainly used for Chinese Word Segmentation.

        Returns
        -------
        ret : list of strs
            List of tokens
        """
        return [tok for tok in self._tokenizer.segment(sample).strip().split()]


class _SentencepieceProcessor(object):
    def __init__(self, path):
        try:
            import sentencepiece
        except ImportError:
            raise ImportError('sentencepiece is not installed. You must install sentencepiece '
                              'in order to use the Sentencepiece tokenizer and detokenizer. '
                              'You can refer to the official installation guide '
                              'in https://github.com/google/sentencepiece#installation')
        self._processor = sentencepiece.SentencePieceProcessor()
        self._processor.Load(path)

    def __len__(self):
        return len(self._processor)

    @property
    def tokens(self):
        return [self._processor.id_to_piece(i) for i in range(len(self))]

class SentencepieceTokenizer(_SentencepieceProcessor):
    r"""Apply the Sentencepiece Tokenizer, which supports subword tokenization such as BPE.

    Users of this class are required to `install sentencepiece
    <https://github.com/google/sentencepiece>`_. For example, one can use
    :samp:`pip install sentencepiece`


    Parameters
    ----------
    path : str
        Path to the pre-trained subword tokenization model.
    num_best : int, default 0
        A scalar for sampling subwords. If num_best = {0,1}, no sampling is performed.
        If num_best > 1, then samples from the num_best results.
        If num_best < 0, then assume that num_best is infinite and
        samples from the all hypothesis (lattice) using forward-filtering-and-backward-sampling
        algorithm.
    alpha : float
        A scalar for a smoothing parameter. Inverse temparature for probablity rescaling.

    Examples
    --------
    >>> url = 'http://repo.mxnet.io/gluon/dataset/vocab/test-0690baed.bpe'
    >>> f = gluon.utils.download(url, overwrite=True)
    -etc-
    >>> tokenizer = gluonnlp.data.SentencepieceTokenizer(f)
    >>> detokenizer = gluonnlp.data.SentencepieceDetokenizer(f)
    >>> sentence = 'This is a very awesome, life-changing sentence.'
    >>> tokenizer(sentence)
    ['▁This', '▁is', '▁a', '▁very', '▁awesome', ',', '▁life', '-', 'ch', 'anging', '▁sentence', '.']
    >>> detokenizer(tokenizer(sentence))
    'This is a very awesome, life-changing sentence.'

    """
    def __init__(self, path, num_best=0, alpha=1.0):
        super(SentencepieceTokenizer, self).__init__(path)
        self._nbest = num_best
        self._alpha = alpha

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample: str
            The string to tokenize.

        Returns
        -------
        ret : list of strs
            List of tokens
        """
        return self._processor.SampleEncodeAsPieces(sample, self._nbest, self._alpha)


class SentencepieceDetokenizer(_SentencepieceProcessor):
    r"""Apply the Sentencepiece detokenizer, which supports recombining subwords such as BPE.

    Users of this class are required to `install sentencepiece
    <https://github.com/google/sentencepiece>`_. For example, one can use
    :samp:`pip install sentencepiece`


    Parameters
    ----------
    path : str
        Path to the pre-trained subword tokenization model.

    Examples
    --------
    >>> url = 'http://repo.mxnet.io/gluon/dataset/vocab/test-0690baed.bpe'
    >>> f = gluon.utils.download(url, overwrite=True)
    -etc-
    >>> tokenizer = gluonnlp.data.SentencepieceTokenizer(f)
    >>> detokenizer = gluonnlp.data.SentencepieceDetokenizer(f)
    >>> sentence = 'This is a very awesome, life-changing sentence.'
    >>> tokenizer(sentence)
    ['▁This', '▁is', '▁a', '▁very', '▁awesome', ',', '▁life', '-', 'ch', 'anging', '▁sentence', '.']
    >>> detokenizer(tokenizer(sentence))
    'This is a very awesome, life-changing sentence.'

    """
    def __call__(self, sample):
        """

        Parameters
        ----------
        sample: list(str)
            The sentence to detokenize

        Returns
        -------
        ret : str
            Detokenized text
        """
        return self._processor.DecodePieces(sample)
