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

# pylint: disable=invalid-encoded-data, too-many-lines
"""Transformer API. It provides tools for common transformation on samples in text dataset, such as
clipping, padding, and tokenization."""


__all__ = [
    'ClipSequence', 'PadSequence', 'SacreMosesTokenizer',
    'SpacyTokenizer', 'SacreMosesDetokenizer',
    'JiebaTokenizer', 'NLTKStanfordSegmenter', 'SentencepieceTokenizer',
    'SentencepieceDetokenizer', 'BERTBasicTokenizer', 'BERTTokenizer',
    'BERTSentenceTransform', 'BERTSPTokenizer',
    'GPT2BPETokenizer', 'GPT2BPEDetokenizer'
]

import functools
import io
import os
import time
import unicodedata
import warnings
import zipfile
from typing import List, Optional

import mxnet as mx
from mxnet.gluon.utils import _get_repo_url, check_sha1, download
import numpy as np

from ..base import get_home_dir
from ..vocab.vocab import Vocab
from .utils import _extract_archive
from .fast_bert_tokenizer import is_control, is_punctuation, is_whitespace
from .fast_bert_tokenizer import BasicTokenizer, WordpieceTokenizer


class ClipSequence:
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


class PadSequence:
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
                new_sample_shape = (self._length, ) + sample.shape[1:]
                ret = mx.nd.full(shape=new_sample_shape,
                                 val=self._pad_val,
                                 ctx=sample.context,
                                 dtype=sample.dtype)
                ret[:sample_length] = sample
                return ret
            elif isinstance(sample, np.ndarray):
                pad_width = [(0, self._length - sample_length)] +\
                            [(0, 0) for _ in range(sample.ndim - 1)]
                return np.pad(sample,
                              mode='constant',
                              constant_values=self._pad_val,
                              pad_width=pad_width)
            elif isinstance(sample, list):
                return sample + [
                    self._pad_val for _ in range(self._length - sample_length)
                ]
            else:
                raise NotImplementedError(
                    'The input must be 1) list or 2) numpy.ndarray or 3) '
                    'mxnet.NDArray, received type=%s' % str(type(sample)))


class SacreMosesTokenizer:
    """Apply the Moses Tokenizer implemented in sacremoses.

    Users of this class are required to install
    `sacremoses <https://github.com/alvations/sacremoses>`_.
    For example, one can use :samp:`pip install sacremoses`.

    .. note::
        sacremoses carries an LGPL 2.1+ license.

    Examples
    --------
    >>> tokenizer = gluonnlp.data.SacreMosesTokenizer()
    >>> tokenizer('Gluon NLP toolkit provides a suite of text processing tools.')
    ['Gluon', 'NLP', 'toolkit', 'provides', 'a', 'suite', 'of', 'text', 'processing', 'tools', '.']
    >>> tokenizer('Das Gluon NLP-Toolkit stellt eine Reihe von Textverarbeitungstools '
    ...           'zur Verfügung.')
    ['Das', 'Gluon', 'NLP-Toolkit', 'stellt', 'eine', 'Reihe', 'von', 'Textverarbeitungstools', \
'zur', 'Verf\xfcgung', '.']
    """

    def __init__(self):
        from sacremoses import MosesTokenizer  # pylint: disable=import-outside-toplevel
        self._tokenizer = MosesTokenizer()

    def __call__(self, sample: str, return_str: bool = False):
        """Tokenize a sample.

        Parameters
        ----------
        sample
            The sentence to tokenize
        return_str
            True: return a single string
            False: return a list of tokens

        Returns
        -------
        ret : list of strs or str
            List of tokens or tokenized text
        """
        return self._tokenizer.tokenize(sample, return_str=return_str)


class SpacyTokenizer:
    """Apply the Spacy Tokenizer.

    Users of this class are required to install `spaCy <https://spacy.io/usage/>`_
    and download corresponding NLP models, such as :samp:`python -m spacy download en`.

    Only spacy>=2.0.0 is supported.

    Parameters
    ----------
    lang : str
        The language to tokenize. Default is 'en', i.e, English.
        You may refer to https://spacy.io/usage/models for supported languages.

    Examples
    --------
    >>> tokenizer = gluonnlp.data.SpacyTokenizer()
    >>> tokenizer('Gluon NLP toolkit provides a suite of text processing tools.')
    ['Gluon', 'NLP', 'toolkit', 'provides', 'a', 'suite', 'of', 'text', 'processing', 'tools', '.']
    >>> tokenizer = gluonnlp.data.SpacyTokenizer('de')
    >>> tokenizer('Das Gluon NLP-Toolkit stellt eine Reihe von Textverarbeitungstools'
    ...           ' zur Verfügung.')
    ['Das', 'Gluon', 'NLP-Toolkit', 'stellt', 'eine', 'Reihe', 'von', 'Textverarbeitungstools', \
'zur', 'Verf\xfcgung', '.']
    """

    def __init__(self, lang='en_core_web_sm'):
        try:
            import spacy  # pylint: disable=import-outside-toplevel
            from pkg_resources import parse_version  # pylint: disable=import-outside-toplevel
            assert parse_version(spacy.__version__) >= parse_version('2.0.0'),\
                'We only support spacy>=2.0.0'
        except ImportError:
            raise ImportError(
                'spaCy is not installed. You must install spaCy in order to use the '
                'SpacyTokenizer. You can refer to the official installation guide '
                'in https://spacy.io/usage/.')
        try:
            self._nlp = spacy.load(lang, disable=['parser', 'tagger', 'ner'])
        except IOError:
            raise IOError(
                'SpaCy Model for the specified language="{lang}" has not been '
                'downloaded. You need to check the installation guide in '
                'https://spacy.io/usage/models. Usually, the installation command '
                'should be `python -m spacy download {lang}`.'.format(
                    lang=lang))

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


class SacreMosesDetokenizer:
    r"""Apply the Moses Detokenizer implemented in sacremoses.

    Users of this class are required to `install sacremoses
    <https://github.com/alvations/sacremoses>`_. For example, one can use
    :samp:`pip install sacremoses`.

    .. note::
        sacremoses carries an LGPL 2.1+ license.

    Parameters
    ----------
    return_str: bool, default False
        True: return a single string
        False: return a list of words

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

    def __init__(self, return_str=True):
        self._return_str = return_str
        from sacremoses import MosesDetokenizer  # pylint: disable=import-outside-toplevel
        self._detokenizer = MosesDetokenizer()

    def __call__(self, sample: List[str], return_str: Optional[bool] = None):
        """

        Parameters
        ----------
        sample
            The sentence to detokenize
        return_str
            True: return a single string
            False: return a list of words
            None: use constructor setting

        Returns
        -------
        ret : list of strs or str
            List of words or detokenized text
        """
        ret_str = self._return_str if return_str is None else return_str
        return self._detokenizer.detokenize(sample, return_str=ret_str)


class JiebaTokenizer:
    r"""Apply the jieba Tokenizer.

    Users of this class are required to `install jieba <https://github.com/fxsjy/jieba>`_

    Parameters
    ----------
    lang : str
        The language to tokenize. Default is "zh", i.e, Chinese.

    Examples
    --------
    >>> tokenizer = gluonnlp.data.JiebaTokenizer()
    >>> tokenizer('我来到北京清华大学')
    ['我', '来到', '北京', '清华大学']
    >>> tokenizer('小明硕士毕业于中国科学院计算所，后在日本京都大学深造')
    ['小明', '硕士', '毕业', '于', '中国科学院', '计算所', '，', '后', '在', '日本京都大学', '深造']

    """

    def __init__(self):
        try:
            with warnings.catch_warnings():  # jieba uses deprecated imp module
                warnings.simplefilter('ignore')
                import jieba  # pylint: disable=import-outside-toplevel
        except ImportError:
            raise ImportError(
                'jieba is not installed. You must install jieba in order to use the '
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
        return [
            tok for tok in self._tokenizer.cut(sample)
            if tok not in (' ', '')
        ]


class NLTKStanfordSegmenter:
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
    >>> tokenizer('我来到北京清华大学') #doctest:+SKIP
    ['我', '来到', '北京', '清华大学']
    >>> tokenizer('小明硕士毕业于中国科学院计算所，后在日本京都大学深造') #doctest:+SKIP
    ['小明', '硕士', '毕业', '于', '中国科学院', '计算所', '，', '后', '在', '日本京都大学', '深造']

    """

    def __init__(self,
                 segmenter_root=os.path.join(get_home_dir(),
                                             'stanford-segmenter'),
                 slf4j_root=os.path.join(get_home_dir(), 'slf4j'),
                 java_class='edu.stanford.nlp.ie.crf.CRFClassifier'):
        is_java_exist = os.system('java -version')
        assert is_java_exist == 0, 'Java is not installed. You must install Java 8.0' \
                                   'in order to use the NLTKStanfordSegmenter'
        try:
            from nltk.tokenize import StanfordSegmenter  # pylint: disable=import-outside-toplevel
        except ImportError:
            raise ImportError(
                'NLTK or relevant packages are not installed. You must install NLTK '
                'in order to use the NLTKStanfordSegmenter. You can refer to the '
                'official installation guide in https://www.nltk.org/install.html.'
            )
        path_to_jar = os.path.join(segmenter_root,
                                   'stanford-segmenter-2018-02-27',
                                   'stanford-segmenter-3.9.1.jar')
        path_to_model = os.path.join(segmenter_root,
                                     'stanford-segmenter-2018-02-27', 'data',
                                     'pku.gz')
        path_to_dict = os.path.join(segmenter_root,
                                    'stanford-segmenter-2018-02-27', 'data',
                                    'dict-chris6.ser.gz')
        path_to_sihan_corpora_dict = os.path.join(
            segmenter_root, 'stanford-segmenter-2018-02-27', 'data')
        segmenter_url = 'https://nlp.stanford.edu/software/stanford-segmenter-2018-02-27.zip'
        segmenter_sha1 = 'aa27a6433704b7b4c6a14be1c126cb4b14b8f57b'
        stanford_segmenter = os.path.join(segmenter_root,
                                          'stanford-segmenter-2018-02-27.zip')
        if not os.path.exists(path_to_jar) or \
                not os.path.exists(path_to_model) or \
                not os.path.exists(path_to_dict) or \
                not os.path.exists(path_to_sihan_corpora_dict) or \
                not check_sha1(filename=stanford_segmenter, sha1_hash=segmenter_sha1):
            # automatically download the files from the website and place them to stanford_root
            if not os.path.exists(segmenter_root):
                os.mkdir(segmenter_root)
            download(url=segmenter_url,
                     path=segmenter_root,
                     sha1_hash=segmenter_sha1)
            _extract_archive(file=stanford_segmenter,
                             target_dir=segmenter_root)

        path_to_slf4j = os.path.join(slf4j_root, 'slf4j-1.7.25',
                                     'slf4j-api-1.7.25.jar')
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
        self._tokenizer = StanfordSegmenter(
            java_class=java_class,
            path_to_jar=path_to_jar,
            path_to_slf4j=path_to_slf4j,
            path_to_dict=path_to_dict,
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
        return self._tokenizer.segment(sample).strip().split()


class _SentencepieceProcessor:
    def __init__(self, path):
        try:
            import sentencepiece  # pylint: disable=import-outside-toplevel
        except ImportError:
            raise ImportError(
                'sentencepiece is not installed. You must install sentencepiece '
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
    alpha : float, default 1.0
        A scalar for a smoothing parameter. Inverse temperature for probability rescaling.

    Examples
    --------
    >>> url = 'http://repo.mxnet.io/gluon/dataset/vocab/test-0690baed.bpe'
    >>> f = gluon.utils.download(url)
    -etc-
    >>> tokenizer = gluonnlp.data.SentencepieceTokenizer(f)
    >>> detokenizer = gluonnlp.data.SentencepieceDetokenizer(f)
    >>> sentence = 'This is a very awesome, life-changing sentence.'
    >>> tokenizer(sentence)
    ['▁This', '▁is', '▁a', '▁very', '▁awesome', ',', '▁life', '-', 'ch', 'anging', '▁sentence', '.']
    >>> detokenizer(tokenizer(sentence))
    'This is a very awesome, life-changing sentence.'
    >>> os.remove('test-0690baed.bpe')

    """

    def __init__(self, path, num_best=0, alpha=1.0):
        super(SentencepieceTokenizer, self).__init__(path)
        self._path = path
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
        return self._processor.SampleEncodeAsPieces(sample, self._nbest,
                                                    self._alpha)


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
    >>> f = gluon.utils.download(url)
    -etc-
    >>> tokenizer = gluonnlp.data.SentencepieceTokenizer(f)
    >>> detokenizer = gluonnlp.data.SentencepieceDetokenizer(f)
    >>> sentence = 'This is a very awesome, life-changing sentence.'
    >>> tokenizer(sentence)
    ['▁This', '▁is', '▁a', '▁very', '▁awesome', ',', '▁life', '-', 'ch', 'anging', '▁sentence', '.']
    >>> detokenizer(tokenizer(sentence))
    'This is a very awesome, life-changing sentence.'
    >>> os.remove('test-0690baed.bpe')

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


class BERTBasicTokenizer:
    r"""Runs basic tokenization

    performs invalid character removal (e.g. control chars) and whitespace.
    tokenize CJK chars.
    splits punctuation on a piece of text.
    strips accents and convert to lower case.(If lower is true)

    Parameters
    ----------
    lower : bool, default True
        whether the text strips accents and convert to lower case.

    Examples
    --------
    >>> tokenizer = gluonnlp.data.BERTBasicTokenizer(lower=True)
    >>> tokenizer(' \tHeLLo!how  \n Are yoU?  ')
    ['hello', '!', 'how', 'are', 'you', '?']
    >>> tokenizer = gluonnlp.data.BERTBasicTokenizer(lower=False)
    >>> tokenizer(' \tHeLLo!how  \n Are yoU?  ')
    ['HeLLo', '!', 'how', 'Are', 'yoU', '?']

    """

    def __init__(self, lower=True):
        self.tokenizer = BasicTokenizer(lower=lower)

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample:  str
            The string to tokenize. Must be unicode.

        Returns
        -------
        ret : list of strs
            List of tokens
        """
        return self.tokenizer.tokenize(sample)

    def _is_control(self, char):
        """Checks whether `chars` is a control character."""
        return is_control(char, unicodedata.category(char))

    def _is_punctuation(self, char):
        """Checks whether `chars` is a punctuation character."""
        return is_punctuation(char, unicodedata.category(char))

    def _is_whitespace(self, char):
        """Checks whether `chars` is a whitespace character."""
        return is_whitespace(char, unicodedata.category(char))


class BERTTokenizer:
    r"""End-to-end tokenization for BERT models.

    Parameters
    ----------
    vocab
        Vocabulary for the corpus.
    lower
        whether the text strips accents and convert to lower case.
        If you use the BERT pre-training model,
        lower is set to Flase when using the cased model,
        otherwise it is set to True.
    max_input_chars_per_word
    lru_cache_size
        Maximum size of a least-recently-used cache to speed up tokenization.
        Use size of 2**20 for example.

    Examples
    --------
    >>> _, vocab = gluonnlp.model.bert_12_768_12(dataset_name='wiki_multilingual_uncased',
    ...                                          pretrained=False, root='./model')
    -etc-
    >>> tokenizer = gluonnlp.data.BERTTokenizer(vocab=vocab)
    >>> tokenizer('gluonnlp: 使NLP变得简单。')
    ['gl', '##uo', '##nn', '##lp', ':', '使', 'nl', '##p', '变', '得', '简', '单', '。']

    """

    _special_prefix = '##'

    def __init__(self, vocab: Vocab, lower: bool = True, max_input_chars_per_word: int = 200,
                 lru_cache_size: Optional[int] = None):
        self.vocab = vocab
        self.max_input_chars_per_word = max_input_chars_per_word
        self.basic_tokenizer = BasicTokenizer(lower=lower)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=vocab,
                                                      unk_token=vocab.unknown_token,
                                                      max_input_chars_per_word=\
                                                              max_input_chars_per_word)

        if lru_cache_size:
            self._word_to_wordpiece_optimized = functools.lru_cache(maxsize=lru_cache_size)(
                self._word_to_wordpiece_optimized)

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

        return self._tokenizer(sample)

    def _tokenizer(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self._word_to_wordpiece_optimized(token):
                split_tokens.append(sub_token)

        return split_tokens

    def _word_to_wordpiece_optimized(self, text):  # pylint: disable=method-hidden
        return self.wordpiece_tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        return self.vocab.to_indices(tokens)

    @staticmethod
    def is_first_subword(token):
        """Check if a token is the beginning of subwords.

        Parameters
        ----------
        token : str
            The input token.

        Returns
        -------
        ret : True if the token is the beginning of a serious of wordpieces.

        Examples
        --------
        >>> _, vocab = gluonnlp.model.bert_12_768_12(dataset_name='wiki_multilingual_uncased',
        ...                                          pretrained=False, root='./bert_tokenizer')
        -etc-
        >>> tokenizer = gluonnlp.data.BERTTokenizer(vocab=vocab)
        >>> tokenizer('gluonnlp: 使NLP变得简单。')
        ['gl', '##uo', '##nn', '##lp', ':', '使', 'nl', '##p', '变', '得', '简', '单', '。']
        >>> tokenizer.is_first_subword('gl')
        True
        >>> tokenizer.is_first_subword('##uo')
        False
        """
        return not token.startswith(BERTTokenizer._special_prefix)


class BERTSPTokenizer:
    r"""End-to-end SentencePiece tokenization for BERT models.

    It works best with BERTSentenceTransform().

    .. note::

        BERTSPTokenizer depends on the sentencepiece library. For multi-processing
        with BERTSPTokenizer, making an extra copy of the BERTSPTokenizer instance
        is recommended before using it.

    Parameters
    ----------
    path : str
        Path to the pre-trained subword tokenization model.
    vocab : gluonnlp.Vocab
        Vocabulary for the corpus.
    num_best : int, default 0
        A scalar for sampling subwords. If num_best = {0,1}, no sampling is performed.
        If num_best > 1, then samples from the num_best results.
        If num_best < 0, then assume that num_best is infinite and
        samples from the all hypothesis (lattice) using forward-filtering-and-backward-sampling
        algorithm.
    alpha : float
        A scalar for a smoothing parameter. Inverse temperature for probability rescaling.
    lower : bool, default True
        Whether the text strips accents and convert to lower case.
        If you use the BERT pre-training model,
        lower is set to False when using the cased model,
        otherwise it is set to True.
    max_input_chars_per_word : int, default 200

    Examples
    --------
    >>> url = 'http://repo.mxnet.io/gluon/dataset/vocab/test-682b5d15.bpe'
    >>> f = gluon.utils.download(url)
    -etc-
    >>> bert_vocab = gluonnlp.vocab.BERTVocab.from_sentencepiece(f)
    >>> sp_tokenizer = BERTSPTokenizer(f, bert_vocab, lower=True)
    >>> sentence = 'Better is to bow than break.'
    >>> sp_tokenizer(sentence)
    ['▁better', '▁is', '▁to', '▁b', 'ow', '▁than', '▁brea', 'k', '▁', '.']
    >>> os.remove('test-682b5d15.bpe')
    """

    _special_prefix = '▁'

    def __init__(self,
                 path,
                 vocab,
                 num_best=0,
                 alpha=1.0,
                 lower=True,
                 max_input_chars_per_word=200):
        self._path = path
        self._num_best = num_best
        self._alpha = alpha
        self.sentencepiece = None
        self.basic_tokenizer = BERTBasicTokenizer(lower=lower)
        self.vocab = vocab
        self.max_input_chars_per_word = max_input_chars_per_word


    def _activate_sp(self):
        self.sentencepiece = SentencepieceTokenizer(self._path, self._num_best,
                                                    self._alpha)

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

        return self._tokenizer(sample)

    def _tokenizer(self, text):
        split_tokens = []
        for token in self.basic_tokenizer(text):
            for sub_token in self._tokenize_wordpiece(token):
                split_tokens.append(sub_token)

        return split_tokens

    def _tokenize_wordpiece(self, text):
        """Tokenizes a piece of text into its word pieces.

        This use Google's SentencePiece tokenizer model file

        For example:
          input = "unaffable"
          output = ["▁un", "aff", "able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BERTBasicTokenizer.

        Returns:
          A list of sentencepieced tokens.
        """
        # Swig object can not be pickled when multiprocessing.
        if self.sentencepiece is None:
            self._activate_sp()
        output_tokens = self.sentencepiece(text)
        return output_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        return self.vocab.to_indices(tokens)

    @staticmethod
    def is_first_subword(token):
        """Check if a string token is a subword following a previous subword,
        instead of the beginning of a word.

        Parameters
        ----------
        token : str
            The input token.

        Returns
        -------
        ret : True if the token is the beginning of a series of subwords,

        Examples
        --------
        >>> url = 'http://repo.mxnet.io/gluon/dataset/vocab/test-682b5d15.bpe'
        >>> f = gluon.utils.download(url)
        -etc-
        >>> bert_vocab = gluonnlp.vocab.BERTVocab.from_sentencepiece(f)
        >>> sp_tokenizer = BERTSPTokenizer(f, bert_vocab, lower=True)
        >>> sp_tokenizer('Better is to bow than break.')
        ['▁better', '▁is', '▁to', '▁b', 'ow', '▁than', '▁brea', 'k', '▁', '.']
        >>> sp_tokenizer.is_first_subword('▁better')
        True
        >>> sp_tokenizer.is_first_subword('ow')
        False
        >>> os.remove('test-682b5d15.bpe')
        """
        return token.startswith(BERTSPTokenizer._special_prefix)


class BERTSentenceTransform:
    r"""BERT style data transformation.

    Parameters
    ----------
    tokenizer : BERTTokenizer.
        Tokenizer for the sentences.
    max_seq_length : int.
        Maximum sequence length of the sentences.
    vocab : Vocab
        The vocabulary which has cls_token and sep_token registered.
        If vocab.cls_token is not present, vocab.bos_token is used instead.
        If vocab.sep_token is not present, vocab.eos_token is used instead.
    pad : bool, default True
        Whether to pad the sentences to maximum length.
    pair : bool, default True
        Whether to transform sentences or sentence pairs.
    """

    def __init__(self, tokenizer, max_seq_length, vocab=None, pad=True, pair=True):
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._pad = pad
        self._pair = pair
        self._vocab = self._tokenizer.vocab if vocab is None else vocab
        # RoBERTa does not register CLS token and SEP token
        if hasattr(self._vocab, 'cls_token'):
            self._cls_token = self._vocab.cls_token
        else:
            self._cls_token = self._vocab.bos_token
        if hasattr(self._vocab, 'sep_token'):
            self._sep_token = self._vocab.sep_token
        else:
            self._sep_token = self._vocab.eos_token
        self._padding_token = self._vocab.padding_token

    def __call__(self, line):
        """Perform transformation for sequence pairs or single sequences.

        The transformation is processed in the following steps:
        - tokenize the input sequences
        - insert [CLS], [SEP] as necessary
        - generate type ids to indicate whether a token belongs to the first
        sequence or the second sequence.
        - generate valid length

        For sequence pairs, the input is a tuple of 2 strings:
        text_a, text_b.

        Inputs:
            text_a: 'is this jacksonville ?'
            text_b: 'no it is not'
        Tokenization:
            text_a: 'is this jack ##son ##ville ?'
            text_b: 'no it is not .'
        Processed:
            tokens: '[CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]'
            type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            valid_length: 14

        For single sequences, the input is a tuple of single string:
        text_a.

        Inputs:
            text_a: 'the dog is hairy .'
        Tokenization:
            text_a: 'the dog is hairy .'
        Processed:
            text_a: '[CLS] the dog is hairy . [SEP]'
            type_ids: 0     0   0   0  0     0 0
            valid_length: 7

        If vocab.cls_token and vocab.sep_token are not present,
        vocab.bos_token and vocab.eos_token are used instead.

        Parameters
        ----------
        line: tuple of str
            Input strings. For sequence pairs, the input is a tuple of 2 strings:
            (text_a, text_b). For single sequences, the input is a tuple of single
            string: (text_a,).

        Returns
        -------
        np.array: input token ids in 'int32', shape (batch_size, seq_length)
        np.array: valid length in 'int32', shape (batch_size,)
        np.array: input token type ids in 'int32', shape (batch_size, seq_length)

        """

        # convert to unicode
        text_a = line[0]
        if self._pair:
            assert len(line) == 2
            text_b = line[1]

        tokens_a = self._tokenizer(text_a)
        tokens_b = None

        if self._pair:
            tokens_b = self._tokenizer(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b,
                                    self._max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self._max_seq_length - 2:
                tokens_a = tokens_a[0:(self._max_seq_length - 2)]

        # The embedding vectors for `type=0` and `type=1` were learned during
        # pre-training and are added to the wordpiece embedding vector
        # (and position vector). This is not *strictly* necessary since
        # the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.

        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        tokens.append(self._cls_token)
        tokens.extend(tokens_a)
        tokens.append(self._sep_token)
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens.extend(tokens_b)
            tokens.append(self._sep_token)
            segment_ids.extend([1] * (len(tokens) - len(segment_ids)))

        input_ids = self._vocab[tokens]

        # The valid length of sentences. Only real  tokens are attended to.
        valid_length = len(input_ids)

        if self._pad:
            # Zero-pad up to the sequence length.
            padding_length = self._max_seq_length - valid_length
            # use padding tokens for the rest
            input_ids.extend([self._vocab[self._padding_token]] * padding_length)
            segment_ids.extend([0] * padding_length)

        return np.array(input_ids, dtype='int32'), np.array(valid_length, dtype='int32'),\
            np.array(segment_ids, dtype='int32')

    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

class _GPT2BPE:
    """Base class for GPT-2 BPE tokenizer and detokenizer."""
    def __init__(self):
        codes = list(range(ord('!'), ord('~') + 1)) +\
                list(range(ord('¡'), ord('¬') + 1)) +\
                list(range(ord('®'), ord('ÿ') + 1))
        chr_fn = chr
        try:
            chr_fn(256)
        except ValueError:
            chr_fn = unichr # noqa: F821
        byte_encoder = {code: chr_fn(code) for code in codes}
        shift = 0
        for code in range(2 ** 8):
            if code not in byte_encoder:
                byte_encoder[code] = chr_fn(2 ** 8 + shift)
                shift += 1
        self._byte_encoder = byte_encoder


class GPT2BPETokenizer(_GPT2BPE):
    """BPE tokenizer used in OpenAI GPT-2 model.

    Parameters
    ----------
    root : str, default '$MXNET_HOME/models'
        Location for keeping the BPE rank file.
        MXNET_HOME defaults to '~/.mxnet'.
    """
    bpe_ranks_file_hash = ('openai_webtext_bpe_ranks-396d4d8e.json',
                           '396d4d8ec90cb02f4d56e049e0e4add868bcd943')
    bpe_ranks_archive_hash = ('openai_webtext_bpe_ranks-396d4d8e.zip',
                              '1a770728fd102bc9dc332f322e6bfb294767a685')
    def __init__(self, root=os.path.join(get_home_dir(), 'models')):
        try:
            import regex  # pylint: disable=import-outside-toplevel
            self._regex = regex
        except ImportError:
            raise ImportError(
                'GPT2BPETokenizer requires regex. '
                'To install regex, use pip install -U regex')
        super(GPT2BPETokenizer, self).__init__()
        root = os.path.expanduser(root)
        file_name, sha1_hash = self.bpe_ranks_file_hash
        file_path = os.path.join(root, file_name)
        if not os.path.exists(file_path) or not check_sha1(file_path, sha1_hash):
            if os.path.exists(file_path):
                print('Detected mismatch in the content of BPE rank file. Downloading again.')
            else:
                print('BPE rank file is not found. Downloading.')
            os.makedirs(root, exist_ok=True)

            prefix = str(time.time())
            zip_file_path = os.path.join(root, prefix + file_name)
            repo_url = _get_repo_url()
            if repo_url[-1] != '/':
                repo_url = repo_url + '/'
            archive_name, archive_hash = self.bpe_ranks_archive_hash
            _url_format = '{repo_url}gluon/dataset/vocab/{file_name}'
            download(_url_format.format(repo_url=repo_url, file_name=archive_name),
                     path=zip_file_path,
                     sha1_hash=archive_hash,
                     overwrite=True)
            with zipfile.ZipFile(zip_file_path) as zf:
                if not os.path.exists(file_path):
                    zf.extractall(root)
            try:
                os.remove(zip_file_path)
            except OSError as e:
                # file has already been removed.
                if e.errno == 2:
                    pass
                else:
                    raise e

            if not check_sha1(file_path, sha1_hash):
                raise ValueError('Downloaded file has different hash. Please try again.')
        self._read_bpe_ranks(file_path)
        self._cache = {}
        self._token_pattern = self._regex.compile(
            r'\'s|\'t|\'re|\'ve|\'m|\'ll|\'d| ?\p{L}+'
            r'| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+')

    def _read_bpe_ranks(self, file_path):
        with io.open(file_path, 'r', encoding='utf-8') as f:
            bpe_data = f.read()
            self._bpe_ranks = {
                tuple(merge_str.split()): i for i, merge_str
                in enumerate(bpe_data.split('\n')[1:-1])}

    def get_bpe_subword(self, token):
        """ Encode the word token into BPE subwords

        Parameters
        ----------
        token : str

        Returns
        -------
        chars : list(str)
        """
        if token in self._cache:
            return self._cache[token]
        chars = list(token)
        while len(chars) > 0:
            min_pair, min_rank = None, float('inf')
            # Find the pair with the minimum rank
            for i in range(1, len(chars)):
                pair = (chars[i - 1], chars[i])
                rank = self._bpe_ranks.get(pair, float('inf'))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair
            if min_pair is None or min_pair not in self._bpe_ranks:
                break
            # Merge the pair
            last, tail = chars[0], 1
            for index in range(1, len(chars)):
                if (last, chars[index]) == min_pair:
                    chars[tail - 1] = last + chars[index]
                    last = last + chars[index]
                else:
                    chars[tail - 1] = last
                    tail += 1
                    last = chars[index]
            chars[tail - 1] = last
            chars = chars[:tail]
        self._cache[token] = chars
        return chars

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample : str

        Returns
        -------
        ret : list(str)
        """
        ret = []
        for word_token in self._regex.findall(self._token_pattern, sample):
            word_token = bytearray(word_token.encode('utf-8'))
            word_token = ''.join(self._byte_encoder[code] for code in word_token)
            ret.extend(self.get_bpe_subword(word_token))
        return ret


class GPT2BPEDetokenizer(_GPT2BPE):
    """BPE detokenizer used in OpenAI GPT-2 model."""
    def __init__(self):
        super(GPT2BPEDetokenizer, self).__init__()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample : list(str)

        Returns
        -------
        ret : str
        """
        text = ''.join(sample)
        ret = bytearray(
            [self._byte_decoder[byte] for byte in text]).decode('utf-8', errors='replace')
        return ret
