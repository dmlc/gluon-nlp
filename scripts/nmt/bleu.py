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

"""BLEU."""
import sys
import re
import math
import unicodedata
from collections import Counter
import six
LIST_TYPES = (list, tuple)


def _ngrams(segment, n):
    """Extracts n-grams from an input segment.

    Parameters
    ----------
    segment: list
        Text segment from which n-grams will be extracted.
    n: int
        Order of n-gram.

    Returns
    -------
    ngram_counts: Counter
        Contain all the nth n-grams in segment with a count of how many times each n-gram occurred.
    """
    ngram_counts = Counter()
    for i in range(0, len(segment) - n + 1):
        ngram = tuple(segment[i:i + n])
        ngram_counts[ngram] += 1
    return ngram_counts


def _split_compound_word(segment):
    """Put compounds in ATAT format.
       rich-text format" --> rich ##AT##-##AT## text format.
    """
    return re.sub(r'(\S)-(\S)', '\\1 ##AT##-##AT## \\2', ' '.join(segment)).split()


def _bpe_to_words(sentence, delimiter='@@'):
    """Convert a sequence of bpe words into sentence."""
    words = []
    word = ''
    delimiter_len = len(delimiter)
    for subwords in sentence:
        if len(subwords) >= delimiter_len and subwords[-delimiter_len:] == delimiter:
            word += subwords[:-delimiter_len]
        else:
            word += subwords
            words.append(word)
            word = ''
    return words


def _tokenize_mteval_13a(segment):
    r"""
    Tokenizes a string following the tokenizer in mteval-v13a.pl.
    See https://github.com/moses-smt/mosesdecoder/"
           "blob/master/scripts/generic/mteval-v14.pl#L917-L942
    Parameters
    ----------
    segment: str
        A string to be tokenzied

    Returns
    -------
    The tokenized string
    """

    norm = segment.rstrip()

    norm = norm.replace('<skipped>', '')
    norm = norm.replace('-\n', '')
    norm = norm.replace('\n', ' ')
    norm = norm.replace('&quot;', '"')
    norm = norm.replace('&amp;', '&')
    norm = norm.replace('&lt;', '<')
    norm = norm.replace('&gt;', '>')

    norm = ' {} '.format(norm)
    norm = re.sub(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', ' \\1 ', norm)
    norm = re.sub(r'([^0-9])([\.,])', '\\1 \\2 ', norm)
    norm = re.sub(r'([\.,])([^0-9])', ' \\1 \\2', norm)
    norm = re.sub(r'([0-9])(-)', '\\1 \\2 ', norm)
    norm = re.sub(r'\s+', ' ', norm)
    norm = re.sub(r'^\s+', '', norm)
    norm = re.sub(r'\s+$', '', norm)

    return norm


class UnicodeRegex(object):
    """Ad-hoc hack to recognize all punctuation and symbols.
    """
    def __init__(self):
        punctuation = self._property_chars('P')
        self.nondigit_punct_re = re.compile(r'([^\d])([' + punctuation + r'])')
        self.punct_nondigit_re = re.compile(r'([' + punctuation + r'])([^\d])')
        self.symbol_re = re.compile('([' + self._property_chars('S') + '])')

    def _property_chars(self, prefix):
        return ''.join(six.unichr(x) for x in range(sys.maxunicode)
                       if unicodedata.category(six.unichr(x)).startswith(prefix))


unicodeRegex = UnicodeRegex()


def _tokenize_mteval_v14_intl(segment):
    r"""Tokenize a string following following the international tokenizer in mteval-v14a.pl.
    See https://github.com/moses-smt/mosesdecoder/"
           "blob/master/scripts/generic/mteval-v14.pl#L954-L983

    Parameters
    ----------
    segment: str
        A string to be tokenized

    Returns
    -------
    The tokenized string
    """
    segment = segment.rstrip()
    segment = unicodeRegex.nondigit_punct_re.sub(r'\1 \2 ', segment)
    segment = unicodeRegex.punct_nondigit_re.sub(r' \1 \2', segment)
    segment = unicodeRegex.symbol_re.sub(r' \1 ', segment)
    return segment.strip()


TOKENIZERS = {
    '13a': _tokenize_mteval_13a,
    'intl': _tokenize_mteval_v14_intl,
    None: lambda x: x,
}


def compute_bleu(reference_corpus_list, translation_corpus, tokenized=True,
                 tokenizer='13a', max_n=4, smooth=False, lower_case=False,
                 bpe=False, split_compound_word=False):
    r"""Compute bleu score of translation against references.

    Parameters
    ----------
    reference_corpus_list: list of list(list(str)) or list of list(str)
        list of list(list(str)): tokenzied references
        list of list(str): plain text
        List of references for each translation.
    translation_corpus: list(list(str)) or list(str)
        list(list(str)): tokenzied translation
        list(str): plain text
        Translations to score.
    tokenized: bool, default True
        Whether the inputs has been tokenized.
    tokenizer: str or None, default '13a'
        '13a': follow the tokenizer in mteval-v13a.pl
        'intl': follow the international tokenzier in mteval-v14.pl
        None: identity mapping on the string.
        This option is ignored if tokenized is True
    max_n: int, default 4
        Maximum n-gram order to use when computing BLEU score.
    smooth: bool, default False
        Whether or not to compute smoothed bleu score.
    lower_case: bool, default False
        Whether or not to use lower case of tokens
    split_compound_word: bool, default False
        Whether or not to split compound words
        "rich-text format" --> rich ##AT##-##AT## text format.
    bpe: bool, default False
        Whether or not the inputs are in BPE format

    Returns
    -------
    5-Tuple with the BLEU score, n-gram precisions, brevity penalty,
        reference length, and translation length
    """
    precision_numerators = [0 for _ in range(max_n)]
    precision_denominators = [0 for _ in range(max_n)]
    ref_length, trans_length = 0, 0
    for references in reference_corpus_list:
        assert len(references) == len(translation_corpus), \
            'The number of translations and their references do not match'
    if tokenized:
        assert isinstance(reference_corpus_list[0][0], LIST_TYPES) and \
               isinstance(translation_corpus[0], LIST_TYPES), \
            'references and translation should have format of list of list(list(str)) ' \
            'and list(list(str)), respectively, when toknized is True.'
    else:
        assert isinstance(reference_corpus_list[0][0], str) and \
               isinstance(translation_corpus[0], str), \
            'references and translation should have format of list(list(str)) ' \
            'and list(str), respectively, when toknized is False.'
    for references, translation in zip(zip(*reference_corpus_list), translation_corpus):
        if not tokenized:
            references = [TOKENIZERS[tokenizer](reference).split() for reference in references]
            translation = TOKENIZERS[tokenizer](translation).split()
        if bpe:
            references = [_bpe_to_words(reference) for reference in references]
            translation = _bpe_to_words(translation)
        if split_compound_word:
            references = [_split_compound_word(reference) for reference in references]
            translation = _split_compound_word(translation)
        if lower_case:
            references = [list(map(str.lower, reference)) for reference in references]
            translation = list(map(str.lower, translation))
        trans_len = len(translation)
        trans_length += trans_len
        ref_length += _closest_ref_length(references, trans_len)
        for n in range(max_n):
            matches, candidates = _compute_precision(references, translation, n + 1)
            precision_numerators[n] += matches
            precision_denominators[n] += candidates

    precision_fractions = [(precision_numerators[n], precision_denominators[n])
                           for n in range(max_n)]
    smooth_const = 0
    if smooth:
        smooth_const = 1
    precisions = _smoothing(precision_fractions, smooth_const)
    if min(precisions) > 0:
        precision_log_average = sum(math.log(p) for p in precisions) / max_n
        precision_exp_log_average = math.exp(precision_log_average)
    else:
        precision_exp_log_average = 0

    bp = _brevity_penalty(ref_length, trans_length)
    bleu = precision_exp_log_average*bp

    return bleu, precisions, bp, ref_length, trans_length


def _compute_precision(references, translation, n):
    """Compute ngram precision.

    Parameters
    ----------
    references: list(list(str))
        A list of references.
    translation: list(str)
        A translation.
    n: int
        Order of n-gram.

    Returns
    -------
    matches: int
        Number of matched nth order n-grams
    candidates
        Number of possible nth order n-grams
    """
    matches = 0
    candidates = 0
    ref_ngram_counts = Counter()

    for reference in references:
        ref_ngram_counts |= _ngrams(reference, n)
    trans_ngram_counts = _ngrams(translation, n)
    overlap_ngram_counts = trans_ngram_counts & ref_ngram_counts
    matches += sum(overlap_ngram_counts.values())
    possible_matches = len(translation) - n + 1
    if possible_matches > 0:
        candidates += possible_matches

    return matches, candidates


def _brevity_penalty(ref_length, trans_length):
    """Calculate brevity penalty.

    Parameters
    ----------
    ref_length: int
        Sum of all closest references'lengths for every translations in a corpus
    trans_length: int
        Sum of all translations's lengths in a corpus.

    Returns
    -------
    bleu's brevity penalty: float
    """
    if trans_length > ref_length:
        return 1
    # If translation is empty, brevity penalty = 0 should result in BLEU = 0.0
    elif trans_length == 0:
        return 0
    else:
        return math.exp(1 - float(ref_length) / trans_length)


def _closest_ref_length(references, trans_length):
    """Find the reference that has the closest length to the translation.

    Parameters
    ----------
    references: list(list(str))
        A list of references.
    trans_length: int
        Length of the translation.

    Returns
    -------
    closest_ref_len: int
        Length of the reference that is closest to the translation.
    """
    ref_lengths = (len(reference) for reference in references)
    closest_ref_len = min(ref_lengths,
                          key=lambda ref_length: (abs(ref_length - trans_length), ref_length))

    return closest_ref_len


def _smoothing(precision_fractions, c=1):
    """Compute the smoothed precision for all the orders.

    Parameters
    ----------
    precision_fractions: list(tuple)
        Contain a list of (precision_numerator, precision_denominator) pairs
    c: int, default 1
        Smoothing constant to use

    Returns
    -------
    ratios: list of floats
        Contain the smoothed precision_fractions.
    """
    ratios = [0] * len(precision_fractions)
    for i, precision_fraction in enumerate(precision_fractions):
        if precision_fraction[1] > 0:
            ratios[i] = float(precision_fraction[0] + c) / (precision_fraction[1] + c)
        else:
            ratios[i] = 0.0

    return ratios
