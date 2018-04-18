import math
from collections import Counter


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


def compute_bleu(reference_corpus, translation_corpus, max_n=4, smooth=False, lower_case=False):
    """Compute bleu score of translation against references.

    Parameters
    ----------
    reference_corpus: list(list(list(str)))
        List of lists of references for each translation.
    translation_corpus: list(list(str))
        List of translations to score.
    max_n: int, default 4
        Maximum n-gram order to use when computing BLEU score.
    smooth: bool, default False
        Whether or not to compute smoothed bleu score.
    lower_case: bool, default False
        Whether or not to use lower case of tokens

    Returns
    -------
    5-Tuple with the BLEU score, n-gram precisions, brevity penalty,
        reference length, and translation length
    """
    precision_numerators = Counter()
    precision_denominators = Counter()
    ref_length, trans_length = 0, 0
    assert len(reference_corpus) == len(translation_corpus), "The number of translations and their references do not match"

    for references, translation in zip(reference_corpus, translation_corpus):
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
    closest_ref_len = min(ref_lengths, key=lambda ref_length: (abs(ref_length - trans_length), ref_length))

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
