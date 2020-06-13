import pytest
import numpy as np
from numpy.testing import assert_allclose
from gluonnlp.utils.preprocessing import get_trimmed_lengths, match_tokens_with_char_spans


def test_get_trimmed_lengths():
    for lengths, do_merge, max_length, gt_trimmed_lengths in\
            [([10, 5, 4, 8], False, 6, [6, 5, 4, 6]),
             ([10, 5, 4, 8], True, 6, [2, 2, 1, 1]),
             ([20], False, 30, [20]),
             ([20], True, 30, [20]),
             ([15, 20], False, 30, [15, 20]),
             ([15, 20], True, 30, [15, 15])]:
        trimmed_lengths = get_trimmed_lengths(lengths,
                                              max_length=max_length,
                                              do_merge=do_merge)
        assert_allclose(trimmed_lengths, np.array(gt_trimmed_lengths))


def test_match_tokens_with_char_spans():
    token_offsets = np.array([(0, 1), (1, 2), (3, 4), (5, 6)])
    spans = np.array([(0, 3), (4, 6)])
    out = match_tokens_with_char_spans(token_offsets, spans)
    assert_allclose(out, np.array([[0, 2],
                                   [2, 3]]))

    token_offsets = np.array([(5, 10), (10, 20), (20, 25), (26, 30)])
    spans = np.array([(0, 3), (4, 6), (10, 30),
                      (22, 23), (15, 25),
                      (10, 35), (36, 38)])
    out = match_tokens_with_char_spans(token_offsets, spans)
    assert_allclose(out, np.array([[0, 0],
                                   [0, 0],
                                   [1, 3],
                                   [2, 2],
                                   [1, 2],
                                   [1, 3],
                                   [3, 3]]))
