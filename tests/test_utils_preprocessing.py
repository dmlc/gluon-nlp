import numpy as np
from numpy.testing import assert_allclose
from gluonnlp.utils.preprocessing import get_trimmed_lengths


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
