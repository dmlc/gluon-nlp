"""test data preprocessing utils"""

import numpy as np
from ..bert.data.preprocessing_utils import truncate_seqs_equal, concat_sequences


def test_truncate():
    seqs = [[j*i for j in range(i)] for i in range(1,10)]
    res1 = [[0], [0, 2], [0, 3, 6], [0, 4, 8], [0, 5, 10], [0, 6], [0, 7], [0, 8], [0, 9]]
    seq = [[i for i in range(20)]]

    truncated = truncate_seqs_equal(seqs, 20)
    truncated2 = truncate_seqs_equal(seq, 20)

    assert all(truncated == np.array(res1))
    assert all(truncated2[0] == np.array(seq)[0])

def test_concat_sequence():
    seqs = [[3 * i + j for j in range(3)] for i in range(3)]
    seperators = [['a'], ['b'], ['c']]
    res = concat_sequences(seqs, seperators)
    assert res[0] == [0, 1, 2, 'a', 3, 4, 5, 'b', 6, 7, 8, 'c']
    assert res[1] == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    assert res[2] == [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]

    seperators = [['a'], [], ['b']]
    res = concat_sequences(seqs, seperators)
    assert res[0] == [0, 1, 2, 'a', 3, 4, 5, 6, 7, 8, 'b']
    assert res[1] == [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    assert res[2] == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]