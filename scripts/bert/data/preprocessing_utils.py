"""Utility classes and functions for data processing"""

__all__ = [
    'truncate_seqs_equal', 'concat_sequences']

import collections
import itertools
import numpy.ma as ma


def truncate_seqs_equal(seqs, max_len):
    """truncate a list of seqs so that the total length equals max length.

    Trying to truncate the seqs to equal length.

    Returns
    -------
    list : list of truncated sequence keeping the origin order
    """
    assert isinstance(seqs, list)
    lens = list(map(len, seqs))
    if sum(lens) <= max_len:
        return seqs

    lens = ma.masked_array(lens, mask=[0] * len(lens))
    while True:
        argmin = lens.argmin()
        minval = lens[argmin]
        quotient, remainder = divmod(max_len, len(lens) - sum(lens.mask))
        if minval <= quotient:  # Ignore values that don't need truncation
            lens.mask[argmin] = 1
            max_len -= minval
        else:  # Truncate all
            lens.data[~lens.mask] = [
                quotient + 1 if i < remainder else quotient for i in range(lens.count())
            ]
            break
    seqs = [seq[:length] for (seq, length) in zip(seqs, lens.data.tolist())]
    return seqs


def concat_sequences(seqs, separators, seq_mask=0, separator_mask=1):
    """Concatenate sequences in a list into a single sequence, using specified separators.

    Example 1:
    seqs: [['is', 'this' ,'jacksonville', '?'], ['no' ,'it' ,'is' ,'not', '.']]
    separator: [[SEP], [SEP], [CLS]]
    seq_mask: 0
    separator_mask: 1
    Returns:
       tokens:      is this jacksonville ? [SEP] no it is not . [SEP] [CLS]
       segment_ids: 0  0    0            0  0    1  1  1  1   1 1     2
       p_mask:      0  0    0            0  1    0  0  0  0   0 1     1

    Example 2:
    separator_mask can also be a list.
    seqs: [['is', 'this' ,'jacksonville', '?'], ['no' ,'it' ,'is' ,'not', '.']]
    separator: [[SEP], [SEP], [CLS]]
    seq_mask: 0
    separator_mask: [[1], [1], [0]]

    Returns:
       tokens:     'is this jacksonville ? [SEP] no it is not . [SEP] [CLS]'
       segment_ids: 0  0    0            0  0    1  1  1  1   1 1     2
       p_mask:      1  1    1            1  1    0  0  0  0   0 1     0

    Example 3:
    seq_mask can also be a list.
    seqs: [['is', 'this' ,'jacksonville', '?'], ['no' ,'it' ,'is' ,'not', '.']]
    separator: [[SEP], [SEP], [CLS]]
    seq_mask: [[1, 1, 1, 1], [0, 0, 0, 0, 0]]
    separator_mask: [[1], [1], [0]]

    Returns:
       tokens:     'is this jacksonville ? [SEP] no it is not . [SEP] [CLS]'
       segment_ids: 0  0    0            0  0    1  1  1  1   1 1     2
       p_mask:      1  1    1            1  1    0  0  0  0   0 1     0

    Parameters
    ----------
    seqs : list
        sequences to be concatenated
    separator : list
        The special tokens to separate sequences.
    seq_mask : int or list
        A single mask value for all sequence items or a list of values for each item in sequences
    separator_mask : int or list
        A single mask value for all separators or a list of values for each separator

    Returns
    -------
    np.array: input token ids in 'int32', shape (batch_size, seq_length)
    np.array: segment ids in 'int32', shape (batch_size, seq_length)
    np.array: mask for special tokens
    """
    assert isinstance(seqs, collections.abc.Iterable) and len(seqs) > 0
    assert isinstance(seq_mask, (list, int))
    assert isinstance(separator_mask, (list, int))
    concat = sum((seq + sep for sep, seq in itertools.zip_longest(separators, seqs, fillvalue=[])),
                 [])
    segment_ids = sum(
        ([i] * (len(seq) + len(sep))
         for i, (sep, seq) in enumerate(itertools.zip_longest(separators, seqs, fillvalue=[]))),
        [])
    if isinstance(seq_mask, int):
        seq_mask = [[seq_mask] * len(seq) for seq in seqs]
    if isinstance(separator_mask, int):
        separator_mask = [[separator_mask] * len(sep) for sep in separators]

    p_mask = sum((s_mask + mask for sep, seq, s_mask, mask in itertools.zip_longest(
        separators, seqs, seq_mask, separator_mask, fillvalue=[])), [])
    return concat, segment_ids, p_mask
