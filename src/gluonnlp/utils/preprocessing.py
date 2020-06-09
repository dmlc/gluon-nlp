import numpy as np
from typing import List


def get_trimmed_lengths(lengths: List[int],
                        max_length: int,
                        do_merge: bool = False) -> np.ndarray:
    """Get the trimmed lengths of multiple text data. It will make sure that
    the trimmed length is smaller than or equal to the max_length

    - do_merge is True
        Make sure that sum(trimmed_lengths) <= max_length.
        The strategy is to always try to trim the longer lengths.
    - do_merge is False
        Make sure that all(trimmed_lengths <= max_length)

    Parameters
    ----------
    lengths
        The original lengths of each sample
    max_length
        When do_merge is True,
            We set the max_length constraint on the total length.
        When do_merge is False,
            We set the max_length constraint on individual sentences.
    do_merge
        Whether these sentences will be merged

    Returns
    -------
    trimmed_lengths
        The trimmed lengths of the
    """
    lengths = np.array(lengths)
    if do_merge:
        total_length = sum(lengths)
        if total_length <= max_length:
            return lengths
        trimmed_lengths = np.zeros_like(lengths)
        while sum(trimmed_lengths) != max_length:
            remainder = max_length - sum(trimmed_lengths)
            budgets = lengths - trimmed_lengths
            nonzero_idx = (budgets > 0).nonzero()[0]
            nonzero_budgets = budgets[nonzero_idx]
            if remainder // len(nonzero_idx) == 0:
                for i in range(remainder):
                    trimmed_lengths[nonzero_idx[i]] += 1
            else:
                increment = min(min(nonzero_budgets), remainder // len(nonzero_idx))
                trimmed_lengths[nonzero_idx] += increment
        return trimmed_lengths
    else:
        return np.minimum(lengths, max_length)
