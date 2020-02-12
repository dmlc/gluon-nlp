"""Utility classes and functions for xlnet squad preprocessing"""

__all__ = ['convert_index', 'lcs_match']

def _preprocess_text(inputs, lower=False, remove_space=True, keep_accents=False):
    """Remove space, convert to lower case, keep accents.

    Parameters
    ----------
    inputs: str
        input string
    lower: bool
        If convert the input string to lower case.
    remove_space: bool
        If remove the spaces in the input string.
    keep_accents: bool
        If keep accents in the input string.

    Returns
    -------
    str: processed input string
    """
    if remove_space:
        outputs = ' '.join(inputs.strip().split())
    else:
        outputs = inputs
    outputs = outputs.replace('``', '"').replace('\'\'', '"')
    if not keep_accents:
        outputs = unicodedata.normalize('NFKD', outputs)
        outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()
    return outputs


def convert_index(index, pos, M=None, is_start=True):
    """Working best with _lcs_match(), convert the token index to origin text index"""
    if index[pos] is not None:
        return index[pos]
    N = len(index)
    rear = pos
    while rear < N - 1 and index[rear] is None:
        rear += 1
    front = pos
    while front > 0 and index[front] is None:
        front -= 1
    assert index[front] is not None or index[rear] is not None
    if index[front] is None:
        if index[rear] >= 1:
            if is_start:
                return 0
            else:
                return index[rear] - 1
        return index[rear]
    if index[rear] is None:
        if M is not None and index[front] < M - 1:
            if is_start:
                return index[front] + 1
            else:
                return M - 1
        return index[front]
    if is_start:
        if index[rear] > index[front] + 1:
            return index[front] + 1
        else:
            return index[rear]
    else:
        if index[rear] > index[front] + 1:
            return index[rear] - 1
        else:
            return index[front]


def lcs_match(max_dist, seq1, seq2, max_seq_length=1024, lower=False):
    """Longest common sequence match.

    unlike standard LCS, this is specifically optimized for the setting
    because the mismatch between sentence pieces and original text will be small

    Parameters
    ----------
    max_dist: int
        The max distance between tokens to be considered.
    seq1: list
        The first sequence to be matched.
    seq2: list
        The second sequence to be matched.
    lower: bool
        If match the lower-cased tokens.
    Returns
    -------
    numpyArray: Token-wise lcs matrix f. Shape of ((max(len(seq1), 1024), max(len(seq2), 1024))
    Map: The dp path in matrix f.
        g[(i ,j)] == 2 if token_i in seq1 matches token_j in seq2.
        g[(i, j)] == 1 if token_i in seq1 matches token_{j-1} in seq2.
        g[(i, j)] == 0 of token_{i-1} in seq1 matches token_j in seq2.
    """
    f = np.zeros((max(len(seq1), max_seq_length), max(len(seq2), max_seq_length)),
                 dtype=np.float32)
    g = {}
    for i, token in enumerate(seq1):
        for j in range(i - max_dist, i + max_dist):
            if j >= len(seq2) or j < 0:
                continue

            if i > 0:
                g[(i, j)] = 0
                f[i, j] = f[i - 1, j]

            if j > 0 and f[i, j - 1] > f[i, j]:
                g[(i, j)] = 1
                f[i, j] = f[i, j - 1]

            f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0
            if (_preprocess_text(token, lower=lower, remove_space=False) == seq2[j]
                    and f_prev + 1 > f[i, j]):
                g[(i, j)] = 2
                f[i, j] = f_prev + 1
    return f, g
