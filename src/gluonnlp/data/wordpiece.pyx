from typing import Dict, List, Tuple

import cython

__all__ = ['tokenize']


def tokenize(text: str, vocab: Dict[str, int], unknown_token: str, max_input_chars_per_word: cython.int = 200):
    """
    Cython implementation of single token tokenization. Average latency
    decreases to 95ms (from 144ms using original Python code).
    """
    output_tokens: List[str] = []
    token_size: cython.int = len(text)
    if token_size > max_input_chars_per_word:
        output_tokens.append(unknown_token)
        return output_tokens
    is_bad: cython.int = 0
    start: cython.int = 0
    sub_tokens: List[str] = []
    while start < token_size:
        end: cython.int = token_size
        cur_substr: str = None
        while start < end:
            substr = text[start:end]
            if start > 0:
                substr = '##' + substr
            if substr in vocab:
                cur_substr = substr
                break
            end -= 1
        if cur_substr is None:
            is_bad = 1
            break
        sub_tokens.append(cur_substr)
        start = end
    if is_bad == 1:
        output_tokens.append(unknown_token)
    else:
        output_tokens.extend(sub_tokens)

    return output_tokens
