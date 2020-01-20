"""Utility classes and functions for data processing"""

__all__ = [
    'truncate_seqs_equal', 'concat_sequences', 'tokenize_and_align_positions', 'get_doc_spans',
    'align_position2doc_spans', 'improve_answer_span', 'check_is_max_context']

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


def tokenize_and_align_positions(origin_text, start_position, end_position, tokenizer):
    """Tokenize the text and align the origin positions to the corresponding position.

    Parameters
    ----------
    origin_text : list
        list of tokens to be tokenized.
    start_position : int
        Start position in the origin_text
    end_position : int
        End position in the origin_text
    tokenizer : callable function, e.g., BERTTokenizer.

    Returns
    -------
    int: Aligned start position
    int: Aligned end position
    list: tokenized text
    list: map from the origin index to the tokenized sequence index
    list: map from tokenized sequence index to the origin index
    """
    orig_to_tok_index = []
    tok_to_orig_index = []
    tokenized_text = []
    for (i, token) in enumerate(origin_text):
        orig_to_tok_index.append(len(tokenized_text))
        sub_tokens = tokenizer(token)
        tokenized_text += sub_tokens
        tok_to_orig_index += [i] * len(sub_tokens)

    start_position = orig_to_tok_index[start_position]
    end_position = orig_to_tok_index[end_position + 1] - 1 if end_position < len(origin_text) - 1  \
        else len(tokenized_text) - 1
    return start_position, end_position, tokenized_text, orig_to_tok_index, tok_to_orig_index


def get_doc_spans(full_doc, max_length, doc_stride):
    """Obtain document spans by sliding a window across the document

    Parameters
    ----------
    full_doc: list
        The origin doc text
    max_length: max_length
        Maximum size of a doc span
    doc_stride: int
        Step of sliding window

    Returns
    -------
    list: a list of processed doc spans
    list: a list of start/end index of each doc span
    """
    doc_spans = []
    start_offset = 0
    while start_offset < len(full_doc):
        length = min(max_length, len(full_doc) - start_offset)
        end_offset = start_offset + length
        doc_spans.append((full_doc[start_offset:end_offset], (start_offset, end_offset)))
        if start_offset + length == len(full_doc):
            break
        start_offset += min(length, doc_stride)
    return list(zip(*doc_spans))


def align_position2doc_spans(positions, doc_spans_indices, offset=0, default_value=-1,
                             all_in_span=True):
    """Align original positions to the corresponding document span positions

    Parameters
    ----------
    positions: list or int
        A single or a list of positions to be aligned
    dic_spans_indices: list or tuple
        (start_position, end_position)
    offset: int
        Offset of aligned positions. Sometimes the doc spans would be added
        after a question text, in this case, the new position should add
        len(question_text)
    default_value: int
        The default value to return if the positions are not in the doc span.
    all_in_span: bool
        If set to True, then as long as one position is out of span, all positions
        would be set to default_value.
    Returns
    -------
    list: a list of aligned positions
    """
    if not isinstance(positions, list):
        positions = [positions]
    doc_start, doc_end = doc_spans_indices
    if all_in_span and not all([p in range(doc_start, doc_end) for p in positions]):
        return [default_value] * len(positions)
    new_positions = [
        p - doc_start + offset if p in range(doc_start, doc_end) else default_value
        for p in positions
    ]
    return new_positions


def improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer.

    The SQuAD annotations are character based. We first project them to
    whitespace-tokenized words. But then after WordPiece tokenization, we can
    often find a "better match". For example:

      Question: What year was John Smith born?
      Context: The leader was John Smith (1895-1943).
      Answer: 1895

    The original whitespace-tokenized answer will be "(1895-1943).". However
    after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    the exact answer, 1895.

    However, this is not always possible. Consider the following:

      Question: What country is the top exporter of electornics?
      Context: The Japanese electronics industry is the lagest in the world.
      Answer: Japan

    In this case, the annotator chose "Japan" as a character sub-span of
    the word "Japanese". Since our WordPiece tokenizer does not split
    "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    in SQuAD, but does happen.

    Parameters
    ----------
    doc_tokens: list
        A list of doc tokens
    input_start: int
        start position of the answer
    input_end: int
        end position of the answer
    tokenizer: callable function
    orig_answer_text: str
        origin answer text.
    Returns
    -------
    tuple: a tuple of improved start position and end position
    """
    tok_answer_text = ' '.join(tokenizer(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = ' '.join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token.

    Because of the sliding window approach taken to scoring documents, a single
    token can appear in multiple documents. E.g.
     Doc: the man went to the store and bought a gallon of milk
     Span A: the man went to the
     Span B: to the store and bought
     Span C: and bought a gallon of
     ...

    Now the word 'bought' will have two scores from spans B and C. We only
    want to consider the score with "maximum context", which we define as
    the *minimum* of its left and right context (the *sum* of left and
    right context will always be the same, of course).

    In the example the maximum context for 'bought' would be span C since
    it has 1 left context and 3 right context, while span B has 4 left context
    and 0 right context.

    Note that position is the absolute position in the origin text.

    Parameters
    ----------
    doc_spans: list
        A list of doc spans
    cur_span_index: int
        The index of doc span to be checked in doc_spans.
    position: int
        Position of the token to be checked.
    Returns
    -------
    bool: True if the token has 'max context'.
    """
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        start, end = doc_span
        end -= 1
        length = end - start + 1
        if position < start:
            continue
        if position > end:
            continue
        num_left_context = position - start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + \
                0.01 * length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index
