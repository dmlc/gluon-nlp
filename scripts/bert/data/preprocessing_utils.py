"""Utility classes and functions for data processing"""

__all__ = [
    'truncate_seqs_equal', 'concat_sequences', 'tokenize_and_align_positions',
    'get_doc_spans', 'align_position2doc_spans', 'improve_answer_span',
    'check_is_max_context', 'convert_squad_examples'
]

import collections
import itertools
import unicodedata
import numpy as np
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
                quotient + 1 if i < remainder else quotient
                for i in range(lens.count())
            ]
            break
    seqs = [seq[:length] for (seq, length) in zip(seqs, lens.data.tolist())]
    return seqs


def concat_sequences(seqs, separators, separator_mask=[]):
    """
    Insert special tokens for sequence list or a single sequence.
    For sequence pairs, the input is a list of 2 strings:
    text_a, text_b.
    Inputs:
       text_a: 'is this jacksonville ?'
       text_b: 'no it is not'
       separator: [[SEP], [SEP]]

    Processed:
       tokens:     'is this jacksonville ? [SEP] no it is not . [SEP]'
       segment_ids: 0  0    0            0  0    1  1  1  1   1 1
       p_mask:      0  0    0            0  1    0  0  0  0   0 1
       valid_length: 11

    Parameters
    ----------
    separator : list
        The special tokens to be appended to each sequence. For example:
        Given:
            seqs: [[1, 2], [3, 4], [5, 6]]
            separator: [[], 7]
        it will be:
            [1, 2, 3, 4, 7, 5, 6]

    seqs : list of sequences or a single sequence

    Returns
    -------
    np.array: input token ids in 'int32', shape (batch_size, seq_length)
    np.array: segment ids in 'int32', shape (batch_size, seq_length)
    np.array: mask for special tokens
    """
    assert isinstance(seqs, collections.abc.Iterable) and len(seqs) > 0
    concat = sum((
        seq + sep
        for sep, seq in itertools.zip_longest(separators, seqs, fillvalue=[])),
                 [])
    segment_ids = sum(
        ([i] * (len(seq) + len(sep)) for i, (sep, seq) in enumerate(
            itertools.zip_longest(separators, seqs, fillvalue=[]))), [])
    p_mask = sum((
        [0] * len(seq) + [separator_mask] * len(sep)
        for sep, seq in itertools.zip_longest(separators, seqs, fillvalue=[])),
                 [])
    return concat, segment_ids, p_mask

def concat_sequences_2(seqs, separators, separator_mask=[]):
    """
    Insert special tokens for sequence list or a single sequence.
    For sequence pairs, the input is a list of 2 strings:
    text_a, text_b.
    Inputs:
       text_a: 'is this jacksonville ?'
       text_b: 'no it is not'
       separator: [[SEP], [SEP]]

    Processed:
       tokens:     'is this jacksonville ? [SEP] no it is not . [SEP]'
       segment_ids: 0  0    0            0  0    1  1  1  1   1 1
       p_mask:      0  0    0            0  1    0  0  0  0   0 1
       valid_length: 11

    Parameters
    ----------
    separator : list
        The special tokens to be appended to each sequence. For example:
        Given:
            seqs: [[1, 2], [3, 4], [5, 6]]
            separator: [[], 7]
        it will be:
            [1, 2, 3, 4, 7, 5, 6]

    seqs : list of sequences or a single sequence

    Returns
    -------
    np.array: input token ids in 'int32', shape (batch_size, seq_length)
    np.array: segment ids in 'int32', shape (batch_size, seq_length)
    np.array: mask for special tokens
    """
    assert isinstance(seqs, collections.abc.Iterable) and len(seqs) > 0
    concat = sum((
        seq + sep
        for sep, seq in itertools.zip_longest(separators, seqs, fillvalue=[])),
                 [])
    segment_ids = sum(
        ([i] * (len(seq) + len(sep)) for i, (sep, seq) in enumerate(
            itertools.zip_longest(separators, seqs, fillvalue=[]))), [])
    p_mask = sum((
        [0] * len(seq) + mask
        for sep, seq, mask in itertools.zip_longest(separators, seqs, separator_mask, fillvalue=[])),
                 [])
    return concat, segment_ids, p_mask

def tokenize_and_align_positions(origin_text, start_position, end_position,
                                 tokenizer):
    """Tokenize the text and align the origin positions to the corresponding position"""
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
    """A simple function that applying a sliding window on the doc and get doc spans

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
        doc_spans.append(
            (full_doc[start_offset:end_offset], (start_offset, end_offset)))
        if start_offset + length == len(full_doc):
            break
        start_offset += min(length, doc_stride)
    return list(zip(*doc_spans))


def align_position2doc_spans(positions,
                             doc_spans_indices,
                             offset=0,
                             default_value=-1,
                             all_in_span=True):
    """Align the origin positions to the corresponding position in doc spans"""
    if not isinstance(positions, list):
        positions = [positions]
    doc_start, doc_end = doc_spans_indices
    if all_in_span and not all([p in range(doc_start, doc_end) for p in positions]):
        return [default_value] * len(positions)
    new_positions = [
        p - doc_start +
        offset if p in range(doc_start, doc_end) else default_value
        for p in positions
    ]
    return new_positions


def improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                        orig_answer_text):
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


SquadExample = collections.namedtuple('SquadExample', [
    'qas_id', 'question_text', 'paragraph_text', 'doc_tokens', 'example_id', 'orig_answer_text',
    'start_position', 'end_position', 'is_impossible'
])


def convert_squad_examples(record, is_training):
    """read a single entry of gluonnlp.data.SQuAD and convert it to an example"""
    example_id = record[0]
    qas_id = record[1]
    question_text = record[2]
    paragraph_text = record[3]
    orig_answer_text = record[4][0] if record[4] else ''
    answer_offset = record[5][0] if record[5] else ''
    is_impossible = record[6] if len(record) == 7 else False

    answer_length = len(orig_answer_text)
    doc_tokens = []

    char_to_word_offset = []
    prev_is_whitespace = True

    for c in paragraph_text:
        if str.isspace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    if not is_training:
        start_position = -1
        end_position = -1
    else:
        start_position = char_to_word_offset[
            answer_offset] if not is_impossible else -1
        end_position = char_to_word_offset[answer_offset + answer_length -
                                           1] if not is_impossible else -1

    example = SquadExample(qas_id=qas_id,
                           question_text=question_text,
                           paragraph_text=paragraph_text,
                           doc_tokens=doc_tokens,
                           example_id=example_id,
                           orig_answer_text=orig_answer_text,
                           start_position=start_position,
                           end_position=end_position,
                           is_impossible=is_impossible)
    return example


def preprocess_text(inputs, lower=False, remove_space=True, keep_accents=False):
    if remove_space:
        outputs = ' '.join(inputs.strip().split())
    else:
        outputs = inputs
    outputs = outputs.replace("``", '"').replace("''", '"')
    if not keep_accents:
        outputs = unicodedata.normalize('NFKD', outputs)
        outputs = ''.join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()

    return outputs


def _convert_index(index, pos, M=None, is_start=True):
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


def _lcs_match(max_dist, seq1, seq2, max_first_seq_len, max_second_seq_len, lower=False):
    f = np.zeros((max(len(seq1), 1024), max(len(seq2), 1024)), dtype=np.float32)
    g = {}
    for i in range(max_first_seq_len):
        for j in range(i - max_dist, i + max_dist):
            if j >= max_second_seq_len or j < 0: continue

            if i > 0:
                g[(i, j)] = 0
                f[i, j] = f[i - 1, j]

            if j > 0 and f[i, j - 1] > f[i, j]:
                g[(i, j)] = 1
                f[i, j] = f[i, j - 1]

            f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0
            if (preprocess_text(seq1[i], lower=lower,
                                remove_space=False) == seq2[j]
                    and f_prev + 1 > f[i, j]):
                g[(i, j)] = 2
                f[i, j] = f_prev + 1
    return f, g
