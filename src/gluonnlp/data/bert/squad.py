"""Utility functions for BERT squad data preprocessing"""

__all__ = [
    'tokenize_and_align_positions', 'get_doc_spans',
    'align_position2doc_spans', 'improve_answer_span', 'check_is_max_context',
    'convert_squad_examples'
]

import collections


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

    Examples
    --------
    >>> from gluonnlp.vocab import BERTVocab
    >>> from gluonnlp.data import count_tokens, BERTTokenizer
    >>> origin_text = ['is', 'this', 'jacksonville', '?']
    >>> vocab_tokens = ['is', 'this', 'jack', '##son', '##ville', '?']
    >>> bert_vocab = BERTVocab(count_tokens(vocab_tokens))
    >>> tokenizer = BERTTokenizer(vocab=bert_vocab)
    >>> out = tokenize_and_align_positions(origin_text, 0, 2, tokenizer)
    >>> out[0] # start_position
    0
    >>> out[1] # end_position
    4
    >>> out[2] # tokenized_text
    ['is', 'this', 'jack', '##son', '##ville', '?']
    >>> out[3] # orig_to_tok_index
    [0, 1, 2, 5]
    >>> out[4] # tok_to_orig_index
    [0, 1, 2, 2, 2, 3]
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
    doc_spans_indices: list or tuple
        Contains the start/end position of the doc_spans. Typically, (start_position, end_position)
    offset: int
        Offset of aligned positions. Sometimes the doc spans would be added to the back of
        a question text, in this case, the new position should add len(question_text).
    default_value: int
        The default value to return if the positions are not in the doc span.
    all_in_span: bool
        If set to True, then as long as one position is out of span, all positions
        would be set to default_value.

    Returns
    -------
    list: a list of aligned positions

    Examples
    --------
    >>> positions = [2, 6]
    >>> doc_span_indices = [1, 4]
    >>> align_position2doc_spans(positions, doc_span_indices, default_value=-2)
    [-2, -2]
    >>> align_position2doc_spans(positions, doc_span_indices, default_value=-2, all_in_span=False)
    [1, -2]
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


SquadExample = collections.namedtuple('SquadExample', [
    'qas_id', 'question_text', 'paragraph_text', 'doc_tokens', 'example_id', 'orig_answer_text',
    'start_position', 'end_position', 'start_offset', 'end_offset', 'is_impossible'
])


def convert_squad_examples(record, is_training):
    """read a single entry of gluonnlp.data.SQuAD and convert it to an example.

    Parameters
    ----------
    record: list
        An entry of gluonnlp.data.SQuAD
    is_training: bool
        If the example is used for training,
        then a rough start/end position will be generated

    Returns
    -------
    SquadExample: An instance of SquadExample
    """
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
        start_position = char_to_word_offset[answer_offset] if not is_impossible else -1
        end_position = char_to_word_offset[answer_offset + answer_length -
                                           1] if not is_impossible else -1
    answer_offset = -1 if is_impossible else answer_offset
    example = SquadExample(
        qas_id=qas_id, question_text=question_text, paragraph_text=paragraph_text,
        doc_tokens=doc_tokens, example_id=example_id, orig_answer_text=orig_answer_text,
        start_position=start_position, end_position=end_position, start_offset=answer_offset,
        end_offset=answer_offset + len(orig_answer_text) - 1, is_impossible=is_impossible)
    return example
