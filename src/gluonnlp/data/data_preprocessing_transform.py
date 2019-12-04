"""glue and squad transform."""

__all__ = [
    'TruncateTransform', 'ConcatSeqTransform',
    'BertStyleGlueTransform', 'BertStyleSQuADTransform',
    'SQuADExampleTransform', 'DocSpanTransform',
    'TokenizeAndPositionAlignTransform', 'SimpleQAPreparation', 'SquadExample'
]

import collections
import itertools
from functools import partial
import numpy as np
from gluonnlp.data.utils import whitespace_splitter
from .qa_preprocessing_utils import truncate_seqs_equal, improve_answer_span


class TruncateTransform:
    """
    Truncate a sequence(list) to max length.

    Parameters
    ----------
    max_len : int
    truncate_fn : callable
        A function determines how to truncate the sequence (list).
        The function should implement argument max_len, and return a list

    Returns
    -------
    list : list
    """
    def __init__(self, max_len, truncate_fn=truncate_seqs_equal):
        self._max_len = max_len
        self.fn = partial(truncate_fn, max_len=max_len)

    def __call__(self, seqs):
        assert isinstance(seqs, collections.abc.Iterable)
        if len(seqs) == 0:
            return seqs
        if not isinstance(seqs[0], collections.abc.Iterable) or isinstance(seqs[0], str):
            seqs = [seqs]

        seqs = self.fn(seqs)
        return seqs


class ConcatSeqTransform:
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
    def __call__(self, seqs, separators):
        assert isinstance(seqs, collections.abc.Iterable) and len(seqs) > 0
        concat = sum((seq + sep for sep, seq in
                      itertools.zip_longest(separators, seqs, fillvalue=[])), [])
        segment_ids = sum(([i] * (len(seq) + len(sep)) for i, (sep, seq) in
                           enumerate(itertools.zip_longest(separators, seqs, fillvalue=[]))), [])
        p_mask = sum(([0] * len(seq) + [1] * len(sep) for sep, seq in
                      itertools.zip_longest(separators, seqs, fillvalue=[])), [])
        return concat, segment_ids, p_mask


class BertStyleGlueTransform:
    """
    Convert from gluonnlp.data.Glue* record to inputs for BERT-style model.
    """
    def __init__(self,
                 tokenizer,
                 max_seq_length,
                 cls_token=None,
                 sep_token=None,
                 task=None,
                 class_labels=None,
                 label_alias=None,
                 vocab=None):
        self._vocab = tokenizer.vocab if vocab is None else vocab
        self.class_labels = task.class_labels if task else class_labels
        self._label_dtype = 'int32' if (task
                                        and task.class_labels) else 'float32'
        self.label_alias = task.label_alias if task else label_alias
        if self.class_labels:
            self._label_map = {}
            for (i, label) in enumerate(self.class_labels):
                self._label_map[label] = i
            if self.label_alias:
                for key in self.label_alias:
                    self._label_map[key] = self._label_map[
                        self.label_alias[key]]
            if len(class_labels) > 1:
                max_seq_length += 3  # account for special tokens
            else:
                max_seq_length += 2

        self.Truncate = TruncateTransform(max_len=max_seq_length)
        self.InsertSpecialTokens = ConcatSeqTransform()
        self._tokenizer = tokenizer
        self._sep_token = sep_token
        self._cls_token = cls_token

    def __call__(self, line):
        #process the token pair
        tokens_raw = [self._tokenizer(l) for l in line[:-1]]
        tokens_trun = self.Truncate(tokens_raw)
        tokens, segment_ids, _ = self.InsertSpecialTokens(
            tokens_trun, [[self._sep_token]] * len(tokens_trun))

        #add cls token
        input_ids = self._vocab[[self._cls_token] + tokens]
        segment_ids = [0] + segment_ids

        #get label
        label = line[-1]
        # map to int if class labels are available
        if self.class_labels:
            label = self._label_map[label]
        label = np.array([label], dtype=self._label_dtype)
        return input_ids, segment_ids, label



SquadExample = collections.namedtuple('SquadExample', [
    'qas_id', 'question_text', 'doc_tokens', 'example_id', 'orig_answer_text',
    'start_position', 'end_position', 'is_impossible'
])


class SQuADExampleTransform:
    """
    Convert from gluonnlp.data.SQuAD's record to SquadExample.
    """
    def __init__(self, training=True):
        self.is_training = training

    def _is_whitespace(self, c):
        if c == ' ' or c == '\t' or c == '\r' or c == '\n' or ord(c) == 0x202F:
            return True
        return False

    def __call__(self, record):
        example_id = record[0]
        qas_id = record[1]
        question_text = record[2]
        paragraph_text = record[3]
        orig_answer_text = record[4][0] if record[4] else ''
        answer_offset = record[5][0] if record[5] else ''
        is_impossible = record[6] if len(record) == 7 else False

        doc_tokens = []

        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if self._is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)
        start_position = -1
        end_position = -1

        if self.is_training:
            if not is_impossible:
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                end_position = char_to_word_offset[answer_offset +
                                                   answer_length - 1]
                # Only add answers where the text can be exactly recovered from the
                # document. If this CAN'T happen it's likely due to weird Unicode
                # stuff so we will just skip the example.
                #
                # Note that this means for training mode, every example is NOT
                # guaranteed to be preserved.
                actual_text = ' '.join(
                    doc_tokens[start_position:(end_position + 1)])
                cleaned_answer_text = ' '.join(
                    whitespace_splitter(orig_answer_text.strip()))
                if actual_text.find(cleaned_answer_text) == -1:
                    print('Could not find answer: %s vs. %s' %
                          (actual_text, cleaned_answer_text))
                    return None
            else:
                start_position = -1
                end_position = -1
                orig_answer_text = ''

        example = SquadExample(qas_id=qas_id,
                               question_text=question_text,
                               doc_tokens=doc_tokens,
                               example_id=example_id,
                               orig_answer_text=orig_answer_text,
                               start_position=start_position,
                               end_position=end_position,
                               is_impossible=is_impossible)
        return example


class TokenizeAndPositionAlignTransform:
    """Tokenize the question and paragraph text and map the origin start/end position
    to the right position in tokenized text.
    """
    def __init__(self, tokenizer, max_query_length, is_training):
        self._tokenizer = tokenizer
        self._max_query_length = max_query_length
        self.is_training = is_training

    def __call__(self, example):
        # tokenize the query text
        query_tokens = self._tokenizer(example.question_text)
        if len(query_tokens) > self._max_query_length:
            query_tokens = query_tokens[0:self._max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []

        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = self._tokenizer(token)
            tok_to_orig_index += [i] * len(sub_tokens)
            all_doc_tokens += sub_tokens
        # tokenize the paragraph text and align the start/end position
        # to the tokenized sequence position
        tok_start_position = None
        tok_end_position = None
        if self.is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if self.is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position +
                                                     1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position,
                self._tokenizer, example.orig_answer_text)
        return tok_start_position, tok_end_position, all_doc_tokens, query_tokens


class DocSpanTransform:
    """
    We can have documents that are longer than the maximum sequence length.
    To deal with this we do a sliding window approach, where we take chunks
    of the up to our max length with a stride of `doc_stride`.
    """
    def __init__(self, doc_stride, max_seq_length=None):
        self._doc_stride = doc_stride
        self._max_seq_length = max_seq_length

    def __call__(self,
                 all_doc_tokens,
                 max_tokens_for_doc=None,
                 query_tokens_length=None):
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            'DocSpan', ['start', 'length'])
        assert max_tokens_for_doc or (self._max_seq_length
                                      and query_tokens_length)
        doc_spans = []
        start_offset = 0
        if not max_tokens_for_doc:
            max_tokens_for_doc = self._max_seq_length - query_tokens_length - 3
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, self._doc_stride)
        return doc_spans


class SimpleQAPreparation:
    """
    Give the tokenized query text and doc spans, convert the data to BERT-style model input.
    Note that this class does not check if max span.
    """
    def __init__(self, cls_token, sep_token, vocab, is_training):
        self.insert = ConcatSeqTransform()
        self.is_training = is_training
        self._cls_token = cls_token
        self._sep_token = sep_token
        self._vocab = vocab

    def __call__(self,
                 query,
                 doc_spans,
                 all_doc_tokens,
                 tok_start_position,
                 tok_end_position,
                 other_features=None,
                 is_impossible=False):
        ret = []
        if not isinstance(other_features, list):
            other_features = [other_features]
        for doc_span in doc_spans:
            span_text = all_doc_tokens[doc_span.start:doc_span.start +
                                       doc_span.length]
            tokens, segment_ids, p_mask = self.insert(
                [query, span_text], [[self._sep_token]] * 2)
            input_ids = self._vocab[[self._cls_token] + tokens]
            segment_ids = [0] + segment_ids
            p_mask = [0] + p_mask
            start_position = 0
            end_position = 0
            if self.is_training and not is_impossible:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start
                        and tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    doc_offset = len(query) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if self.is_training and is_impossible:
                start_position = 0
                end_position = 0
            if not other_features:
                other_features = []
            ret.append(other_features + [
                input_ids, segment_ids, p_mask, start_position,
                end_position, is_impossible
            ])
        return ret


class BertStyleSQuADTransform:
    """Dataset Transformation for BERT-style QA.

        The transformation is processed in the following steps:
        - Convert from gluonnlp.data.SQuAD's record to SquadExample.
        - Tokenize the question_text in the example.
        - For examples where the document is too long,
          use a sliding window to split into multiple features and
          record whether each token is a maximum context.
        - Tokenize the split document chunks.
        - Combine the token of question_text with the token
          of the document and insert [CLS] and [SEP].
        - Generate the start position and end position of the answer.
        - Generate valid length.
    """
    def __init__(self,
                 tokenizer,
                 vocab=None,
                 max_seq_length=384,
                 doc_stride=128,
                 max_query_length=64,
                 is_training=True):

        self._vocab = tokenizer.vocab if vocab is None else vocab
        # RoBERTa does not register CLS token and SEP token
        if hasattr(self._vocab, 'cls_token'):
            self._cls_token = self._vocab.cls_token
        else:
            self._cls_token = self._vocab.bos_token
        if hasattr(self._vocab, 'sep_token'):
            self._sep_token = self._vocab.sep_token
        else:
            self._sep_token = self._vocab.eos_token

        self.get_example = SQuADExampleTransform(training=is_training)
        self.get_aligned = TokenizeAndPositionAlignTransform(
            tokenizer, max_query_length, is_training)
        self.doc_span_transform = DocSpanTransform(doc_stride, max_seq_length)
        self.doc_span_preparation = SimpleQAPreparation(
            self._cls_token, self._sep_token, self._vocab, is_training)

    def __call__(self, line):
        example = self.get_example(line)
        tok_start_position, tok_end_position, all_doc_tokens, query_tokens = self.get_aligned(
            example)
        doc_spans = self.doc_span_transform(
            all_doc_tokens, query_tokens_length=len(query_tokens))
        #features contain example_id,input_ids, segment_ids,
        # valid_length, start_position, end_position
        features = self.doc_span_preparation(
            query_tokens,
            doc_spans,
            all_doc_tokens,
            tok_start_position,
            tok_end_position,
            other_features=example.example_id,
            is_impossible=example.is_impossible)
        return features
