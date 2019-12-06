"""glue and squad transform."""

__all__ = [
    'BertStyleGlueTransform','BertStyleSQuADTransform',
    'SQuADExampleTransform', 'SquadExample'
]

import collections
import numpy as np
from .preprocessing_utils import truncate_seqs_equal, improve_answer_span, \
    ConcatSeqTransform, TokenizeAndPositionAlign, get_doc_spans, align_position2doc_spans

class BertStyleGlueTransform:
    """Convert from gluonnlp.data.Glue* record to inputs for BERT-style model."""
    def __init__(self,
                 tokenizer,
                 truncate_length,
                 cls_token=None,
                 sep_token=None,
                 class_labels=None,
                 label_dtype='float32',
                 label_alias=None,
                 vocab=None):
        self._vocab = tokenizer.vocab if vocab is None else vocab
        self.class_labels = class_labels
        self._label_dtype = label_dtype
        self.label_alias = label_alias
        if self.class_labels:
            self._label_map = {}
            for (i, label) in enumerate(self.class_labels):
                self._label_map[label] = i
            if self.label_alias:
                for key in self.label_alias:
                    self._label_map[key] = self._label_map[
                        self.label_alias[key]]
            truncate_length += 3 if len(class_labels) > 1 else 2
        self._truncate_length = truncate_length
        self._tokenizer = tokenizer
        self._sep_token = sep_token
        self._cls_token = cls_token

    def __call__(self, line):
        #process the token pair
        tokens_raw = [self._tokenizer(l) for l in line[:-1]]
        tokens_trun = truncate_seqs_equal(tokens_raw, self._truncate_length)
        tokens_trun[0] = [self._cls_token] + tokens_trun[0]
        tokens, segment_ids, _ = ConcatSeqTransform(
            tokens_trun, [[self._sep_token]] * len(tokens_trun))
        input_ids = self._vocab[tokens]
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
    """Convert from gluonnlp.data.SQuAD's record to SquadExample."""
    def __init__(self, training=True, version_2=False):
        self.is_training = training
        self._version_2 = version_2

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

        answer_length = len(orig_answer_text)
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

        start_position = char_to_word_offset[answer_offset] if not is_impossible else -1
        end_position = char_to_word_offset[answer_offset + answer_length -1] if not is_impossible else -1

        example = SquadExample(qas_id=qas_id,
                               question_text=question_text,
                               doc_tokens=doc_tokens,
                               example_id=example_id,
                               orig_answer_text=orig_answer_text,
                               start_position=start_position,
                               end_position=end_position,
                               is_impossible=is_impossible)
        return example


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
                 cls_token,
                 sep_token,
                 vocab=None,
                 max_seq_length=384,
                 doc_stride=128,
                 max_query_length=64,
                 is_training=True):
        self._tokenizer = tokenizer
        self._vocab = tokenizer.vocab if vocab is None else vocab
        self._cls_token = cls_token
        self._sep_token = sep_token
        self._max_seq_length = max_seq_length
        self._doc_stride = doc_stride
        self._max_query_length = max_query_length
        self._get_example = SQuADExampleTransform(training=is_training)

    def __call__(self, line):
        example = self._get_example(line)
        query_tokenized = [self._cls_token] + self._tokenizer(example.question_text)[:self._max_query_length]
        #get the start/end position of the answer in tokenized paragraph
        (tok_start_position, tok_end_position), all_doc_tokens = \
            TokenizeAndPositionAlign(example.doc_tokens,
                                      [example.start_position,
                                      example.end_position],
                                      self._tokenizer)
        if not example.is_impossible:
            (tok_start_position, tok_end_position) = improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position,
                self._tokenizer, example.orig_answer_text)
        else:
            tok_start_position, tok_end_position = -1, -1

        #get doc spans
        doc_spans, doc_spans_indices = get_doc_spans(all_doc_tokens, self._max_seq_length - self._max_query_length - 3,
                                  self._doc_stride)
        #get sequence features: tokens, segment_ids, p_masks
        seq_features = [ConcatSeqTransform([query_tokenized, doc_span], [[self._sep_token]] * 2)
                    for doc_span in doc_spans]
        #get the new start/end position
        positions = [align_position2doc_spans([tok_start_position, tok_end_position], doc_idx,
                                              offset=len(query_tokenized) + 1,
                                              default_value=0) for doc_idx in doc_spans_indices]
        features = [[example.example_id] + [self._vocab[tokens], segment_id, p_mask]
                    + [start, end, example.is_impossible]
                    for (tokens, segment_id, p_mask), (start, end) in zip(seq_features, positions)]

        return features
