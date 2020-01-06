# Copyright 2018 The Google AI Language Team Authors and DMLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""XLNet for QA datasets."""
import collections
import multiprocessing as mp
import unicodedata
import gc
from mxnet.gluon.data import SimpleDataset
import numpy as np
import gluonnlp as nlp
__all__ = ['SQuADTransform']


class SquadExample:
    """A single training/test example for SQuAD question.

       For examples without an answer, the start and end position are -1.
    """
    def __init__(self,
                 qas_id,
                 question_text,
                 paragraph_text,
                 example_id,
                 orig_answer_text=None,
                 start_position=None,
                 is_impossible=False):
        self.qas_id = qas_id
        self.question_text = question_text
        self.paragraph_text = paragraph_text
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.is_impossible = is_impossible
        self.example_id = example_id


def convert_single_example_to_input(example):
    """convert a single example into necessary features for model input"""
    features = []
    for _example in example:
        feature = []
        feature.append(_example.example_id)
        feature.append(_example.input_ids)
        feature.append(_example.segment_ids)
        feature.append(_example.valid_length)
        feature.append(_example.p_mask)
        feature.append(_example.start_position)
        feature.append(_example.end_position)
        feature.append(_example.is_impossible)
        feature.append(len(_example.input_ids))
        features.append(feature)
    return features


def convert_examples_to_inputs(examples, num_workers=8):
    """convert examples into necessary features for model input"""
    pool = mp.Pool(num_workers)
    dataset_transform = []
    for data in pool.map(convert_single_example_to_input, examples):
        if data:
            for _data in data:
                dataset_transform.append(_data[:-1])
    dataset = SimpleDataset(dataset_transform).transform(
        lambda x: (x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]))

    pool.close()
    return dataset


def _encode_pieces(sp_model, text, sample=False):
    """apply sentence pieces to raw text"""
    if not sample:
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)

    new_pieces = []
    for piece in pieces:
        if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
            cur_pieces = sp_model.EncodeAsPieces(piece[:-1].replace(u'▁', ''))
            if piece[0] != u'▁' and cur_pieces[0][0] == u'▁':
                if len(cur_pieces[0]) == 1:
                    cur_pieces = cur_pieces[1:]
                else:
                    cur_pieces[0] = cur_pieces[0][1:]
            cur_pieces.append(piece[-1])
            new_pieces.extend(cur_pieces)
        else:
            new_pieces.append(piece)
    return new_pieces


class InputFeatures:
    """A single set of features of data."""
    def __init__(self,
                 example_id,
                 qas_id,
                 doc_span_index,
                 tok_start_to_orig_index,
                 tok_end_to_orig_index,
                 token_is_max_context,
                 input_ids,
                 tokens,
                 valid_length,
                 p_mask,
                 segment_ids,
                 paragraph_text,
                 paragraph_len,
                 cls_index,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.example_id = example_id
        self.qas_id = qas_id
        self.doc_span_index = doc_span_index
        self.tok_start_to_orig_index = tok_start_to_orig_index
        self.tok_end_to_orig_index = tok_end_to_orig_index
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.tokens = tokens
        self.valid_length = valid_length
        self.p_mask = p_mask
        self.segment_ids = segment_ids
        self.paragraph_text = paragraph_text
        self.paragraph_len = paragraph_len
        self.cls_index = cls_index
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def _convert_index(index, pos, M=None, is_start=True):
    """convert tokenized index to corresponding origin text index"""
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


def preprocess_text(inputs, lower=False, remove_space=True,
                    keep_accents=False):
    """simple text clean"""
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


class SQuADTransform:
    """Dataset Transformation for XLNet-style QA.

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

    E.g:

    Inputs:

        question_text: 'When did BBC Japan begin broadcasting?'
        doc_tokens: ['BBC','Japan','was','a','general','entertainment','channel,',
                    'which','operated','between','December','2004','and','April',
                    '2006.','It','ceased','operations','after','its','Japanese',
                    'distributor','folded.']
        start_position: 10
        end_position: 11
        orig_answer_text: 'December 2004'

    Processed:

        tokens: ['when','did','bbc','japan','begin','broadcasting','?',
                '[SEP]','bbc','japan','was','a','general','entertainment','channel',
                ',','which','operated','between','december','2004','and','april',
                '2006','.','it','ceased','operations','after','its','japanese',
                'distributor','folded','.','[SEP]','[CLS]']
        segment_ids: [0,0,0,0,0,0,0,0,1,1,1,1,1,1,
                      1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        start_position: 19
        end_position: 20
        valid_length: 36

    Because of the sliding window approach taken to scoring documents, a single
    token can appear in multiple documents.
    So you need to record whether each token is a maximum context. E.g.
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

    Parameters
    ----------
    tokenizer : XLNetTokenizer.
        Tokenizer for the sentences.
    labels : list of int.
        List of all label ids for the classification task.
    max_seq_length : int, default 384
        Maximum sequence length of the sentences.
    doc_stride : int, default 128
        When splitting up a long document into chunks,
        how much stride to take between chunks.
    max_query_length : int, default 64
        The maximum length of the query tokens.
    is_pad : bool, default True
        Whether to pad the sentences to maximum length.
    is_training : bool, default True
        Whether to run training.
    do_lookup : bool, default True
        Whether to do vocabulary lookup for convert tokens to indices.
    """
    def __init__(self,
                 tokenizer,
                 vocab,
                 max_seq_length=384,
                 doc_stride=128,
                 max_query_length=64,
                 is_pad=True,
                 uncased=False,
                 is_training=True):
        self.tokenizer = tokenizer
        self.sp_model = nlp.data.SentencepieceTokenizer(
            self.tokenizer._sentencepiece_path)._processor
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.doc_stride = doc_stride
        self.is_pad = is_pad
        self.is_training = is_training
        self.uncased = uncased

    def _is_whitespace(self, c):
        if c == ' ' or c == '\t' or c == '\r' or c == '\n' or ord(c) == 0x202F:
            return True
        return False

    def _toSquadExample(self, record):
        example_id = record[0]
        qas_id = record[1]
        question_text = record[2]
        paragraph_text = record[3]
        orig_answer_text = record[4][0] if record[4] else ''
        answer_offset = record[5][0] if record[5] else ''
        is_impossible = record[6] if len(record) == 7 else False

        example = SquadExample(qas_id=qas_id,
                               question_text=question_text,
                               paragraph_text=paragraph_text,
                               example_id=example_id,
                               orig_answer_text=orig_answer_text,
                               start_position=answer_offset,
                               is_impossible=is_impossible)
        return example

    def _transform(self, *record):
        """Loads a data file into a list of `InputBatch`s."""

        example = self._toSquadExample(record)
        max_N, max_M = 1024, 1024
        f = np.zeros((max_N, max_M), dtype=np.float32)
        sp_model = self.sp_model
        query_tokens = self.tokenizer(example.question_text)
        if len(query_tokens) > self.max_query_length:
            query_tokens = query_tokens[0:self.max_query_length]
        query_tokens = self.vocab.to_indices(query_tokens)

        paragraph_text = example.paragraph_text
        para_tokens = _encode_pieces(
            sp_model, preprocess_text(example.paragraph_text, self.uncased))

        chartok_to_tok_index = []
        tok_start_to_chartok_index = []
        tok_end_to_chartok_index = []
        char_cnt = 0
        for i, token in enumerate(para_tokens):
            chartok_to_tok_index.extend([i] * len(token))
            tok_start_to_chartok_index.append(char_cnt)
            char_cnt += len(token)
            tok_end_to_chartok_index.append(char_cnt - 1)

        tok_cat_text = ''.join(para_tokens).replace(u'▁', ' ')
        N, M = len(paragraph_text), len(tok_cat_text)

        if N > max_N or M > max_M:
            max_N = max(N, max_N)
            max_M = max(M, max_M)
            f = np.zeros((max_N, max_M), dtype=np.float32)

        g = {}

        def _lcs_match(max_dist):
            f.fill(0)
            g.clear()

            ### longest common sub sequence
            # f[i, j] = max(f[i - 1, j], f[i, j - 1], f[i - 1, j - 1] + match(i, j))
            for i in range(N):

                # note(zhiliny):
                # unlike standard LCS, this is specifically optimized for the setting
                # because the mismatch between sentence pieces and original text will
                # be small
                for j in range(i - max_dist, i + max_dist):
                    if j >= M or j < 0:
                        continue
                    if i > 0:
                        g[(i, j)] = 0
                        f[i, j] = f[i - 1, j]

                    if j > 0 and f[i, j - 1] > f[i, j]:
                        g[(i, j)] = 1
                        f[i, j] = f[i, j - 1]

                    f_prev = f[i - 1, j - 1] if i > 0 and j > 0 else 0
                    if (preprocess_text(paragraph_text[i],
                                        lower=self.uncased,
                                        remove_space=False) == tok_cat_text[j]
                            and f_prev + 1 > f[i, j]):
                        g[(i, j)] = 2
                        f[i, j] = f_prev + 1

        max_dist = abs(N - M) + 5
        for _ in range(2):
            _lcs_match(max_dist)
            if f[N - 1, M - 1] > 0.8 * N:
                break
            max_dist *= 2

        orig_to_chartok_index = [None] * N
        chartok_to_orig_index = [None] * M
        i, j = N - 1, M - 1
        while i >= 0 and j >= 0:
            if (i, j) not in g:
                break
            if g[(i, j)] == 2:
                orig_to_chartok_index[i] = j
                chartok_to_orig_index[j] = i
                i, j = i - 1, j - 1
            elif g[(i, j)] == 1:
                j = j - 1
            else:
                i = i - 1

        if all(v is None
               for v in orig_to_chartok_index) or f[N - 1, M - 1] < 0.8 * N:
            print('MISMATCH DETECTED!')
            return None

        tok_start_to_orig_index = []
        tok_end_to_orig_index = []
        for i in range(len(para_tokens)):
            start_chartok_pos = tok_start_to_chartok_index[i]
            end_chartok_pos = tok_end_to_chartok_index[i]
            start_orig_pos = _convert_index(chartok_to_orig_index,
                                            start_chartok_pos,
                                            N,
                                            is_start=True)
            end_orig_pos = _convert_index(chartok_to_orig_index,
                                          end_chartok_pos,
                                          N,
                                          is_start=False)

            tok_start_to_orig_index.append(start_orig_pos)
            tok_end_to_orig_index.append(end_orig_pos)

        if not self.is_training:
            tok_start_position = tok_end_position = None

        if self.is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1

        if self.is_training and not example.is_impossible:
            start_position = example.start_position
            end_position = start_position + len(example.orig_answer_text) - 1

            start_chartok_pos = _convert_index(orig_to_chartok_index,
                                               start_position,
                                               is_start=True)
            tok_start_position = chartok_to_tok_index[start_chartok_pos]

            end_chartok_pos = _convert_index(orig_to_chartok_index,
                                             end_position,
                                             is_start=False)
            tok_end_position = chartok_to_tok_index[end_chartok_pos]
            assert tok_start_position <= tok_end_position

        def _piece_to_id(x):
            return sp_model.PieceToId(x)

        all_doc_tokens = list(map(_piece_to_id, para_tokens))

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = self.max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            'DocSpan', ['start', 'length'])
        doc_spans = []
        features = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, self.doc_stride)
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_is_max_context = {}
            segment_ids = []
            p_mask = []

            cur_tok_start_to_orig_index = []
            cur_tok_end_to_orig_index = []

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i

                cur_tok_start_to_orig_index.append(
                    tok_start_to_orig_index[split_token_index])
                cur_tok_end_to_orig_index.append(
                    tok_end_to_orig_index[split_token_index])

                is_max_context = _check_is_max_context(doc_spans,
                                                       doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(0)
                p_mask.append(0)

            paragraph_len = len(tokens)

            #add sep token
            tokens.append(4)
            segment_ids.append(0)
            p_mask.append(1)

            # note(zhiliny): we put P before Q
            # because during pretraining, B is always shorter than A
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(1)
                p_mask.append(1)
            #add sep token
            tokens.append(4)
            segment_ids.append(1)
            p_mask.append(1)

            #add cls token
            tokens.append(3)
            segment_ids.append(2)
            p_mask.append(0)

            input_ids = tokens

            # The mask has 0 for real tokens and 1 for padding tokens. Only real
            # tokens are attended to.
            valid_length = len(input_ids)
            # Zero-pad up to the sequence length.
            cls_index = len(input_ids) - 1
            while len(input_ids) < self.max_seq_length:
                padding_length = self.max_seq_length - valid_length
                input_ids = input_ids + [0] * padding_length
                segment_ids = segment_ids + [3] * padding_length
                p_mask = p_mask + [1] * padding_length

            assert len(input_ids) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length
            assert len(p_mask) == self.max_seq_length

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            if self.is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start
                        and tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    # continue
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    # note(zhiliny): we put P before Q, so doc_offset should be zero.
                    # doc_offset = len(query_tokens) + 2
                    doc_offset = 0
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if self.is_training and span_is_impossible:
                start_position = cls_index
                end_position = cls_index

            if example.example_id < 20:
                print('*** Example ***')
                print('qas_id: %s' % (example.qas_id))
                print('example_index: %s' % (example.example_id))
                print('doc_span_index: %s' % (doc_span_index))
                print('tok_start_to_orig_index: %s' %
                      ' '.join([str(x) for x in cur_tok_start_to_orig_index]))
                print('tok_end_to_orig_index: %s' %
                      ' '.join([str(x) for x in cur_tok_end_to_orig_index]))
                print('token_is_max_context: %s' % ' '.join([
                    '%d:%s' % (x, y)
                    for (x, y) in token_is_max_context.items()
                ]))
                print('input_ids: %s' % ' '.join([str(x) for x in input_ids]))
                print('p_mask: %s' % ' '.join([str(x) for x in p_mask]))
                print('segment_ids: %s' %
                      ' '.join([str(x) for x in segment_ids]))

                if self.is_training and span_is_impossible:
                    print('impossible example span')

                if self.is_training and not span_is_impossible:
                    pieces = [
                        sp_model.IdToPiece(token)
                        for token in tokens[start_position :(end_position + 1)]
                    ]
                    answer_text = sp_model.DecodePieces(pieces)
                    print('start_position: %d' %
                          (start_position))
                    print('end_position: %d' % (end_position))
                    print('answer: %s' % (answer_text))

                    # note(zhiliny): With multi processing,
                    # the example_index is actually the index within the current process
                    # therefore we use example_index=None to avoid being used in the future.
                    # The current code does not use example_index of training data.
            # if self.is_training:
            #     feat_example_index = None
            # else:
            #     feat_example_index = example.example_id

            feature = InputFeatures(
                example_id=example.example_id,
                qas_id=example.qas_id,
                doc_span_index=doc_span_index,
                tok_start_to_orig_index=cur_tok_start_to_orig_index,
                tok_end_to_orig_index=cur_tok_end_to_orig_index,
                token_is_max_context=token_is_max_context,
                tokens=tokens,
                input_ids=input_ids,
                valid_length=valid_length,
                p_mask=p_mask,
                segment_ids=segment_ids,
                paragraph_text=example.paragraph_text,
                paragraph_len=paragraph_len,
                cls_index=cls_index,
                start_position=start_position,
                end_position=end_position,
                is_impossible=span_is_impossible)
            features.append(feature)

        return features

    def __call__(self, record, evaluate=False):
        examples = self._transform(*record)
        if not examples:
            return None
        features = []

        for _example in examples:
            feature = []
            feature.append(_example.example_id)
            feature.append(_example.input_ids)
            feature.append(_example.segment_ids)
            feature.append(_example.valid_length)
            feature.append(_example.p_mask)
            feature.append(_example.start_position)
            feature.append(_example.end_position)
            feature.append(_example.is_impossible)
            feature.append(len(_example.input_ids))
            features.append(feature)

        return features


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + \
            0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index
