# coding=utf-8
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
"""BERT datasets."""

__all__ = ['MRPCDataset', 'ClassificationTransform']

import os
import numpy as np
try:
    from tokenization import convert_to_unicode
except ImportError:
    from .tokenization import convert_to_unicode
from gluonnlp.data import TSVDataset
from gluonnlp.data.registry import register

@register(segment=['train', 'dev', 'test'])
class MRPCDataset(TSVDataset):
    """The Microsoft Research Paraphrase Corpus dataset.

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'val', 'test' or their combinations.
    root : str, default '$GLUE_DIR/MRPC'
        Path to the folder which stores the MRPC dataset.
        The datset can be downloaded by the following script:
        https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e
    """
    def __init__(self, segment='train',
                 root=os.path.join(os.getenv('GLUE_DIR', 'glue_data'), 'MRPC')):
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, 'Unsupported segment: %s'%segment
        path = os.path.join(root, '%s.tsv'%segment)
        A_IDX, B_IDX, LABEL_IDX = 3, 4, 0
        fields = [A_IDX, B_IDX, LABEL_IDX]
        super(MRPCDataset, self).__init__(path, num_discard_samples=1, field_indices=fields)

    @staticmethod
    def get_labels():
        """Get classification label ids of the dataset."""
        return ['0', '1']

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class ClassificationTransform(object):
    """Dataset Transformation for BERT-style Sentence Classification.

    Parameters
    ----------
    tokenizer : BasicTokenizer or FullTokensizer.
        Tokenizer for the sentences.
    labels : list of int.
        List of all label ids for the classification task.
    max_seq_length : int.
        Maximum sequence length of the sentences.
    pad : bool, default True
        Whether to pad the sentences to maximum length.
    pair : bool, default True
        Whether to transform sentences or sentence pairs.
    """
    def __init__(self, tokenizer, labels, max_seq_length, pad=True, pair=True):
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._pad = pad
        self._label_map = {}
        for (i, label) in enumerate(labels):
            self._label_map[label] = i
        self._pair = pair

    def __call__(self, line):
        """Perform transformation for sequence pairs or single sequences.

        The transformation is processed in the following steps:
        - tokenize the input sequences
        - insert [CLS], [SEP] as necessary
        - generate type ids to indicate whether a token belongs to the first
          sequence or the second sequence.
        - generate valid length

        For sequence pairs, the input is a tuple of 3 strings:
        text_a, text_b and label.

        Inputs:
            text_a: 'is this jacksonville ?'
            text_b: 'no it is not'
            label: '0'
        Tokenization:
            text_a: 'is this jack ##son ##ville ?'
            text_b: 'no it is not .'
        Processed:
            tokens:  '[CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]'
            type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
            valid_length: 14
            label: 0

        For single sequences, the input is a tuple of 2 strings: text_a and label.
        Inputs:
            text_a: 'the dog is hairy .'
            label: '1'
        Tokenization:
            text_a: 'the dog is hairy .'
        Processed:
            text_a:  '[CLS] the dog is hairy . [SEP]'
            type_ids: 0     0   0   0  0     0 0
            valid_length: 7
            label: 1

        Parameters
        ----------
        line: tuple of str
            Input strings. For sequence pairs, the input is a tuple of 3 strings:
            (text_a, text_b, label). For single sequences, the input is a tuple
            of 2 strings: (text_a, label).

        Returns
        -------
        np.array: input token ids in 'int32', shape (batch_size, seq_length)
        np.array: valid length in 'int32', shape (batch_size,)
        np.array: input token type ids in 'int32', shape (batch_size, seq_length)
        np.array: label id in 'int32', shape (batch_size, 1)
        """
        # convert to unicode
        text_a = line[0]
        label = line[-1]
        text_a = convert_to_unicode(text_a)
        label = convert_to_unicode(label)
        if self._pair:
            assert len(line) == 3
            text_b = line[1]
            text_b = convert_to_unicode(text_b)

        tokens_a = self._tokenizer.tokenize(text_a)
        tokens_b = None

        if self._pair:
            tokens_b = self._tokenizer.tokenize(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, self._max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self._max_seq_length - 2:
                tokens_a = tokens_a[0:(self._max_seq_length - 2)]

        # The embedding vectors for `type=0` and `type=1` were learned during
        # pre-training and are added to the wordpiece embedding vector
        # (and position vector). This is not *strictly* necessary since
        # the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.

        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append('[CLS]')
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append('[SEP]')
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append('[SEP]')
            segment_ids.append(1)

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)
        label_id = self._label_map[label]

        # The valid length of sentences. Only real  tokens are attended to.
        valid_length = len(input_ids)

        if self._pad:
            # Zero-pad up to the sequence length.
            padding_length = self._max_seq_length - valid_length
            input_ids.extend([0] * padding_length)
            segment_ids.extend([0] * padding_length)

        return np.array(input_ids, dtype='int32'), np.array(valid_length, dtype='int32'),\
               np.array(segment_ids, dtype='int32'), np.array([label_id], dtype='int32')
