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

__all__ = ['MRPCDataset']

import os
from gluonnlp.data import TSVDataset
from gluonnlp.data.registry import register
import gluonnlp

@register(segment=['train', 'dev', 'test'])
class MRPCDataset(TSVDataset):
    """A Small Translation Dataset for Testing Scripts.

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'val', 'test' or their combinations.
    src_lang : str, default 'en'
        The source language. Option for source and target languages are 'en' <-> 'de'
    tgt_lang : str, default 'de'
        The target language. Option for source and target languages are 'en' <-> 'de'
    root : str, default '$GLUE_DIR/MRPC'
        Path to temp folder for storing data.
    """
    def __init__(self, segment='train',
                 root=os.path.join(os.environ['GLUE_DIR'], 'MRPC')):
        self._supported_segments = ['train', 'dev', 'test']
        assert segment in self._supported_segments, "Unsupported segment: %s"%segment
        path = os.path.join(root, '%s.tsv'%segment)
        super(MRPCDataset, self).__init__(path, num_discard_samples=1)

    @staticmethod
    def get_labels():
        return ["0", "1"]

    #def transform():p

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


import mxnet as mx
import numpy as np


label_list = MRPCDataset.get_labels()
import tokenization
VOCAB_FILE = '/home/ubuntu/bert/uncased_L-12_H-768_A-12/vocab.txt'
do_lower_case=True
tokenizer = tokenization.FullTokenizer(
    vocab_file=VOCAB_FILE, do_lower_case=do_lower_case)

print(tokenizer)
MAX_SEQ_LENGTH = 128

class MPRCTransform(object):
    def __init__(self, tokenizer, labels, max_seq_length, has_text_b=True):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self._label_map = {}
        for (i, label) in enumerate(labels):
          self._label_map[label] = i
        self._has_next_b = has_text_b

    def __call__(self, line):
        assert self._has_next_b
        text_a = tokenization.convert_to_unicode(line[3])
        text_b = tokenization.convert_to_unicode(line[4])
        label = tokenization.convert_to_unicode(line[0])

        tokens_a = self.tokenizer.tokenize(text_a)
        tokens_b = None

        # TODO check text_b
        if self._has_next_b:
           tokens_b = tokenizer.tokenize(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, self.max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self.max_seq_length - 2:
              tokens_a = tokens_a[0:(self.max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
          for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
          tokens.append("[SEP]")
          segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_id = self._label_map[label]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < self.max_seq_length:
          input_ids.append(0)
          input_mask.append(0)
          segment_ids.append(0)

        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        return np.array(input_ids, dtype='int32'), np.array(input_mask, dtype='int32'),\
               np.array(segment_ids, dtype='int32'), np.array([label_id], dtype='int32')

trans = MPRCTransform(tokenizer, label_list, MAX_SEQ_LENGTH)
train_examples2 = MRPCDataset('train')
train_examples2 = train_examples2.transform(trans)
data_mx = train_examples2

dev_examples2 = MRPCDataset('dev')
dev_examples2 = dev_examples2.transform(trans)
data_mx_dev = dev_examples2
