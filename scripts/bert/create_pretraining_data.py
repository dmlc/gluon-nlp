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
"""Create masked LM/next sentence masked_lm examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

PREFER_INT = False
PREFER_INT = True

_PAD_TOKEN = '[PAD]'
_CLS_TOKEN = '[CLS]'
_SEP_TOKEN = '[SEP]'
_MASK_TOKEN = '[MASK]'

if PREFER_INT:
    _PAD_TOKEN = 1
    _CLS_TOKEN = 2
    _SEP_TOKEN = 3
    _MASK_TOKEN = 4

import argparse
import logging
import io
import os
import json
import glob
import collections
import random
import time
from multiprocessing import Pool
import numpy as np
import mxnet as mx
import gluonnlp as nlp
from gluonnlp.data import BERTTokenizer


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions,
                 masked_lm_labels, is_random_next, vocab):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
        self.vocab = vocab

    def __str__(self):
        if self.vocab is not None:
           tks = self.vocab.to_tokens(self.tokens)
           mask_tks = self.vocab.to_tokens(self.masked_lm_labels)
        else:
           tks = self.tokens
           mask_tks = self.masked_lm_labels
        s = ''
        s += 'tokens: %s\n' % (' '.join(tks))
        s += 'segment_ids: %s\n' % (' '.join(
            [str(x) for x in self.segment_ids]))
        s += 'is_random_next: %s\n' % self.is_random_next
        s += 'masked_lm_positions: %s\n' % (' '.join(
            [str(x) for x in self.masked_lm_positions]))
        s += 'masked_lm_labels: %s\n' % (' '.join(mask_tks))
        s += '\n'
        return s

    def __repr__(self):
        return self.__str__()

def transform(instance, vocab, max_seq_length, max_predictions_per_seq, int_inputs):
    """Transform instance to inputs for MLM and NSP."""
    input_ids = vocab[instance.tokens] if not PREFER_INT else instance.tokens
    assert len(input_ids) <= max_seq_length
    segment_ids = instance.segment_ids
    masked_lm_positions = instance.masked_lm_positions
    valid_lengths = len(input_ids)

    masked_lm_ids = vocab[instance.masked_lm_labels] if not PREFER_INT else instance.masked_lm_labels
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    next_sentence_label = 1 if instance.is_random_next else 0

    features = {}
    features['input_ids'] = input_ids
    features['segment_ids'] = segment_ids
    features['masked_lm_positions'] = masked_lm_positions
    features['masked_lm_ids'] = masked_lm_ids
    features['masked_lm_weights'] = masked_lm_weights
    features['next_sentence_labels'] = [next_sentence_label]
    features['valid_lengths'] = [valid_lengths]
    return features

def print_example(instance, features):
    logging.debug('*** Example Instance ***')
    logging.debug('\n%s', instance)

    for feature_name in features.keys():
        feature = features[feature_name]
        logging.debug('Generated %s: %s', feature_name, feature)

def write_to_files_np(features, tokenizer, max_seq_length,
                      max_predictions_per_seq, output_files):
    # pylint: disable=unused-argument
    """Write to numpy files from `TrainingInstance`s."""
    next_sentence_labels = []
    valid_lengths = []

    assert len(output_files) == 1, 'numpy format only support single output file'
    output_file = output_files[0]
    (input_ids, segment_ids, masked_lm_positions, masked_lm_ids,
     masked_lm_weights, next_sentence_labels, valid_lengths) = features
    total_written = len(next_sentence_labels)

    # store variable length numpy array object directly.
    outputs = collections.OrderedDict()
    outputs['input_ids'] = np.array(input_ids, dtype=object)
    outputs['segment_ids'] = np.array(segment_ids, dtype=object)
    outputs['masked_lm_positions'] = np.array(masked_lm_positions, dtype=object)
    outputs['masked_lm_ids'] = np.array(masked_lm_ids, dtype=object)
    outputs['masked_lm_weights'] = np.array(masked_lm_weights, dtype=object)
    outputs['next_sentence_labels'] = np.array(next_sentence_labels, dtype='int32')
    outputs['valid_lengths'] = np.array(valid_lengths, dtype='int32')

    np.savez_compressed(output_file, **outputs)
    logging.info('Wrote %d total instances', total_written)

def create_training_instances(packed_arguments):
    """Create `TrainingInstance`s from raw text.

    Parameters
    ----------
    input_files : list of str
        List of paths to input text files.
    output_file : str or None
        Path to the output file. If None, the result is not serialized.
    tokenizer : BERTTokenizer
        The BERT tokenizer
    max_seq_length : int
        The hard limit of maximum sequence length of sentence pairs
    dupe_factor : int
        Duplication factor.
    short_seq_prob : float
        The probability of producing short sequences.
    masked_lm_prob : float
        The probability of replacing texts with masks/random words/original words.
    max_predictions_per_seq : int
        The hard limit of the number of predictions for masked words
    int_inputs : bool
        Whether the inputs are stored as integers (ids in the vocabulary).
    vocab : BERTVocab
        The BERTVocab
    """
    (input_files, output_file, tokenizer, max_seq_length, dupe_factor, short_seq_prob,
     masked_lm_prob, max_predictions_per_seq, int_inputs, vocab) = packed_arguments

    time_start = time.time()
    logging.info('Processing %s', input_files)
    all_documents = [[]]

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    for input_file in input_files:
        with io.open(input_file, 'r', encoding='utf-8') as reader:
            while True:
                line = reader.readline()
                if not line:
                    break
                line = line.strip()

                # Empty lines are used as document delimiters
                if not line and all_documents[-1]:
                    all_documents.append([])
                if int_inputs:
                    tokens = [int(x) for x in line.split()]
                else:
                    tokens = vocab[tokenizer(line)] if PREFER_INT else tokenizer(line)
                if tokens:
                    all_documents[-1].append(tokens)

    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document(
                    all_documents, document_index, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab))

    input_ids = []
    segment_ids = []
    masked_lm_positions = []
    masked_lm_ids = []
    masked_lm_weights = []
    next_sentence_labels = []
    valid_lengths = []

    for inst_index, instance in enumerate(instances):
        feature = transform(instance, vocab, max_seq_length, max_predictions_per_seq, int_inputs)
        input_ids.append(
            np.ascontiguousarray(feature['input_ids'], dtype='int32'))
        segment_ids.append(
            np.ascontiguousarray(feature['segment_ids'], dtype='int32'))
        masked_lm_positions.append(
            np.ascontiguousarray(feature['masked_lm_positions'], dtype='int32'))
        masked_lm_ids.append(np.ascontiguousarray(feature['masked_lm_ids'], dtype='int32'))
        masked_lm_weights.append(
            np.ascontiguousarray(feature['masked_lm_weights'], dtype='float32'))
        next_sentence_labels.append(feature['next_sentence_labels'][0])
        valid_lengths.append(feature['valid_lengths'][0])
        if inst_index < 1:
            print_example(instance, feature)

    # write output to files. Used when pre-generating files
    if output_file is not None:
        features = (input_ids, segment_ids, masked_lm_positions, masked_lm_ids, \
                    masked_lm_weights, next_sentence_labels, valid_lengths)
        logging.info('*** Writing to output file %s ***', output_file)
        write_to_files_np(features, tokenizer, args.max_seq_length,
                          args.max_predictions_per_seq, [output_file])
    else:
        # return feature vectors. Used when generating samples online
        features = (input_ids, masked_lm_ids, masked_lm_positions, masked_lm_weights,
                    next_sentence_labels, segment_ids, valid_lengths)
    time_end = time.time()
    logging.info('Process %d files took %.1f s', len(input_files), time_end - time_start)
    return features


def create_instances_from_document(
        all_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1, 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random.random() < short_seq_prob:
        target_seq_length = random.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):  # pylint: disable=R1702
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or random.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = random.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = random.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we 'put them back' so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append(_CLS_TOKEN)
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append(_SEP_TOKEN)
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append(_SEP_TOKEN)
                segment_ids.append(1)

                (tokens, masked_lm_positions,
                 masked_lm_labels) = create_masked_lm_predictions(
                     tokens, masked_lm_prob, max_predictions_per_seq, vocab)
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels,
                    vocab=vocab if PREFER_INT else None)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple('MaskedLmInstance',
                                          ['index', 'label'])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token in [_CLS_TOKEN, _SEP_TOKEN]:
            continue
        cand_indexes.append(i)

    random.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        masked_token = None
        # 80% of the time, replace with [MASK]
        if random.random() < 0.8:
            masked_token = _MASK_TOKEN
        else:
            # 10% of the time, keep original
            if random.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                # generate a random word in [0, vocab_size - 1]
                word_idx = random.randint(0, len(vocab) - 1)
                masked_token = vocab.idx_to_token[word_idx] if not PREFER_INT else word_idx

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def main():
    """Main function."""
    time_start = time.time()

    # random seed
    random.seed(args.random_seed)

    # vocabulary
    logging.info('loading vocab file from dataset: %s', args.vocab)
    vocab = nlp.data.utils._load_pretrained_vocab(args.vocab)
    tokenizer = BERTTokenizer(
        vocab=vocab, lower='uncased' in args.vocab)

    # count the number of input files
    input_files = []
    for input_pattern in args.input_file.split(','):
        input_files.extend(glob.glob(os.path.expanduser(input_pattern)))
    logging.info('*** Reading from %d input files ***', len(input_files))
    for input_file in input_files:
        logging.info('\t%s', input_file)
    num_outputs = min(args.num_outputs, len(input_files))

    # create output dir
    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # calculate the number of splits
    nworker = args.num_workers
    file_splits = []
    split_size = (len(input_files) + num_outputs - 1) // num_outputs
    for i in range(num_outputs - 1):
        file_splits.append(input_files[i*split_size:(i+1)*split_size])
    file_splits.append(input_files[(num_outputs-1)*split_size:])

    # prepare workload
    suffix = 'npz'
    count = 0
    packed_args = []
    fixed_args = (tokenizer, args.max_seq_length, args.dupe_factor,\
                  args.short_seq_prob, args.masked_lm_prob,
                  args.max_predictions_per_seq, args.int_inputs and PREFER_INT, vocab)
    for i, file_split in enumerate(file_splits):
        out = os.path.join(output_dir, 'part-{}.{}'.format(str(i).zfill(3), suffix))
        count += len(file_split)
        packed_args.append((file_split, out) + fixed_args)

    # sanity check
    assert count == len(input_files)

    # dispatch to workers
    if nworker > 1:
        pool = Pool(nworker)
        pool.map(create_training_instances, packed_args)
    else:
        for packed_arg in packed_args:
            create_training_instances(packed_arg)

    time_end = time.time()
    logging.info('Time cost=%.1f', time_end - time_start)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-training data generator for BERT')

    parser.add_argument(
        '--input_file',
        type=str,
        default=None,
        help='input file(s). For example, "~/data/*.txt"')

    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory.')

    parser.add_argument(
        '--vocab',
        type=str,
        default=None,
        choices=['book_corpus_wiki_en_uncased', 'book_corpus_wiki_en_cased',
                 'wiki_multilingual_uncased', 'wiki_multilingual_cased', 'wiki_cn_cased'],
        help='The dataset name for the vocab file BERT model was trained on. For example, '
             '"book_corpus_wiki_en_uncased"')

    parser.add_argument(
        '--max_seq_length', type=int, default=128, help='Maximum sequence length. Default is 128.')

    parser.add_argument(
        '--max_predictions_per_seq',
        type=int,
        default=20,
        help='Maximum number of masked LM predictions per sequence. Default is 20.')

    parser.add_argument(
        '--random_seed',
        type=int,
        default=12345,
        help='Random seed for data generation.')

    parser.add_argument(
        '--dupe_factor',
        type=int,
        default=10,
        help='Number of times to duplicate the input data (with different masks). Default is 10.')

    parser.add_argument(
        '--masked_lm_prob',
        type=float,
        default=0.15,
        help='Masked LM probability. Default is 0.15')

    parser.add_argument(
        '--short_seq_prob',
        type=float,
        default=0.1,
        help='Probability of creating sequences which are shorter than the '
        'maximum length. Default is 0.1')

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print debug information')

    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='Number of workers for parallel processing, where each worker generates an output file.'
             ' Default is 1')

    parser.add_argument(
        '--num_outputs',
        type=int,
        default=1,
        help='Number of desired output files, where each one is processed independently by a worker.'
             'Default is 1')

    parser.add_argument(
        '--int-inputs',
        action='store_true',
        help='Use --int-inputs to indicate that the input tokens are already stored as integers. '
             'Tokenization and vocabulary lookup will be skipped when generating training '
             'samples. Enabling this will accelerate the generation speed')

    args = parser.parse_args()
    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logging.info(args)
    main()
