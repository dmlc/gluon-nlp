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
    '--format',
    type=str,
    default='numpy',
    choices=['numpy', 'recordio'],
    help='Output file format. If set to "numpy", examples are serialized as a single "npz" file.'
         'If set to "recordio", examples are serialized using IndexedRecordIO format.')

parser.add_argument(
    '--vocab',
    type=str,
    default=None,
    choices=['book_corpus_wiki_en_uncased', 'book_corpus_wiki_en_cased',
             'wiki_multilingual_uncased', 'wiki_multilingual_cased', 'wiki_cn_cased'],
    help='The dataset name for the vocab file BERT model was trained on. For example, '
         '"book_corpus_wiki_en_uncased"')

parser.add_argument(
    '--max_seq_length', type=int, default=128, help='Maximum sequence length.')

parser.add_argument(
    '--max_predictions_per_seq',
    type=int,
    default=20,
    help='Maximum number of masked LM predictions per sequence.')

parser.add_argument(
    '--random_seed',
    type=int,
    default=12345,
    help='Random seed for data generation.')

parser.add_argument(
    '--dupe_factor',
    type=int,
    default=10,
    help='Number of times to duplicate the input data (with different masks).')

parser.add_argument(
    '--masked_lm_prob',
    type=float,
    default=0.15,
    help='Masked LM probability.')

parser.add_argument(
    '--short_seq_prob',
    type=float,
    default=0.1,
    help='Probability of creating sequences which are shorter than the '
    'maximum length.')

parser.add_argument(
    '--verbose',
    action='store_true',
    help='Print debug information')

parser.add_argument(
    '--num_workers',
    type=int,
    default=1,
    help='Number of workers for parallel processing, where each worker generates an output file')

parser.add_argument(
    '--num_outputs',
    type=int,
    default=1,
    help='Number of desired output files, where each one is processed independently by a worker.')

parser.add_argument(
    '--tokenized',
    action='store_true',
    help='Use --tokenized to indicate that the input data is already tokenized. '
         'Hence tokenization will be skipped when generating training samples. '
         'Setting this may accelerate the generation speed')

args = parser.parse_args()
logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)
logging.info(args)


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions,
                 masked_lm_labels, is_random_next, segment_a_lengths, segment_b_lengths):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels
        self.segment_a_lengths = segment_a_lengths
        self.segment_b_lengths = segment_b_lengths

    def __str__(self):
        s = ''
        s += 'tokens: %s\n' % (' '.join(self.tokens))
        s += 'segment_ids: %s\n' % (' '.join(
            [str(x) for x in self.segment_ids]))
        s += 'is_random_next: %s\n' % self.is_random_next
        s += 'masked_lm_positions: %s\n' % (' '.join(
            [str(x) for x in self.masked_lm_positions]))
        s += 'masked_lm_labels: %s\n' % (' '.join(self.masked_lm_labels))
        s += '\n'
        return s

    def __repr__(self):
        return self.__str__()

def transform(instance, tokenizer, max_seq_length, max_predictions_per_seq, do_pad=True):
    """Transform instance to inputs for MLM and NSP."""
    pad = tokenizer.convert_tokens_to_ids(['[PAD]'])[0]
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length
    valid_lengths = len(input_ids)

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(
        instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)
    masked_lm_valid_lengths = len(masked_lm_ids)
    if do_pad:
        while len(input_ids) < max_seq_length:
            input_ids.append(pad)
            # Padding index MUST be defined to 0 on input_mask, segment_ids
            input_mask.append(0)
            segment_ids.append(0)
        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(pad)
            masked_lm_weights.append(0.0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

    next_sentence_label = 1 if instance.is_random_next else 0

    features = {}
    features['input_ids'] = input_ids
    features['input_mask'] = input_mask
    features['segment_ids'] = segment_ids
    features['masked_lm_positions'] = masked_lm_positions
    features['masked_lm_ids'] = masked_lm_ids
    features['segment_a_lengths'] = [instance.segment_a_lengths]
    features['segment_b_lengths'] = [instance.segment_b_lengths]
    features['masked_lm_ids'] = masked_lm_ids
    features['masked_lm_weights'] = masked_lm_weights
    features['next_sentence_labels'] = [next_sentence_label]
    features['valid_lengths'] = [valid_lengths]
    features['masked_lm_valid_lengths'] = [masked_lm_valid_lengths]

    return features

def print_example(instance, features):
    logging.debug('*** Example ***')
    logging.debug('tokens: %s' % ' '.join(instance.tokens))  # pylint: disable=W1201

    for feature_name in features.keys():
        feature = features[feature_name]
        logging.debug('%s: %s', feature_name, feature)

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

    np.savez(output_file, **outputs)
    logging.info('Wrote %d total instances', total_written)

def write_to_files_rec(instances, tokenizer, max_seq_length,
                       max_predictions_per_seq, output_files):
    """Create IndexedRecordIO files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(
            mx.recordio.MXIndexedRecordIO(
                os.path.splitext(output_file)[0] + '.idx', output_file, 'w'))

    writer_index = 0
    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        features = transform(instance, tokenizer, max_seq_length, max_predictions_per_seq)
        example = features

        writers[writer_index].write_idx(inst_index, json.dumps(example).encode('UTF-8'))
        writer_index = (writer_index + 1) % len(writers)
        total_written += 1

    for writer in writers:
        writer.close()

    logging.info('Wrote %d total instances', total_written)


def create_training_instances(x):
    """Create `TrainingInstance`s from raw text."""
    (input_files, out, tokenizer, max_seq_length, dupe_factor,
     short_seq_prob, masked_lm_prob, max_predictions_per_seq, rng) = x
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
        with io.open(input_file, 'r', encoding='UTF-8') as reader:
            while True:
                line = reader.readline()
                if not line:
                    break
                line = line.strip()

                # Empty lines are used as document delimiters
                if not line:
                    all_documents.append([])
                tokens = tokenizer(line) if not args.tokenized else line.split(' ')
                if tokens:
                    all_documents[-1].append(tokens)

    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = tokenizer.vocab.idx_to_token
    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document(
                    all_documents, document_index, max_seq_length,
                    short_seq_prob, masked_lm_prob, max_predictions_per_seq,
                    vocab_words, rng))

    rng.shuffle(instances)
    input_ids = []
    segment_ids = []
    masked_lm_positions = []
    masked_lm_ids = []
    masked_lm_weights = []
    next_sentence_labels = []
    valid_lengths = []

    for inst_index, instance in enumerate(instances):
        feature = transform(instance, tokenizer, max_seq_length, max_predictions_per_seq, False)
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
        if inst_index < 20:
            print_example(instance, feature)

    features = (input_ids, segment_ids, masked_lm_positions, masked_lm_ids, \
                masked_lm_weights, next_sentence_labels, valid_lengths)

    logging.info('*** Writing to output file %s ***', out)
    if args.format == 'numpy':
        write_to_files_np(features, tokenizer, args.max_seq_length,
                          args.max_predictions_per_seq, [out])
    elif args.format == 'recordio':
        write_to_files_rec(instances, tokenizer, args.max_seq_length,
                           args.max_predictions_per_seq, [out])
    else:
        raise ValueError('unsupported format: %s'%args.format)

    time_end = time.time()
    logging.info('Process %d files took %.1f s', len(input_files), time_end - time_start)


def create_instances_from_document(
        all_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

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
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = rng.randint(
                            0,
                            len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
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
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append('[CLS]')
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)
                tokens.append('[SEP]')
                segment_ids.append(0)
                segment_a_lengths = len(segment_ids)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append('[SEP]')
                segment_ids.append(1)
                segment_b_lengths = len(segment_ids) - segment_a_lengths

                (tokens, masked_lm_positions,
                 masked_lm_labels) = create_masked_lm_predictions(
                     tokens, masked_lm_prob, max_predictions_per_seq,
                     vocab_words, rng)
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels,
                    segment_a_lengths=segment_a_lengths,
                    segment_b_lengths=segment_b_lengths)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple('MaskedLmInstance',
                                          ['index', 'label'])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token in ['[CLS]', '[SEP]']:
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

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
        if rng.random() < 0.8:
            masked_token = '[MASK]'
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_words[rng.randint(0,
                                                       len(vocab_words) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def main():
    """Main function."""
    time_start = time.time()
    logging.info('loading vocab file from dataset: %s', args.vocab)
    vocab_obj = nlp.data.utils._load_pretrained_vocab(args.vocab)
    tokenizer = BERTTokenizer(
        vocab=vocab_obj, lower='uncased' in args.vocab)

    input_files = []
    for input_pattern in args.input_file.split(','):
        input_files.extend(glob.glob(os.path.expanduser(input_pattern)))

    logging.info('*** Reading from %d input files ***', len(input_files))
    for input_file in input_files:
        logging.info('  %s', input_file)

    num_outputs = args.num_outputs
    assert len(input_files) >= num_outputs, \
        'Number of outputs must be fewer than that of inputs'

    output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    rng = random.Random(args.random_seed)
    nworker = args.num_workers

    # calculate the number of splits
    file_splits = []
    split_size = (len(input_files) + num_outputs - 1) // num_outputs
    for i in range(num_outputs - 1):
        file_splits.append(input_files[i*split_size:(i+1)*split_size])
    file_splits.append(input_files[(num_outputs-1)*split_size:])

    # prepare workload
    suffix = 'npz' if args.format == 'numpy' else 'rec'
    count = 0
    map_args = []
    pool_args = (tokenizer, args.max_seq_length, args.dupe_factor,\
                 args.short_seq_prob, args.masked_lm_prob,
                 args.max_predictions_per_seq, rng)
    for i, file_split in enumerate(file_splits):
        out = os.path.join(output_dir, 'part-{}.{}'.format(str(i).zfill(3), suffix))
        count += len(file_split)
        map_args.append((file_split, out) + pool_args)

    # sanity check
    assert count == len(input_files)

    # dispatch to workers
    if nworker > 1:
        pool = Pool(nworker)
        pool.map(create_training_instances, map_args)
    else:
        for map_arg in map_args:
            create_training_instances(map_arg)

    time_end = time.time()
    logging.info('Time cost=%.1f', time_end - time_start)

if __name__ == '__main__':
    main()
