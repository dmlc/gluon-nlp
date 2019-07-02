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
import glob
import collections
import warnings
import random
import time
from multiprocessing import Pool
import numpy as np
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
        tks = self.vocab.to_tokens(self.tokens)
        mask_tks = self.vocab.to_tokens(self.masked_lm_labels)
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

def transform(instance, max_seq_length):
    """Transform instance to inputs for MLM and NSP."""
    input_ids = instance.tokens
    assert len(input_ids) <= max_seq_length
    segment_ids = instance.segment_ids
    masked_lm_positions = instance.masked_lm_positions
    valid_lengths = len(input_ids)

    masked_lm_ids = instance.masked_lm_labels
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

def tokenize_lines_fn(x):
    """Worker function to tokenize lines based on the tokenizer, and perform vocabulary lookup."""
    lines, tokenizer, vocab = x
    results = []
    for line in lines:
        if not line:
            break
        line = line.strip()
        # Empty lines are used as document delimiters
        if not line:
            results.append([])
        else:
            tokens = vocab[tokenizer(line)]
            if tokens:
                results.append(tokens)
    return results

def convert_to_npz(instances, max_seq_length):
    """Create masked language model and next sentence prediction samples as numpy arrays."""
    input_ids = []
    segment_ids = []
    masked_lm_positions = []
    masked_lm_ids = []
    masked_lm_weights = []
    next_sentence_labels = []
    valid_lengths = []

    for inst_index, instance in enumerate(instances):
        features = transform(instance, max_seq_length)
        input_id = features['input_ids']
        segment_id = features['segment_ids']
        masked_lm_position = features['masked_lm_positions']
        masked_lm_id = features['masked_lm_ids']
        masked_lm_weight = features['masked_lm_weights']
        next_sentence_label = features['next_sentence_labels'][0]
        valid_length = features['valid_lengths'][0]

        input_ids.append(np.ascontiguousarray(input_id, dtype='int32'))
        segment_ids.append(np.ascontiguousarray(segment_id, dtype='int32'))
        masked_lm_positions.append(np.ascontiguousarray(masked_lm_position, dtype='int32'))
        masked_lm_ids.append(np.ascontiguousarray(masked_lm_id, dtype='int32'))
        masked_lm_weights.append(np.ascontiguousarray(masked_lm_weight, dtype='float32'))
        next_sentence_labels.append(next_sentence_label)
        valid_lengths.append(valid_length)
        # debugging information
        if inst_index < 1:
            print_example(instance, features)
    return input_ids, masked_lm_ids, masked_lm_positions, masked_lm_weights,\
           next_sentence_labels, segment_ids, valid_lengths

def create_training_instances(x):
    """Create `TrainingInstance`s from raw text.

    The expected input file format is the following:

    (1) One sentence per line. These should ideally be actual sentences, not
    entire paragraphs or arbitrary spans of text. (Because we use the
    sentence boundaries for the "next sentence prediction" task).
    (2) Blank lines between documents. Document boundaries are needed so
    that the "next sentence prediction" task doesn't span between documents.

    The function expect arguments packed in a tuple as described below.

    Parameters
    ----------
    input_files : list of str
        List of paths to input text files.
    tokenizer : BERTTokenizer
        The BERT tokenizer
    max_seq_length : int
        The hard limit of maximum sequence length of sentence pairs
    dupe_factor : int
        Duplication factor.
    short_seq_prob : float
        The probability of sampling sequences shorter than the max_seq_length.
    masked_lm_prob : float
        The probability of replacing texts with masks/random words/original words.
    max_predictions_per_seq : int
        The hard limit of the number of predictions for masked words
    whole_word_mask : bool
        Whether to do masking for whole words
    vocab : BERTVocab
        The BERTVocab
    nworker : int
        The number of processes to help processing texts in parallel
    worker_pool : multiprocessing.Pool
        Must be provided if nworker > 1. The caller is responsible for the destruction of
        the worker pool.
    output_file : str or None
        Path to the output file. If None, the result is not serialized. If provided,
        results are  stored in the order of (input_ids, segment_ids, masked_lm_positions,
        masked_lm_ids, masked_lm_weights, next_sentence_labels, valid_lengths).

    Returns
    -------
    A tuple of np.ndarray : input_ids, masked_lm_ids, masked_lm_positions, masked_lm_weights
                            next_sentence_labels, segment_ids, valid_lengths
    """
    (input_files, tokenizer, max_seq_length, short_seq_prob,
     masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab,
     dupe_factor, nworker, worker_pool, output_file) = x

    time_start = time.time()
    if nworker > 1:
        assert worker_pool is not None

    all_documents = [[]]

    for input_file in input_files:
        with io.open(input_file, 'r', encoding='utf-8') as reader:
            lines = reader.readlines()
            num_lines = len(lines)
            num_lines_per_worker = (num_lines + nworker - 1) // nworker
            process_args = []

            # tokenize in parallel
            for worker_idx in range(nworker):
                start = worker_idx * num_lines_per_worker
                end = min((worker_idx + 1) * num_lines_per_worker, num_lines)
                process_args.append((lines[start:end], tokenizer, vocab))
            if worker_pool:
                tokenized_results = worker_pool.map(tokenize_lines_fn, process_args)
            else:
                tokenized_results = [tokenize_lines_fn(process_args[0])]

            for tokenized_result in tokenized_results:
                for line in tokenized_result:
                    if not line:
                        if all_documents[-1]:
                            all_documents.append([])
                    else:
                        all_documents[-1].append(line)

    # remove the last empty document if any
    if not all_documents[-1]:
        all_documents = all_documents[:-1]

    # generate training instances
    instances = []
    if worker_pool:
        process_args = []
        for document_index in range(len(all_documents)):
            process_args.append((all_documents, document_index, max_seq_length, short_seq_prob,
                                 masked_lm_prob, max_predictions_per_seq, whole_word_mask,
                                 vocab, tokenizer))
        for _ in range(dupe_factor):
            instances_results = worker_pool.map(create_instances_from_document, process_args)
            for instances_result in instances_results:
                instances.extend(instances_result)
        npz_instances = worker_pool.apply(convert_to_npz, (instances, max_seq_length))
    else:
        for _ in range(dupe_factor):
            for document_index in range(len(all_documents)):
                instances.extend(
                    create_instances_from_document(
                        (all_documents, document_index, max_seq_length, short_seq_prob,
                         masked_lm_prob, max_predictions_per_seq, whole_word_mask,
                         vocab, tokenizer)))
        npz_instances = convert_to_npz(instances, max_seq_length)

    (input_ids, masked_lm_ids, masked_lm_positions, masked_lm_weights,
     next_sentence_labels, segment_ids, valid_lengths) = npz_instances

    # write output to files. Used when pre-generating files
    if output_file:
        features = (input_ids, segment_ids, masked_lm_positions, masked_lm_ids,
                    masked_lm_weights, next_sentence_labels, valid_lengths)
        logging.debug('*** Writing to output file %s ***', output_file)
        write_to_files_np(features, tokenizer, max_seq_length,
                          max_predictions_per_seq, [output_file])
        features = None
    else:
        features = (input_ids, masked_lm_ids, masked_lm_positions, masked_lm_weights,
                    next_sentence_labels, segment_ids, valid_lengths)
    time_end = time.time()
    logging.debug('Process %d files took %.1f s', len(input_files), time_end - time_start)
    return features

def create_instances_from_document(x):
    """Creates `TrainingInstance`s for a single document."""
    (all_documents, document_index, max_seq_length, short_seq_prob,
     masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab, tokenizer) = x
    document = all_documents[document_index]
    _MASK_TOKEN = vocab[vocab.mask_token]
    _CLS_TOKEN = vocab[vocab.cls_token]
    _SEP_TOKEN = vocab[vocab.sep_token]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # According to the original tensorflow implementation:
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

                    # randomly choose a document other than itself
                    random_document_index = random.randint(0, len(all_documents) - 2)
                    if random_document_index >= document_index:
                        random_document_index += 1

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
                     tokens, masked_lm_prob, max_predictions_per_seq,
                     whole_word_mask, vocab, tokenizer,
                     _MASK_TOKEN, _CLS_TOKEN, _SEP_TOKEN)
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels,
                    vocab=vocab)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


MaskedLmInstance = collections.namedtuple('MaskedLmInstance',
                                          ['index', 'label'])


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq,
                                 whole_word_mask, vocab, tokenizer,
                                 _MASK_TOKEN, _CLS_TOKEN, _SEP_TOKEN):
    """Creates the predictions for the masked LM objective."""
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token in [_CLS_TOKEN, _SEP_TOKEN]:
            continue
        # Whole Word Masking means that if we mask all of the subwords
        # corresponding to an original word. When a word has been split into
        # subwords, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each subword independently, softmaxed
        # over the entire vocabulary.
        if whole_word_mask and len(cand_indexes) >= 1 and \
           not tokenizer.is_first_subword(vocab.idx_to_token[token]):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    random.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
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
                    masked_token = random.randint(0, len(vocab) - 1)

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
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

    # create output dir
    output_dir = os.path.expanduser(args.output_dir)
    nlp.utils.mkdir(output_dir)

    # vocabulary and tokenizer
    if args.sentencepiece:
        logging.info('loading vocab file from sentence piece model: %s', args.sentencepiece)
        if args.dataset_name:
            warnings.warn('Both --dataset_name and --sentencepiece are provided. '
                          'The vocabulary will be loaded based on --sentencepiece.')
        vocab = nlp.vocab.BERTVocab.from_sentencepiece(args.sentencepiece)
        tokenizer = nlp.data.BERTSPTokenizer(args.sentencepiece, vocab, num_best=args.sp_nbest,
                                             alpha=args.sp_alpha, lower=not args.cased)
    else:
        logging.info('loading vocab file from pre-defined dataset: %s', args.dataset_name)
        vocab = nlp.data.utils._load_pretrained_vocab(args.dataset_name, root=output_dir,
                                                      cls=nlp.vocab.BERTVocab)
        tokenizer = BERTTokenizer(vocab=vocab, lower='uncased' in args.dataset_name)

    # count the number of input files
    input_files = []
    for input_pattern in args.input_file.split(','):
        input_files.extend(glob.glob(os.path.expanduser(input_pattern)))
    for input_file in input_files:
        logging.info('\t%s', input_file)
    num_inputs = len(input_files)
    num_outputs = min(args.num_outputs, len(input_files))
    logging.info('*** Reading from %d input files ***', num_inputs)

    # calculate the number of splits
    file_splits = []
    split_size = (num_inputs + num_outputs - 1) // num_outputs
    for i in range(num_outputs):
        split_start = i * split_size
        split_end = min(num_inputs, (i + 1) * split_size)
        file_splits.append(input_files[split_start:split_end])

    # prepare workload
    count = 0
    process_args = []

    for i, file_split in enumerate(file_splits):
        output_file = os.path.join(output_dir, 'part-{}.npz'.format(str(i).zfill(3)))
        count += len(file_split)
        process_args.append((file_split, tokenizer, args.max_seq_length, args.short_seq_prob,
                             args.masked_lm_prob, args.max_predictions_per_seq,
                             args.whole_word_mask,
                             vocab, args.dupe_factor, 1, None, output_file))

    # sanity check
    assert count == len(input_files)

    # dispatch to workers
    nworker = args.num_workers
    if nworker > 1:
        pool = Pool(nworker)
        pool.map(create_training_instances, process_args)
    else:
        for process_arg in process_args:
            create_training_instances(process_arg)

    time_end = time.time()
    logging.info('Time cost=%.1f', time_end - time_start)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-training data generator for BERT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_file',
        type=str,
        required=True,
        help='Input files, separated by comma. For example, "~/data/*.txt"')

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory.')

    parser.add_argument(
        '--dataset_name',
        type=str,
        default=None,
        choices=['book_corpus_wiki_en_uncased', 'book_corpus_wiki_en_cased',
                 'wiki_multilingual_uncased', 'wiki_multilingual_cased', 'wiki_cn_cased'],
        help='The dataset name for the vocab file BERT model was trained on. For example, '
             '"book_corpus_wiki_en_uncased"')

    parser.add_argument(
        '--sentencepiece',
        type=str,
        default=None,
        help='Path to the sentencepiece .model file for both tokenization and vocab.')

    parser.add_argument(
        '--cased',
        action='store_true',
        help='Effective only if --sentencepiece is set')

    parser.add_argument('--sp_nbest', type=int, default=0,
                        help='Number of best candidates for sampling subwords with sentencepiece. ')

    parser.add_argument('--sp_alpha', type=float, default=1.0,
                        help='Inverse temperature for probability rescaling for sentencepiece '
                             'unigram sampling')

    parser.add_argument(
        '--whole_word_mask',
        action='store_true',
        help='Whether to use whole word masking rather than per-subword masking.')

    parser.add_argument(
        '--max_seq_length', type=int, default=512, help='Maximum sequence length.')

    parser.add_argument(
        '--max_predictions_per_seq',
        type=int,
        default=80,
        help='Maximum number of masked LM predictions per sequence. ')

    parser.add_argument(
        '--random_seed',
        type=int,
        default=12345,
        help='Random seed for data generation.')

    parser.add_argument(
        '--dupe_factor',
        type=int,
        default=1,
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
        'maximum length. ')

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print debug information')

    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='Number of workers for parallel processing, where each generates an output file.')

    parser.add_argument(
        '--num_outputs',
        type=int,
        default=1,
        help='Number of desired output files, where each is processed independently by a worker.')

    args = parser.parse_args()
    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logging.info(args)
    main()
