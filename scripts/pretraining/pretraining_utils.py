"""Utilities for pre-training."""
import io
import os
import re
import random
import logging
import collections

import numpy as np
from mxnet.gluon import HybridBlock
from mxnet.gluon.data import ArrayDataset

import gluonnlp.data.batchify as bf
from gluonnlp.utils.misc import glob
from gluonnlp.data.loading import NumpyDataset, DatasetLoader
from gluonnlp.data.sampler import SplitSampler, FixedBucketSampler
from gluonnlp.op import select_vectors_by_position, update_vectors_by_position

PretrainFeature = collections.namedtuple(
    'PretrainFeature',
    ['input_id',
     'segment_id',
     'valid_length'])


def tokenize_lines_to_ids(lines, tokenizer):
    """
    Worker function to tokenize lines based on the tokenizer, and perform vocabulary lookup.

    Parameters
    ----------
    lines
        Lines to be tokenized of the whole file
    tokenizer
        The trained tokenizer

    Returns
    -------
    results
        A list storing the valid tokenized lines
    """
    results = []
    # tag line delimiters or doc delimiters
    for line in lines:
        if not line:
            break
        line = line.strip()
        # Single empty lines are used as line delimiters
        # Double empty lines are used as document delimiters
        if not line:
            results.append([])
        else:
            token_ids = tokenizer.encode(line, int)
            if token_ids:
                results.append(token_ids)
    return results


def get_all_features(x):
    """
    Get the feature data in numpy form.

    Parameters
    ----------
    x
        List/tuple that contains:

        - file_list
            A list of text files
        - output_file
             The path to a output file that store the np_features
        - tokenizer
            The trained tokenizer
        - max_seq_length
            Maximum sequence length of the training features
        - short_seq_prob
             The probability of sampling sequences shorter than the max_seq_length.

    Returns
    -------
    np_features
        A tuple of (input_ids, segment_ids, valid_lengths),
        in which each item is a list of numpy arrays.
    """
    file_list, output_file, tokenizer, max_seq_length, short_seq_prob = x
    all_features = []
    for text_file in file_list:
        features = process_a_text(text_file, tokenizer, max_seq_length, short_seq_prob)
        all_features.extend(features)
    np_features = convert_to_npz(all_features, output_file)
    return np_features


def process_a_text(text_file, tokenizer, max_seq_length, short_seq_prob=0.05):
    """
    Create features from a single raw text file, in which one line is treated
    as a sentence, and double blank lines represent document separators.

    In this process, mxnet-unrelated features are generated, to easily convert
     to features of a particular deep learning framework in subsequent steps

    Parameters
    ----------
    text_file
        The path to a single text file
    tokenizer
        The trained tokenizer
    max_seq_length
        Maximum sequence length of the training features
    short_seq_prob
        The probability of sampling sequences shorter than the max_seq_length.

    Returns
    -------
    features
        A list of processed features from a single text file
    """
    vocab = tokenizer.vocab
    features = []
    # TODO(zheyuye), support whole word masking
    with io.open(text_file, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        tokenized_lines = tokenize_lines_to_ids(lines, tokenizer)
        target_seq_length = max_seq_length
        current_sentences = []
        current_length = 0
        for tokenized_line in tokenized_lines:
            current_sentences.append(tokenized_line)
            current_length += len(tokenized_line)
            # Create feature when meets the empty line or reaches the target length
            if (not tokenized_line and current_length != 0) or (
                    current_length >= target_seq_length):
                first_segment, second_segment = \
                    sentenceize(current_sentences, max_seq_length, target_seq_length)

                input_id = [vocab.cls_id] + first_segment + [vocab.sep_id]
                segment_id = [0] * len(input_id)

                if second_segment:
                    input_id += second_segment + [vocab.sep_id]
                    segment_id += [1] * (len(second_segment) + 1)

                # Padding with zeros for parallel storage
                valid_length = len(input_id)
                input_id += [0] * (max_seq_length - len(input_id))
                segment_id += [0] * (max_seq_length - len(segment_id))

                feature = PretrainFeature(input_id=input_id,
                                          segment_id=segment_id,
                                          valid_length=valid_length)
                features.append(feature)

                current_sentences = []
                current_length = 0
                # small chance for random-length instead of max_length-length feature
                if random.random() < short_seq_prob:
                    target_seq_length = random.randint(5, max_seq_length)
                else:
                    target_seq_length = max_seq_length

    return features


def convert_to_npz(all_features, output_file=None):
    """
    Convert features to numpy array and store if output_file provided

    Parameters
    ----------
    all_features
        A list of processed features.
    output_file
        The path to a output file that store the np_features.
    Returns
    -------
    input_ids
        A tuple of features
    segment_ids
        The segment ids
    valid_lengths
        The valid lengths
    """
    input_ids = []
    segment_ids = []
    valid_lengths = []
    for fea_index, feature in enumerate(all_features):
        input_ids.append(np.ascontiguousarray(feature.input_id, dtype='int32'))
        segment_ids.append(np.ascontiguousarray(feature.segment_id, dtype='int32'))
        valid_lengths.append(feature.valid_length)
        if fea_index < 1:
            logging.debug('*** Example Feature ***')
            logging.debug('Generated {}'.format(feature))

    if output_file:
        # The length numpy array are fixed to max_seq_length with zero padding
        npz_outputs = collections.OrderedDict()
        npz_outputs['input_ids'] = np.array(input_ids, dtype='int32')
        npz_outputs['segment_ids'] = np.array(segment_ids, dtype='int32')
        npz_outputs['valid_lengths'] = np.array(valid_lengths, dtype='int32')
        np.savez_compressed(output_file, **npz_outputs)
        logging.info("Saved {} features in {} ".format(len(all_features), output_file))
    return input_ids, segment_ids, valid_lengths


def sentenceize(current_sentences, max_seq_length, target_seq_length):
    """
    Generate a pair of sentences based on a segmentation strategy
    cloned from official electra model.

    Parameters
    ----------
    current_sentences
    max_seq_length
        Maximum sequence length of the training features
    target_seq_length
        Target sequence length of the training features
    Returns
    -------
    first_segment
        The first sentence of the pretraining sequence
    second_segment
        The second sentence of the pretraining sequence.
        Could be None for diversity of training instances.
    """
    # 10% chance to only produce one segment
    if random.random() < 0.1:
        first_segment_target_length = 100000
    else:
        # The reserved space for [CLS] and [SEP] tokens
        first_segment_target_length = (target_seq_length - 3) // 2
    first_segment = []
    second_segment = []
    for sentence in current_sentences:
        if sentence:
            # the sentence goes to the first segment if (1) the first segment is
            # empty, (2) the sentence doesn't put the first segment over length or
            # (3) 50% of the time when it does put the first segment over length
            if (len(first_segment) == 0 or
                    len(first_segment) + len(sentence) < first_segment_target_length or
                    (len(second_segment) == 0 and
                     len(first_segment) < first_segment_target_length and
                     random.random() < 0.5)):
                first_segment += sentence
            else:
                second_segment += sentence

    # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
    first_segment = first_segment[:max_seq_length - 2]
    second_segment = second_segment[:max(0, max_seq_length -
                                         len(first_segment) - 3)]

    return first_segment, second_segment


def prepare_pretrain_npz_dataset(filename, allow_pickle=False):
    """Create dataset based on the numpy npz file"""
    if isinstance(filename, (list, tuple)):
        assert len(filename) == 1, \
            'When .npy/.npz data file is loaded, len(filename) must be 1.' \
            ' Received len(filename)={}.'.format(len(filename))
        filename = filename[0]
    logging.debug('start to load file %s ...', filename)
    return NumpyDataset(filename, allow_pickle=allow_pickle)


def prepare_pretrain_text_dataset(
        filenames,
        tokenizer,
        max_seq_length,
        short_seq_prob,
        cached_file_path):
    """Create dataset based on the raw text files"""
    if not isinstance(filenames, (list, tuple)):
        filenames = [filenames]
    if cached_file_path:
        # generate a filename based on the input filename ensuring no crash.
        # filename example: urlsf_subset00-130_data.txt
        suffix = re.split(r'\.|/', filenames[0])[-2]
        output_file = os.path.join(cached_file_path, "{}-pretrain-record.npz".format(suffix))
    else:
        output_file = None
    np_features = get_all_features(
        (filenames, output_file, tokenizer, max_seq_length, short_seq_prob))

    return ArrayDataset(*np_features)


def prepare_pretrain_bucket_sampler(dataset, batch_size, shuffle=False, num_buckets=1):
    """Create data sampler based on the dataset"""
    if isinstance(dataset, NumpyDataset):
        lengths = dataset.get_field('valid_lengths')
    else:
        lengths = dataset.transform(lambda input_ids, segment_ids,
                                    valid_lengths: valid_lengths, lazy=False)
    sampler = FixedBucketSampler(lengths,
                                 batch_size=batch_size,
                                 num_buckets=num_buckets,
                                 ratio=0,
                                 shuffle=shuffle)
    logging.debug('Sampler created for a new dataset:\n {}'.format(sampler))
    return sampler


def get_pretrain_data_npz(data, batch_size, shuffle, num_buckets,
                          vocab, num_parts=1, part_idx=0,
                          num_dataset_workers=1, num_batch_workers=1,
                          circle_length=1, repeat=1,
                          dataset_cached=False,
                          num_max_dataset_cached=0):
    """Get a data iterator from pre-processed npz files.

    Parameters
    ----------
    data: str
        The path to the dataset directory
    batch_size : int
        The batch size per GPU.
    shuffle : bool
        Whether to shuffle the data.
    num_buckets : int
        The number of buckets for the FixedBucketSampler for training.
    vocab : Vocab
        The vocabulary.
    num_parts : int
        The number of partitions for the dataset.
    part_idx : int
        The index of the partition to read.
    num_dataset_workers : int
        The number of worker processes for dataset construction.
    num_batch_workers : int
        The number of worker processes for batch contruction.
    circle_length : int, default is 1
        The number of files to be read for a single worker at the same time.
        When circle_length is larger than 1, we merge circle_length files.
    repeat : int, default is 1
        The number of times that files are repeated.
    dataset_cached : bool, default is False
        Whether or not to cache last processed dataset.
        Each processed dataset can only be cached for once.
        When there is no new available processed dataset to be fetched,
        we pop a cached processed dataset.
    num_max_dataset_cached : int, default is 0
        Maximum number of cached datasets. It is valid only if dataset_cached is True
    """
    num_files = len(glob(data))
    logging.info('%d files are found.', num_files)
    assert num_files >= num_parts, \
        'The number of text files must be no less than the number of ' \
        'workers/partitions (%d). Only %d files at %s are found.' % (num_parts, num_files, data)
    split_sampler = SplitSampler(num_files, num_parts=num_parts,
                                 part_index=part_idx, repeat=repeat)
    dataset_fn = prepare_pretrain_npz_dataset
    sampler_fn = prepare_pretrain_bucket_sampler
    dataset_params = {'allow_pickle': True}
    sampler_params = {'batch_size': batch_size, 'shuffle': shuffle, 'num_buckets': num_buckets}
    batchify_fn = bf.Tuple(
        bf.Pad(val=vocab.pad_id),  # input_ids
        bf.Pad(val=0),  # segment_ids
        bf.Stack(),  # valid_lengths
    )
    dataloader = DatasetLoader(data,
                               file_sampler=split_sampler,
                               dataset_fn=dataset_fn,
                               batch_sampler_fn=sampler_fn,
                               dataset_params=dataset_params,
                               batch_sampler_params=sampler_params,
                               batchify_fn=batchify_fn,
                               num_dataset_workers=num_dataset_workers,
                               num_batch_workers=num_batch_workers,
                               pin_memory=False,
                               circle_length=circle_length)
    return dataloader


def get_pretrain_data_text(data, batch_size, shuffle, num_buckets, tokenizer, vocab,
                           max_seq_length, short_seq_prob=0.05, num_parts=1,
                           part_idx=0, num_dataset_workers=1, num_batch_workers=1,
                           circle_length=1, repeat=1, cached_file_path=None):
    """Get a data iterator from raw text documents.

    Parameters
    ----------
    batch_size : int
        The batch size per GPU.
    shuffle : bool
        Whether to shuffle the data.
    num_buckets : int
        The number of buckets for the FixedBucketSampler for training.
    vocab : Vocab
        The vocabulary.
    tokenizer : HuggingFaceWordPieceTokenizer or SentencepieceTokenizer
        The tokenizer.
    max_seq_length : int
        The hard limit of maximum sequence length of sentence pairs.
    short_seq_prob : float
        The probability of sampling sequences shorter than the max_seq_length.
    num_parts : int
        The number of partitions for the dataset.
    part_idx : int
        The index of the partition to read.
    num_dataset_workers : int
        The number of worker processes for dataset construction.
    num_batch_workers : int
        The number of worker processes for batch construction.
    circle_length : int, default is 1
        The number of files to be read for a single worker at the same time.
        When circle_length is larger than 1, we merge circle_length files.
    repeat : int, default is 1
        The number of times that files are repeated.
    cached_file_path: str, default is None
        Directory for saving preprocessed features
    """
    num_files = len(glob(data))
    logging.info('%d files are found.', num_files)
    assert num_files >= num_parts, \
        'The number of text files must be no less than the number of ' \
        'workers/partitions (%d). Only %d files at %s are found.' % (num_parts, num_files, data)
    split_sampler = SplitSampler(num_files, num_parts=num_parts,
                                 part_index=part_idx, repeat=repeat)
    dataset_fn = prepare_pretrain_text_dataset
    sampler_fn = prepare_pretrain_bucket_sampler
    dataset_params = {'tokenizer': tokenizer, 'max_seq_length': max_seq_length,
                      'short_seq_prob': short_seq_prob, 'cached_file_path': cached_file_path}
    sampler_params = {'batch_size': batch_size, 'shuffle': shuffle, 'num_buckets': num_buckets}
    batchify_fn = bf.Tuple(
        bf.Pad(val=vocab.pad_id),  # input_ids
        bf.Pad(val=0),  # segment_ids
        bf.Stack(),  # valid_lengths
    )

    dataloader = DatasetLoader(data,
                               file_sampler=split_sampler,
                               dataset_fn=dataset_fn,
                               batch_sampler_fn=sampler_fn,
                               dataset_params=dataset_params,
                               batch_sampler_params=sampler_params,
                               batchify_fn=batchify_fn,
                               num_dataset_workers=num_dataset_workers,
                               num_batch_workers=num_batch_workers,
                               pin_memory=False,
                               circle_length=circle_length)
    return dataloader


class ElectraMasker(HybridBlock):
    """ELECTRA pre-processes and applies masks pretrain data

    Parameters
    ----------
    tokenizer : gluonnlp.data.tokenizers
        Used to tokenize the pretrained text sequence.
    max_seq_length : int
        Maximum sequence length of preprocessed text for pretraining.
    mask_prob : float
        The probability of applying masks on the token in the sequence.
    proposal_distribution : float
        A predefined probability distribution for each position in the sequence.
    replace_prob : float
        The probability of replace current token with a generator-predicted token.
    """
    MaskedInput = collections.namedtuple('MaskedInput',
                                         ['input_ids',
                                          'masks',
                                          'unmasked_tokens',
                                          'masked_positions',
                                          'masked_weights'])

    def __init__(self, tokenizer, max_seq_length, mask_prob=0.15,
                 proposal_distribution=1.0, replace_prob=0.85):
        super().__init__()
        self._max_seq_length = max_seq_length
        self._mask_prob = mask_prob
        self._max_num_masked_position = int((self._mask_prob + 0.005) *
                                            self._max_seq_length)
        self._proposal_distribution = proposal_distribution
        self.vocab = tokenizer.vocab
        self._replace_prob = replace_prob

    def dynamic_masking(self, F, input_ids, valid_lengths):
        # TODO(zheyuye), two additional flag `disallow_from_mask` and `already_masked`
        # that control the masking status for each positions in the sequence.
        """
        Generate masking positions on-the-fly instead of during preprocessing
        Parameters
        ----------
        input_ids
            The batchified input_ids with shape (batch_size, max_seq_length)
        valid_lengths
            The batchified valid_lengths with shape (batch_size, )
        Returns
        ------
        masked_input_ids
            The masked input sequence with 15% tokens are masked with [MASK]
            shape (batch_size, max_seq_length)
        length_masks
            The masking matrix for the whole sequence that indicates the positions
            are greater than valid_length.

            shape (batch_size, max_seq_length)
        unmasked_tokens
            The original tokens that appear in the unmasked input sequence
            shape (batch_size, num_masked_positions)
        masked_positions
            The masking positions in mx.np.ndarray with shape (batch_size, num_masked_positions)
            shape (batch_size, num_masked_positions)
        masked_lm_weights
            The weight matrix containing 0 or 1 to mark the actual effect of masked positions
            shape (batch_size, num_masked_positions)
        """
        N = self._max_num_masked_position
        # Only valid token without special token are allowed to mask
        valid_candidates = F.np.ones_like(input_ids, dtype=np.bool)
        ignore_tokens = [self.vocab.cls_id, self.vocab.sep_id, self.vocab.pad_id]

        for ignore_token in ignore_tokens:
            # TODO(zheyuye), Update when operation += supported
            valid_candidates = valid_candidates * \
                F.np.not_equal(input_ids, ignore_token)
        valid_lengths = valid_lengths.astype(np.float32)
        valid_candidates = valid_candidates.astype(np.float32)
        num_masked_position = F.np.maximum(
            1, F.np.minimum(N, round(valid_lengths * self._mask_prob)))

        # Get the masking probability of each position
        sample_probs = self._proposal_distribution * valid_candidates
        sample_probs /= F.np.sum(sample_probs, axis=-1, keepdims=True)
        sample_probs = F.npx.stop_gradient(sample_probs)
        gumbels = F.np.random.gumbel(F.np.zeros_like(sample_probs))
        # Following the instruction of official repo to avoid deduplicate postions
        # with Top_k Sampling as https://github.com/google-research/electra/issues/41
        masked_positions = F.npx.topk(
            F.np.log(sample_probs) + gumbels, k=N,
            axis=-1, ret_typ='indices', dtype=np.int32)

        masked_weights = F.npx.sequence_mask(
            F.np.ones_like(masked_positions),
            sequence_length=num_masked_position,
            use_sequence_length=True, axis=1, value=0)
        masked_positions = masked_positions * masked_weights
        length_masks = F.npx.sequence_mask(
            F.np.ones_like(input_ids, dtype=np.float32),
            sequence_length=valid_lengths,
            use_sequence_length=True, axis=1, value=0)
        unmasked_tokens = select_vectors_by_position(
            F, input_ids, masked_positions) * masked_weights
        masked_weights = masked_weights.astype(np.float32)
        replaced_positions = (
            F.np.random.uniform(
                F.np.zeros_like(masked_positions),
                F.np.ones_like(masked_positions)) < self._replace_prob) * masked_positions
        # dealing with multiple zero values in replaced_positions which causes
        # the [CLS] being replaced
        filled = F.np.where(
            replaced_positions,
            self.vocab.mask_id,
            self.vocab.cls_id).astype(
            np.int32)
        # Masking token by replacing with [MASK]
        masked_input_ids = update_vectors_by_position(F, input_ids, filled, replaced_positions)

        # Note: It is likely have multiple zero values in masked_positions if number of masked of
        # positions not reached the maximum. However, this example hardly exists since valid_length
        # is almost always equal to max_seq_length
        masked_input = self.MaskedInput(input_ids=masked_input_ids,
                                        masks=length_masks,
                                        unmasked_tokens=unmasked_tokens,
                                        masked_positions=masked_positions,
                                        masked_weights=masked_weights)
        return masked_input
