"""Pretraining on code"""
import tqdm
import argparse
import time
import os
import pathlib
import random
import math
import multiprocessing

from smart_open import open
import gluonnlp as nlp
import numpy as np
import pyarrow as pa
import pyarrow.feather
import pyarrow.compute


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description=__doc__)

    # Model
    group = parser.add_argument_group('Model')
    group.add_argument('--model-name', type=str, default='google_en_cased_bert_base',
                       choices=nlp.models.bert.list_pretrained_bert(),
                       help='Name of the model configuration.')

    # Input
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--inputs', type=pathlib.Path, nargs='+', help='Input files.')
    group.add_argument('--input-reference', type=pathlib.Path,
                       help='Path to file containing input file paths on each line.')
    group = parser.add_argument_group('Output')
    parser.add_argument('--output', required=True, type=pathlib.Path)

    # data pre-processing
    group = parser.add_argument_group('Data pre-processing')
    group.add_argument('--dupe-factor', type=int, default=5,
                       help='Number of times to duplicate the input data (with different masks).')
    group.add_argument('--max-seq-length', type=int, default=512,
                       help='Maximum input sequence length.')
    group.add_argument('--short-seq-prob', type=float, default=0.1,
                       help='The probability of producing sequences shorter than max_seq_length.')
    group.add_argument('--masked-lm-prob', type=float, default=0.15, help='Probability for masks.')
    group.add_argument('--max-predictions-per-seq', type=int, default=80,
                       help='Maximum number of predictions per sequence.')
    # group.add_argument('--whole_word_mask', action=nlp.utils.misc.BooleanOptionalAction,
    #                    default=False,
    #                    help='Whether to use whole word masking rather than per-subword masking.')

    # Computation and communication
    group = parser.add_argument_group('Computation')
    group.add_argument('--processes', type=int, default=os.cpu_count(),
                       help='Number of processes in process pool.')
    group.add_argument('--seed', type=int, default=100, help='Random seed')

    # Misc
    group = parser.add_argument_group('Misc')
    group.add_argument('--logging-steps', type=int, default=10)

    args = parser.parse_args()

    return args


def set_seed(seed):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)


def create_masked_lm_predictions(*, args, tokens, cls_token_id, sep_token_id, mask_token_id,
                                 non_special_ids):
    """Creates the predictions for the masked LM objective."""
    cand_indexes = [i for i, tok in enumerate(tokens) if tok not in (cls_token_id, sep_token_id)]
    output_tokens = list(tokens)
    random.shuffle(cand_indexes)
    num_to_predict = min(args.max_predictions_per_seq,
                         max(1, int(round(len(tokens) * args.masked_lm_prob))))
    mlm_positions = []
    mlm_labels = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(mlm_positions) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)
        masked_token = None

        # 80% of the time, replace with [MASK]
        if random.random() < 0.8:
            masked_token = mask_token_id
        else:
            # 10% of the time, keep original
            if random.random() < 0.5:
                masked_token = tokens[index]
                # 10% of the time, replace with random word
            else:
                masked_token = random.choice(non_special_ids)

        output_tokens[index] = masked_token
        mlm_positions.append(index)
        mlm_labels.append(tokens[index])
    assert len(mlm_positions) <= num_to_predict
    assert len(mlm_positions) == len(mlm_labels)
    return output_tokens, mlm_positions, mlm_labels


def process_file(input_file):
    # Retrieve process-local state
    tokenizer = process_file.tokenizer
    schema = process_file.schema
    args = process_file.args
    vocab = process_file.vocab
    non_special_ids = process_file.non_special_ids

    # Process file
    pa_batches = []  # List of PyArrow RecordBatches
    buffers = [[] for _ in range(len(schema))]  # Arrays in same order as schema
    with open(input_file, 'r', encoding='utf-8') as f:
        document_lines = f.readlines()
    if len(document_lines) < 2:
        print(f'Skipped {input_file} as it contains less than 2 sentences.')
        return None

    # According to the original tensorflow implementation: We *sometimes*
    # (i.e., short_seq_prob == 0.1, 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and
    # fine-tuning.
    target_seq_length = args.max_seq_length - 2  # [CLS] ... [SEP]
    if random.random() < args.short_seq_prob:
        target_seq_length = random.randint(2, target_seq_length)

    toks = tokenizer.encode(document_lines, int)

    current_chunk_1, current_chunk_2 = [], []
    current_length_1, current_length_2 = 0, 0

    i = 0
    while i < len(toks):  # While there are samples in the file
        segment = toks[i]
        i += 1
        if current_length_1 < target_seq_length:
            current_chunk_1.append(segment)
            current_length_1 += len(segment)
        else:
            current_chunk_2.append(segment)
            current_length_2 += len(segment)

        # Reached target length or end of document
        if (current_chunk_1 and current_chunk_2 and current_length_1 >= target_seq_length
                and current_length_2 >= target_seq_length):
            toks1 = sum(current_chunk_1, [])[:target_seq_length]
            toks2 = sum(current_chunk_2, [])[:target_seq_length]
            current_chunk_1, current_chunk_2 = [], []
            current_length_1, current_length_2 = 0, 0

            toks1, mlmpositions1, mlmlabels1 = create_masked_lm_predictions(
                args=args, tokens=toks1, cls_token_id=vocab.cls_id, sep_token_id=vocab.sep_id,
                mask_token_id=vocab.mask_id, non_special_ids=non_special_ids)
            toks2, mlmpositions2, mlmlabels2 = create_masked_lm_predictions(
                args=args, tokens=toks2, cls_token_id=vocab.cls_id, sep_token_id=vocab.sep_id,
                mask_token_id=vocab.mask_id, non_special_ids=non_special_ids)

            # Arrays in same order as schema
            buffers[0].append(toks1)
            buffers[1].append(toks2)
            buffers[2].append(len(toks1))
            buffers[3].append(len(toks2))
            buffers[4].append(mlmpositions1)
            buffers[5].append(mlmpositions2)
            buffers[6].append(mlmlabels1)
            buffers[7].append(mlmlabels2)

            if len(buffers[0]) >= 1000:
                batch = pa.RecordBatch.from_arrays(buffers, schema=schema)
                pa_batches.append(batch)
                buffers = [[] for _ in range(len(schema))]

    if len(buffers[0]):
        batch = pa.RecordBatch.from_arrays(buffers, schema=schema)
        pa_batches.append(batch)
        buffers = [[] for _ in range(len(schema))]

    batch_tbl = pa.Table.from_batches(pa_batches, schema=schema)
    args.output.mkdir(parents=True, exist_ok=True)
    pa.feather.write_feather(
        batch_tbl, args.output / f'quickthought_{os.getpid()}_{process_file.process_idx}.feather')
    process_file.process_idx += 1


def get_input_files(args):
    if args.input_reference:
        assert not args.inputs
        with open(args.input_reference, 'r') as f:
            args.inputs = f.read().splitlines()

    print(f'Processing {len(args.inputs)} input files {args.dupe_factor} times.')

    # Duplicate input files based on dupe-factor
    args.inputs = sum((args.inputs for _ in range(args.dupe_factor)), [])

    # if args.num_nodes > 1:
    #     local_size = math.ceil(len(args.inputs) / args.num_nodes)
    #     args.inputs = args.inputs[local_size * args.rank:local_size * (args.rank + 1)]


def main():
    args = parse_args()
    set_seed(args.seed)
    get_input_files(args)

    def _initializer(function):
        """Initialize state of each process in multiprocessing pool.

        The process local state is stored as an attribute of the function
        object, which is specified in Pool(..., initargs=(function, )) and by
        convention refers to the function executed during map.

        """
        # TODO gluonnlp shouldn't provide a slow LegacyHuggingFaceTokenizer here...
        _, tokenizer, _, _ = nlp.models.bert.get_pretrained_bert(args.model_name,
                                                                 load_backbone=False,
                                                                 load_mlm=False)
        function.tokenizer = tokenizer
        function.args = args
        function.vocab = tokenizer.vocab
        function.non_special_ids = tokenizer.vocab[tokenizer.vocab.non_special_tokens]
        function.process_idx = 0

        tok_type = pa.uint16() if len(tokenizer.vocab) <= np.iinfo(np.uint16).max else pa.uint32()
        assert len(tokenizer.vocab) <= np.iinfo(np.uint32).max
        length_type = pa.uint16()
        assert args.max_seq_length * 2 <= np.iinfo(np.uint16).max

        # pa.large_list instead of pa.list_ to use 64bit offsets
        # See https://issues.apache.org/jira/browse/ARROW-9773
        schema = pa.schema({
            "quickthought1": pa.large_list(tok_type),
            "quickthought2": pa.large_list(tok_type),
            "validlength1": length_type,
            "validlength2": length_type,
            "mlmpositions1": pa.large_list(length_type),
            "mlmpositions2": pa.large_list(length_type),
            "mlmlabels1": pa.large_list(tok_type),
            "mlmlabels2": pa.large_list(tok_type),
        })
        function.schema = schema

    if args.processes:
        with multiprocessing.Pool(initializer=_initializer, initargs=(process_file, ),
                                  processes=args.processes) as pool:
            # pool.map(process_file, args.inputs) with tqdm progress bar
            with tqdm.tqdm(total=len(args.inputs)) as pbar:
                for i, _ in enumerate(pool.imap_unordered(process_file, args.inputs)):
                    pbar.update()
    else:
        _initializer(process_file)
        for _ in tqdm.tqdm(map(process_file, args.inputs), total=len(args.inputs)):
            pass


if __name__ == '__main__':
    main()
