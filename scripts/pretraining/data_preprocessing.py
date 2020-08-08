"""
Prepare the feature for openwebtext dataset
"""
import os
import time
import math
import random
import argparse
import multiprocessing

import numpy as np

from pretraining_utils import get_all_features
from gluonnlp.models import get_backbone


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True,
                        help="path to extraed openwebtext dataset")
    parser.add_argument("-o", "--output", default="preprocessed_owt",
                        help="directory for preprocessed features")
    parser.add_argument("--num_process", type=int, default=8,
                        help="number of processes for multiprocessing")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="the maximum length of the pretraining sequence")
    parser.add_argument("--num_out_files", type=int, default=1000,
                        help="Number of desired output files, where each is processed"
                             " independently by a worker.")
    parser.add_argument('--model_name', type=str, default='google_electra_small',
                        help='Name of the pretrained model.')
    parser.add_argument("--shuffle", action="store_true",
                        help="Wether to shuffle the data order")
    parser.add_argument("--do_lower_case", dest='do_lower_case',
                        action="store_true", help="Lower case input text.")
    parser.add_argument("--no_lower_case", dest='do_lower_case',
                        action='store_false', help="Don't lower case input text.")
    parser.add_argument("--short_seq_prob", type=float, default=0.05,
                        help="The probability of sampling sequences shorter than"
                             " the max_seq_length.")
    parser.set_defaults(do_lower_case=True)
    return parser


def main(args):
    num_process = min(multiprocessing.cpu_count(), args.num_process)
    _, cfg, tokenizer, _, _ = \
        get_backbone(args.model_name, load_backbone=False)

    fnames = sorted(os.listdir(args.input))
    fnames = [os.path.join(args.input, fname) for fname in fnames]
    if args.shuffle:
        random.shuffle(fnames)
    num_files = len(fnames)
    num_out_files = min(args.num_out_files, num_files)
    splited_files = np.array_split(fnames, num_out_files)
    output_files = [os.path.join(
        args.output, "owt-pretrain-record-{}.npz".format(str(i).zfill(4))) for i in range(num_out_files)]
    print("All preprocessed features will be saved in {} npz files".format(num_out_files))
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    num_process = min(num_process, num_out_files)
    print('Start preprocessing {} text files with {} cores'.format(
        num_files, num_process))
    process_args = [
        (splited_files[i],
         output_files[i],
         tokenizer,
         args.max_seq_length,
         args.short_seq_prob) for i in range(
            num_out_files)]
    start_time = time.time()
    with multiprocessing.Pool(num_process) as pool:
        iter = pool.imap(get_all_features, process_args)
        fea_written = 0
        f_read = 0
        for i, np_features in enumerate(iter):
            elapsed = time.time() - start_time
            fea_written += len(np_features[0])
            f_read += len(splited_files[i])
            print("Processed {:} files, Elapsed: {:.2f}s, ETA: {:.2f}s, ".format(
                fea_written, elapsed, (num_files - f_read) / (f_read / elapsed)))
    print("Done processing within {:.2f} seconds".format(elapsed))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
