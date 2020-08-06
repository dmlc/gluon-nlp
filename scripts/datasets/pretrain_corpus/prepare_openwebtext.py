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
"""Prepare the OpenWebText Dataset Corpus for pre-training. """

import os
import re
import time
import random
import tarfile
import argparse
import functools
import multiprocessing
from gluonnlp.registry import DATA_PARSER_REGISTRY, DATA_MAIN_REGISTRY

_CITATIONS = r"""
@misc{Gokaslan2019OpenWeb,
    title={OpenWebText Corpus},
    author={Aaron Gokaslan and Vanya Cohen},
    howpublished{\url{http://Skylion007.github.io/OpenWebTextCorpus}},
    year={2019}
}
"""


@DATA_PARSER_REGISTRY.register('prepare_openwebtext')
def get_parser():
    parser = argparse.ArgumentParser(description='Prepare the OpenWebText corpus for pretraining')
    parser.add_argument("-i", "--input", required=True,
                        help="path to openwebtext dataset")
    parser.add_argument("-o", "--output", default="openwebtext",
                        help="directory for extracted files")
    parser.add_argument("--num_process", type=int, default=8,
                        help="number of processes for multiprocessing")
    parser.add_argument("--shuffle", action="store_true",
                        help="Wether to shuffle the data order")
    return parser


def extract_files(full_name, output_dir, shuffle=False):
    """
    Extract the file and concatenate all the TXT files it archives
    """
    if not full_name.endswith(".xz"):
        return
    file_prefix = re.split(r'\.|/', full_name)[-2]
    file_prefix = file_prefix.replace('urlsf_subset', 'openwebtext-prepared-')
    with open("{}.txt".format(os.path.join(output_dir, file_prefix)), "w") as fp:
        with tarfile.open(full_name) as t:
            txt_names = t.getnames()
            if shuffle:
                random.shuffle(txt_names)
            for txt_name in txt_names:
                f = t.extractfile(txt_name)
                for line in f.readlines():
                    # skip empty line
                    line = line.strip()
                    if line:
                        fp.write(line.decode() + '\n')
                # Two extra line break to mark the document separation
                fp.write('\n')


@DATA_MAIN_REGISTRY.register('prepare_openwebtext')
def main(args):
    num_process = min(multiprocessing.cpu_count(), args.num_process)
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
    fnames = sorted(os.listdir(args.input))
    fnames = [os.path.join(args.input, fname) for fname in fnames]
    if args.shuffle:
        random.shuffle(fnames)
    print('Start extracting {} files with {} cores'.format(len(fnames), num_process))
    start_time = time.time()
    with multiprocessing.Pool(num_process) as pool:
        iter = pool.imap(
            functools.partial(
                extract_files,
                output_dir=args.output,
                shuffle=args.shuffle),
            fnames)
        for f_index, _ in enumerate(iter):
            if f_index > 0 and f_index % 250 == 0:
                elapsed = time.time() - start_time
                print("Extracted {:}, Elapsed: {:}s, ETA: {:}s, ".format(
                    f_index, int(elapsed), int((len(fnames) - f_index) / (f_index / elapsed))))

    print("Done!")


def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli_main()
