# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=
"""Extract the vocabulary from a file and write it to disk."""

import argparse
import itertools
import json
import logging
import time

import gluonnlp as nlp


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Vocabulary extractor.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max-size', type=int, default=None)
    parser.add_argument('--min-freq', type=int, default=5)
    parser.add_argument('--max-word-length', type=int, default=50)
    parser.add_argument('files', type=str, nargs='+')
    parser.add_argument('--vocab-output', type=str, default='vocab.json')
    parser.add_argument('--counts-output', type=str, default='counts.json')
    args = parser.parse_args()
    return args


def get_vocab(args):
    """Compute the vocabulary."""
    counter = nlp.data.Counter()
    start = time.time()
    for filename in args.files:
        print('Starting processing of {} after {:.1f} seconds.'.format(
            filename,
            time.time() - start))
        with open(filename, 'r') as f:
            tokens = itertools.chain.from_iterable((l.split() for l in f))
            counter.update(tokens)

    if args.max_word_length:
        counter = {
            w: c
            for w, c in counter.items() if len(w) < args.max_word_length
        }

    total_time = time.time() - start
    print('Finished after {:.1f} seconds.'.format(total_time))
    num_words = sum(counter.values())
    print('Got {} words. Processed {:.1f} per second.'.format(
        num_words, num_words / total_time))

    start = time.time()
    print('Starting creation of vocabulary.')
    vocab = nlp.Vocab(counter, max_size=args.max_size, min_freq=args.min_freq,
                      unknown_token=None, padding_token=None, bos_token=None,
                      eos_token=None)
    with open(args.vocab_output, 'w') as f:
        f.write(vocab.to_json())
    print('Finished creation of vocabulary after {:.1f} seconds.'.format(
        time.time() - start))

    print('Writing word counts.')
    start = time.time()
    idx_to_counts = [counter[t] for t in vocab.idx_to_token]
    with open(args.counts_output, 'w') as f:
        json.dump(idx_to_counts, f)
    print('Finished writing word counts after {:.1f} seconds..'.format(
        time.time() - start))


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    args_ = parse_args()
    get_vocab(args_)
