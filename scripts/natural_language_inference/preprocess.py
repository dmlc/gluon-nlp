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
# pylint: disable=redefined-outer-name

"""
Tokenize the SNLI dataset.
"""

import argparse
import csv
import nltk

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        help='.txt file for the SNLI dataset')
    parser.add_argument('--output',
                        help='path for tokenized output file')
    args = parser.parse_args()
    return args

def read_tokens(tree_str):
    t = nltk.Tree.fromstring(tree_str)
    return t.leaves()

def main(args):
    """
    Read tokens from the provided parse tree in the SNLI dataset.
    Illegal examples are removed.
    """
    examples = []
    with open(args.input, 'r') as fin:
        reader = csv.DictReader(fin, delimiter='\t')
        for cols in reader:
            s1 = read_tokens(cols['sentence1_parse'])
            s2 = read_tokens(cols['sentence2_parse'])
            label = cols['gold_label']
            if label in ('neutral', 'contradiction', 'entailment'):
                examples.append((s1, s2, label))
    with open(args.output, 'w') as fout:
        for s1, s2, l in examples:
            fout.write('{}\t{}\t{}\n'.format(' '.join(s1), ' '.join(s2), l))


if __name__ == '__main__':
    args = parse_args()
    main(args)
