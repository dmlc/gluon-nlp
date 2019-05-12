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
"""Evaluation method using the official perl script."""

import os
import tempfile
import time
from functools import reduce

from scripts.parsing.common.data import DataLoader


def evaluate_official_script(parser, vocab, num_buckets_test, test_batch_size, test_file,
                             output_file, debug=False):
    """Evaluate parser on a data set.

    Parameters
    ----------
    parser : BiaffineParser
        biaffine parser
    vocab : ParserVocabulary
        vocabulary built from data set
    num_buckets_test : int
        size of buckets (cluster sentences into this number of clusters)
    test_batch_size : int
        batch size
    test_file : str
        gold test file
    output_file : str
        output result to this file
    debug : bool
        only evaluate first 1000 sentences for debugging

    Returns
    -------
    tuple
        UAS, LAS, speed
    """
    if output_file is None:
        output_file = tempfile.NamedTemporaryFile().name
    data_loader = DataLoader(test_file, num_buckets_test, vocab)
    record = data_loader.idx_sequence
    results = [None] * len(record)
    idx = 0
    seconds = time.time()
    for words, tags, arcs, rels in data_loader.get_batches(batch_size=test_batch_size,
                                                           shuffle=False):
        outputs = parser.forward(words, tags)
        for output in outputs:
            sent_idx = record[idx]
            results[sent_idx] = output
            idx += 1
    assert idx == len(results), 'parser swallowed some sentences'
    seconds = time.time() - seconds
    speed = len(record) / seconds

    arcs = reduce(lambda x, y: x + y, [list(result[0]) for result in results])
    rels = reduce(lambda x, y: x + y, [list(result[1]) for result in results])
    idx = 0
    with open(test_file) as f:
        if debug:
            f = f.readlines()[:1000]
        with open(output_file, 'w') as fo:
            for line in f:
                info = line.strip().split()
                if info:
                    arc_offset = 5
                    rel_offset = 6
                    if len(info) == 10:  # conll or conllx
                        arc_offset = 6
                        rel_offset = 7
                    # assert len(info) == 10, 'Illegal line: %s' % line
                    info[arc_offset] = str(arcs[idx])
                    info[rel_offset] = vocab.id2rel(rels[idx])
                    fo.write('\t'.join(info) + '\n')
                    idx += 1
                else:
                    fo.write('\n')

    os.system('perl %s -q -b -g %s -s %s -o tmp' % (
        os.path.join(os.path.dirname(os.path.realpath(__file__)), 'eval.pl'),
        test_file, output_file))
    os.system('tail -n 3 tmp > score_tmp')
    LAS, UAS = [float(line.strip().split()[-2]) for line in open('score_tmp').readlines()[:2]]
    # print('UAS %.2f, LAS %.2f' % (UAS, LAS))
    os.system('rm tmp score_tmp')
    os.remove(output_file)
    return UAS, LAS, speed
