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
"""Evaluation module for parsing results."""

import time
from functools import reduce

import numpy as np

from scripts.parsing.common.data import DataLoader


def evaluate_official_script(parser, vocab, num_buckets_test, test_batch_size,
                             test_file, output_file, debug=False):
    """Evaluate parser on a data set

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
    data_loader = DataLoader(test_file, num_buckets_test, vocab)
    record = data_loader.idx_sequence
    results = [None] * len(record)
    idx = 0
    seconds = time.time()
    uc, lc, total = 0, 0, 0
    for words, tags, arcs, rels in data_loader.get_batches(batch_size=test_batch_size,
                                                           shuffle=False):
        outputs = parser.forward(words, tags)
        for output, gold_arc, gold_rel in zip(
                outputs, arcs.transpose([1, 0]), rels.transpose([1, 0])):
            pred_arc = output[0]
            pred_rel = output[1]
            length = pred_arc.shape[0]
            gold_arc = gold_arc[1:length + 1]
            gold_rel = gold_rel[1:length + 1]

            arc_mask = np.equal(pred_arc, gold_arc)
            uc += np.sum(arc_mask)
            total += length

            lc += np.sum(np.equal(pred_rel, gold_rel) * arc_mask)
            sent_idx = record[idx]
            results[sent_idx] = output
            idx += 1
    speed = len(record) / seconds
    UAS = uc / total * 100
    LAS = lc / total * 100
    if output_file:
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
    return UAS, LAS, speed


def prf(correct, pred_sum, gold_sum):
    """
    Calculate precision, recall and f1 score
    Parameters
    ----------
    correct : int
                number of correct predictions
    pred_sum : int
                number of predictions
    gold_sum : int
                number of gold answers
    Returns
    -------
    tuple
                (p, r, f)
    """
    if pred_sum:
        p = correct / pred_sum
    else:
        p = 0
    if gold_sum:
        r = correct / gold_sum
    else:
        r = 0
    if p + r:
        f = 2 * p * r / (p + r)
    else:
        f = 0
    return p, r, f
