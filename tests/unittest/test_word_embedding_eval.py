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

from __future__ import print_function

import collections
import json

import mxnet as mx

import gluonnlp as nlp


def test_similarity():
    token_embedding = nlp.embedding.create("glove", source="glove.6B.300d.txt")

    # Construct vocabulary
    counter = collections.Counter(token_embedding._token_to_idx.keys())
    vocab = nlp.vocab.Vocab(counter)
    vocab.set_embedding(token_embedding)

    for name, cls, score in [("WordSim353", nlp.data.WordSim353, 0.65),
                             ("RW", nlp.data.RareWords, 0.38)]:
        data = cls()
        evaluator = nlp.evaluation.WordEmbeddingSimilarityEvaluator(
            data, vocab)
        r = evaluator(token_embedding)
        print(name, r, score)


if __name__ == '__main__':
    import nose
    nose.runmodule()
