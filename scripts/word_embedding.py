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
# pylint: disable=eval-used, redefined-outer-name

"""Word Embeddings
===============

This example shows how to load and perform intrinsic evaluatio of word
embeddings using a variety of datasets all part of the Gluon NLP Toolkit.

"""


import argparse
import time

import mxnet as mx
import numpy as np
from scipy import stats

import gluonnlp as nlp

parser = argparse.ArgumentParser(
    description='Word embedding training with Gluon.')

# Embeddings arguments
possible_embedding_names = [
    '{k}:{v}'.format(k=k, v=v) for k in nlp.embedding.list_sources().keys()
    for v in nlp.embedding.list_sources()[k]
]
parser.add_argument(
    '--embedding-name',
    type=str,
    default='glove:glove.6B.300d.txt',
    help=('Name of embedding type to load. '
          'Valid entries: {}'.format(', '.join(possible_embedding_names))))
# Evaluation arguments
parser.add_argument(
    '--eval-similarity',
    type=str,
    default='*',
    nargs='+',
    help='Word similarity datasets to use for intrinsic evaluation. '
    'Defaults to all (wildcard "*")')
parser.add_argument(
    '--disable-eval-nearest-neighbors',
    action='store_false',
    help='Print nearest neighbors of 5 random words in SimVerb3500')

args = parser.parse_args()

print(args)

###############################################################################
# Load data
###############################################################################
embedding_name, source = args.embedding_name.split(':')

print('Loading embedding ', args.embedding_name)
token_embedding = nlp.embedding.create(embedding_name, source=source)

# Construct a vocabulary encompassing all tokens in the TokenEmbedding
counter = nlp.data.utils.Counter(token_embedding.idx_to_token)
vocab = nlp.vocab.Vocab(counter)


###############################################################################
# Evaluators
###############################################################################
class _WordEmbeddingEvaluator(object):
    """Helper class to evaluate word embeddings."""

    def __init__(self, dataset, vocabulary):
        self.dataset = dataset
        self.vocabulary = vocabulary


class WordEmbeddingSimilarityEvaluator(_WordEmbeddingEvaluator):
    """Helper class to evaluate word embeddings based on similarity task."""

    # Words and ground truth scores
    _w1s = None
    _w2s = None
    _scores = None
    _context = None

    def __init__(self,
                 dataset,
                 vocabulary,
                 correlation_coefficient='spearmanr'):
        super(WordEmbeddingSimilarityEvaluator, self).__init__(
            dataset=dataset, vocabulary=vocabulary)
        assert correlation_coefficient in ['spearmanr', 'pearsonr']
        self.correlation_coefficient = correlation_coefficient

        # Construct nd arrays from dataset
        w1s = []
        w2s = []
        scores = []
        for word1, word2, score in self.dataset:
            if (word1 in self.vocabulary and word2 in self.vocabulary):
                w1s.append(word1)
                w2s.append(word2)
                scores.append(score)

        print(('Using {num_use} of {num_total} word pairs '
               'from {ds} for evaluation.').format(
                   num_use=len(w1s),
                   num_total=len(self.dataset),
                   ds=self.dataset.__class__.__name__))

        self._w1s = w1s
        self._w2s = w2s
        self._scores = np.array(scores)

    def __len__(self):
        return len(self._w1s)

    def __call__(self, token_embedding):
        if not len(self):
            return 0

        w1s_embedding = mx.nd.L2Normalization(token_embedding[self._w1s])
        w2s_embedding = mx.nd.L2Normalization(token_embedding[self._w2s])

        batch_size, embedding_size = w1s_embedding.shape

        cosine_similarity = mx.nd.batch_dot(
            w1s_embedding.reshape((batch_size, 1, embedding_size)),
            w2s_embedding.reshape((batch_size, embedding_size, 1)))
        cosine_similarity_np = cosine_similarity.asnumpy().flatten()

        if self.correlation_coefficient == 'spearmanr':
            r = stats.spearmanr(cosine_similarity_np, self._scores).correlation
        elif self.correlation_coefficient == 'pearsonr':
            r = stats.pearsonr(cosine_similarity_np, self._scores).correlation
        else:
            raise ValueError('Invalid correlation_coefficient: {}'.format(
                self.correlation_coefficient))

        return r


# @attr.s()
# class WordEmbeddingNearestNeighborEvaluator(WordEmbeddingEvaluator):
#     num_base_words = attr.ib(default=5)
#     num_nearest_neighbors = attr.ib(default=5)

#     # Words and ground truth scores
#     _words = None
#     _indices = None

#     def __attrs_post_init__(self):
#         # Construct nd arrays from dataset
#         self._words = []
#         for word1, word2, score in self.dataset:
#             for word in [word1, word2]:
#                 if word in self.token_embedding.token_to_idx:
#                     self._words.append(words)
#         random.shuffle(self._words)
#         self._indices = mx.nd.array(
#             [self.token_embedding.token_to_idx[w] for w in self._words],
#             ctx=mx.cpu())

#         print('Using ' + str(self._words[:self.num_base_words]) +
#               ' as seeds for NN evaluation.')

#     def __len__(self):
#         return self._indices.shape[0]

#     def __call__(self, embedding):
#         words = self._indices.as_in_context(embedding.weight.list_ctx()[0])
#         embedding = mx.nd.L2Normalization(embedding(words))

#         similarity = mx.nd.dot(embedding, embedding.T).argsort(
#             axis=1, is_ascend=0)

#         eval_strs = []
#         for i in range(self.num_nearest_neighbors):
#             eval_strs.append(' '.join(
#                 words[int(idx.asscalar())]
#                 for idx in similarity[i][:self.num_nearest_neighbors]))
#         return '\n'.join(eval_strs)

# @attr.s()
# class WordEmbeddingAnalogyEvaluator(WordEmbeddingEvaluator):
#     analogy = attr.ib(
#         default='3CosMul',
#         validator=attr.validators.in_(['3CosMul', '3CosAdd', 'PairDirection']))

#     # Words and ground truth scores
#     _w1s = None
#     _w2s = None
#     _scores = None

#     def __attrs_post_init__(self):
#         # Construct nd arrays from dataset
#         w1s = []
#         w2s = []
#         scores = []
#         for word1, word2, score in self.dataset:
#             if (word1 in self.token_embedding.token_to_idx
#                     and word2 in self.token_embedding.token_to_idx):
#                 w1s.append(self.token_embedding.token_to_idx[word1])
#                 w2s.append(self.token_embedding.token_to_idx[word2])
#                 scores.append(score)

#         print(('Using {num_use} of {num_total} word pairs '
#                'from {ds} for evaluation.').format(
#                    num_use=len(w1s),
#                    num_total=len(self.dataset),
#                    ds=self.dataset.__class__.__name__))

#         self._w1s = mx.nd.array(w1s, ctx=mx.cpu())
#         self._w2s = mx.nd.array(w2s, ctx=mx.cpu())
#         self._scores_np = np.array(scores)

#     def __len__(self):
#         return self._w1s.shape[0]

#     def __call__(self, embedding):
#         w1s = self._w1s.as_in_context(embedding.weight.list_ctx()[0])
#         w2s = self._w2s.as_in_context(embedding.weight.list_ctx()[0])

#         w1s_embedding = mx.nd.L2Normalization(embedding(w1s))
#         w2s_embedding = mx.nd.L2Normalization(embedding(w2s))

#         batch_size, embedding_size = w1s_embedding.shape

#         cosine_similarity = mx.nd.batch_dot(
#             w1s_embedding.reshape((batch_size, 1, embedding_size)),
#             w2s_embedding.reshape((batch_size, embedding_size, 1)))
#         cosine_similarity_np = cosine_similarity.asnumpy().flatten()
#         pearson_r = np.corrcoef(cosine_similarity_np, self._scores_np)[0, 1]
#         return pearson_r

###############################################################################
# Evaluation code
###############################################################################

evaluators = []

# Word similarity based evaluation
if args.eval_similarity:
    similarity_datasets = \
        nlp.data.word_embedding_evaluation.word_similarity_datasets
    if args.eval_similarity == '*':
        args.eval_similarity = similarity_datasets

    for ds in args.eval_similarity:
        if ds not in similarity_datasets:
            print(('{ds} is not a supported dataset. '
                   'Only {supported} are supported').format(
                       ds=ds, supported=', '.join(similarity_datasets)))
            continue

        ds_class = eval('nlp.data.{ds}'.format(ds=ds))
        evaluator = WordEmbeddingSimilarityEvaluator(
            dataset=ds_class(), vocabulary=vocab)
        if len(evaluator):
            evaluators.append(evaluator)

# # Nearest neighbor printing based evaluation
# TODO
# if not args.disable_eval_nearest_neighbors:
#     nn_evaluator = nlp.evaluation.WordEmbeddingNearestNeighborEvaluator(
#         dataset=nlp.data.SimVerb3500(), token_to_idx=sgdataset._token_to_idx)


def evaluate():
    eval_dict = {}
    for evaluator in evaluators:
        score = evaluator(token_embedding)
        eval_dict[evaluator.dataset.__class__.__name__] = score
        print(evaluator.dataset.__class__.__name__, score)

    return eval_dict


if __name__ == '__main__':
    start_pipeline_time = time.time()
    evaluate()
