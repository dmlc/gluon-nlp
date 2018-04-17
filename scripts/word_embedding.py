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


###############################################################################
# Metrics
###############################################################################
@mx.metric.register
@mx.metric.alias('spearmanr')
class SpearmanRankCorrelation(mx.metric.EvalMetric):
    """Computes Spearman rank correlation.

    The Spearman correlation coefficient is defined as the Pearson correlation
    coefficient between the ranked variables.

    .. math::
        \\frac{cov(\\operatorname{rg}_y, \\operatorname{rg}_\\hat{y})}
        {\\sigma{\\operatorname{rg}_y}\\sigma{\\operatorname{rg}_\\hat{y}}}

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.

    Examples
    --------
    >>> predicts = [mx.nd.array([[0.3, 0.7], [0, 1.], [0.4, 0.6]])]
    >>> labels   = [mx.nd.array([[1, 0], [0, 1], [0, 1]])]
    >>> pr = SpearmanRankCorrelation()
    >>> pr.update(labels, predicts)
    >>> print pr.get()
    ('spearmanr', 0.42163704544016178)

    """

    def __init__(self, name='spearmanr', output_names=None, label_names=None):
        super(SpearmanRankCorrelation, self).__init__(
            name, output_names=output_names, label_names=label_names)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.
        preds : list of `NDArray`
            Predicted values.
        """
        labels, preds = mx.metric.check_label_shapes(labels, preds, True)

        for label, pred in zip(labels, preds):
            mx.metric.check_label_shapes(label, pred, False, True)
            label = label.asnumpy()
            pred = pred.asnumpy()
            self.sum_metric += stats.spearmanr(pred.ravel(),
                                               label.ravel()).correlation
            self.num_inst += 1


###############################################################################
# Similarity functions
###############################################################################
def cosine_similarity(w1s_embedding, w2s_embedding):
    w1s_embedding = mx.nd.L2Normalization(w1s_embedding)
    w2s_embedding = mx.nd.L2Normalization(w2s_embedding)

    batch_size, embedding_size = w1s_embedding.shape

    similarity = mx.nd.batch_dot(
        w1s_embedding.reshape((batch_size, 1, embedding_size)),
        w2s_embedding.reshape((batch_size, embedding_size, 1)))
    return similarity.reshape((-1, ))


###############################################################################
# Evaluators
###############################################################################
class WordEmbeddingSimilarityEvaluator(object):
    """Helper class to evaluate word embeddings based on similarity task.

    The Evaluator must be initialized, giving the option to adapt the
    parameters listed below. An Evaluator object can be called with the
    signature defined at Call Signature.

    Parameters
    ----------
    binary_score_metric : mx.metric.EvalMetric
        Metric for computing the overall score given the list of predicted
        similarities and ground truth similarities. Defaults to
        SpearmanRankCorrelation.
    similarity_function : function
        Given to mx.nd.NDArray's of shape (dataset_size, embedding_size),
        compute a similarity score.

    Call Signature
    --------------
    token_embedding : gluonnlp.embedding.TokenEmbedding
        Embedding to evaluate.
    dataset : mx.gluon.Dataset
        Dataset consisting of rows with 3 elements: [word1, word2, score]

    """

    def __init__(self,
                 binary_score_metric=SpearmanRankCorrelation(),
                 similarity_function=cosine_similarity):
        super(WordEmbeddingSimilarityEvaluator, self).__init__()
        self.binary_score_metric = binary_score_metric
        self.similarity_function = similarity_function

        if isinstance(self.similarity_function, str):
            assert self.similarity_function in ['cosinesimilarity']

    def __call__(self, token_embedding, dataset):
        # Construct nd arrays from dataset
        w1s = []
        w2s = []
        groundtruth_scores = []
        for word1, word2, score in dataset:
            if (word1 in token_embedding and word2 in token_embedding):
                w1s.append(word1)
                w2s.append(word2)
                groundtruth_scores.append(score)

        print(('Using {num_use} of {num_total} word pairs '
               'from {ds} for evaluation.').format(
                   num_use=len(w1s),
                   num_total=len(dataset),
                   ds=dataset.__class__.__name__))

        if not len(w1s):
            return 0

        w1s_embedding = mx.nd.L2Normalization(token_embedding[w1s])
        w2s_embedding = mx.nd.L2Normalization(token_embedding[w2s])
        groundtruth_scores = mx.nd.array(groundtruth_scores)

        similarity_pred = self.similarity_function(w1s_embedding,
                                                   w2s_embedding)
        score = self.binary_score_metric.update(similarity_pred,
                                                groundtruth_scores)

        return self.binary_score_metric.get()


###############################################################################
# Evaluation code
###############################################################################

# Word similarity based evaluation
similarity_evaluator = WordEmbeddingSimilarityEvaluator()
similarity_datasets = []
if args.eval_similarity:
    similarity_datasets_classnames = \
        nlp.data.word_embedding_evaluation.word_similarity_datasets
    if args.eval_similarity == '*':
        args.eval_similarity = similarity_datasets_classnames

    for ds in args.eval_similarity:
        if ds not in similarity_datasets_classnames:
            print(('{ds} is not a supported dataset. '
                   'Only {supported} are supported').format(
                       ds=ds,
                       supported=', '.join(similarity_datasets_classnames)))
            continue

        ds_class = eval('nlp.data.{ds}'.format(ds=ds))
        similarity_datasets.append(ds_class())


def evaluate():
    # Similarity based evaluation
    for dataset in similarity_datasets:
        print(similarity_evaluator(token_embedding, dataset))


if __name__ == '__main__':
    start_pipeline_time = time.time()
    evaluate()
