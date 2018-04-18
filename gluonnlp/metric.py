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
"""Metrics not (yet) included in mxnet."""

import mxnet as mx

try:
    from scipy import stats
except ImportError:
    stats = None


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
    >>> predicts = [mx.nd.array([1,2,3,4,5])]
    >>> labels   = [mx.nd.array([5,6,7,8,7])]
    >>> pr = SpearmanRankCorrelation()
    >>> pr.update(labels, predicts)
    >>> print pr.get()
    ('spearmanr', 0.82078268166812329)

    """

    def __init__(self, name='spearmanr', output_names=None, label_names=None):
        super(SpearmanRankCorrelation, self).__init__(
            name, output_names=output_names, label_names=label_names)

        if stats is None:
            raise RuntimeError(
                'SpearmanRankCorrelation requires scipy.'
                'You may install scipy via `pip install scipy`.')

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
        self._labels.append(labels)
        self._preds.append(preds)

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self._labels = []
        self._preds = []

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """

        if not len(self._labels):
            return (self.name, float('nan'))

        labels = [
            mx.nd.concat(*ndarrays, dim=0) for ndarrays in zip(*self._labels)
        ]
        preds = [
            mx.nd.concat(*ndarrays, dim=0) for ndarrays in zip(*self._preds)
        ]

        sum_metric = 0
        num_inst = 0
        for label, pred in zip(labels, preds):
            mx.metric.check_label_shapes(label, pred, False, True)
            label = label.asnumpy()
            pred = pred.asnumpy()
            sum_metric += stats.spearmanr(pred.ravel(),
                                          label.ravel()).correlation
            num_inst += 1
        return (self.name, sum_metric / num_inst)
