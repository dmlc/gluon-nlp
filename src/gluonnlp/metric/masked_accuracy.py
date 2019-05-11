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
"""Masked accuracy metric."""

from mxnet import ndarray
from mxnet.metric import check_label_shapes

__all__ = ['EvalMetric', 'MaskedAccuracy']

class EvalMetric(object):
    """Base class for all evaluation metrics.

    .. note::

        This is a base class that provides common metric interfaces.
        One should not use this class directly, but instead create new metric
        classes that extend it.

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
    """
    def __init__(self, name, output_names=None,
                 label_names=None, **kwargs):
        self.name = str(name)
        self.output_names = output_names
        self.label_names = label_names
        self._has_global_stats = kwargs.pop('has_global_stats', False)
        self._kwargs = kwargs
        self.reset()

    def __str__(self):
        return 'EvalMetric: {}'.format(dict(self.get_name_value()))

    def get_config(self):
        """Save configurations of metric. Can be recreated
        from configs with metric.create(``**config``)
        """
        config = self._kwargs.copy()
        config.update({
            'metric': self.__class__.__name__,
            'name': self.name,
            'output_names': self.output_names,
            'label_names': self.label_names})
        return config

    def update_dict(self, label, pred):
        """Update the internal evaluation with named label and pred

        Parameters
        ----------
        labels : OrderedDict of str -> NDArray
            name to array mapping for labels.

        preds : OrderedDict of str -> NDArray
            name to array mapping of predicted outputs.
        """
        if self.output_names is not None:
            pred = [pred[name] for name in self.output_names]
        else:
            pred = list(pred.values())

        if self.label_names is not None:
            label = [label[name] for name in self.label_names]
        else:
            label = list(label.values())

        self.update(label, pred)

    def update(self, labels, preds):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        raise NotImplementedError()

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = 0
        self.sum_metric = 0.0
        self.global_num_inst = 0
        self.global_sum_metric = 0.0

    def reset_local(self):
        """Resets the local portion of the internal evaluation results
        to initial state."""
        self.num_inst = 0
        self.sum_metric = 0.0

    def get(self):
        """Gets the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            return (self.name, self.sum_metric / self.num_inst)

    def get_global(self):
        """Gets the current global evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self._has_global_stats:
            if self.global_num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.global_sum_metric / self.global_num_inst)
        else:
            return self.get()

    def get_name_value(self):
        """Returns zipped name and value pairs.

        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))

    def get_global_name_value(self):
        """Returns zipped name and value pairs for global results.

        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        if self._has_global_stats:
            name, value = self.get_global()
            if not isinstance(name, list):
                name = [name]
            if not isinstance(value, list):
                value = [value]
            return list(zip(name, value))
        else:
            return self.get_name_value()


class MaskedAccuracy(EvalMetric):
    """Computes accuracy classification score.

    The accuracy score is defined as

    .. math::
        \\text{accuracy}(y, \\hat{y}, mask) = \\frac{1}{m} \\sum_{i=0}^{n-1}
        \\text{mask_i}(\\hat{y_i} == y_i)

        \\text{m} = \\sum_{i=0}^{n-1} mask_i

    Parameters
    ----------
    axis : int, default=1
        The axis that represents classes
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
    >>> labels   = [mx.nd.array([0, 1, 1])]
    >>> masks    = [mx.nd.array([1, 1, 0])]
    >>> acc = MaskedAccuracy()
    >>> acc.update(preds=predicts, labels=labels, masks=masks)
    >>> acc.get()
    ('masked-accuracy', 0.5)
    >>> acc2 = MaskedAccuracy()
    >>> acc2.update(preds=predicts, labels=labels)
    >>> acc2.get()
    ('masked-accuracy', 0.6666666666666666)
    """
    def __init__(self, axis=1, name='masked-accuracy',
                 output_names=None, label_names=None):
        super(MaskedAccuracy, self).__init__(
            name, axis=axis,
            output_names=output_names, label_names=label_names,
            has_global_stats=True)
        self.axis = axis

    def update(self, labels, preds, masks=None):
        # pylint: disable=arguments-differ
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data with class indices as values, one per sample.
        preds : list of `NDArray`
            Prediction values for samples. Each prediction value can either be the class index,
            or a vector of likelihoods for all classes.
        masks : list of `NDArray` or None, optional
            Masks for samples, with the same shape as `labels`. value of its element must
            be either 1 or 0. If None, all samples are considered valid.
        """
        labels, preds = check_label_shapes(labels, preds, True)
        masks = [None] * len(labels) if masks is None else masks

        for label, pred_label, mask in zip(labels, preds, masks):
            if pred_label.shape != label.shape:
                # TODO(haibin) topk does not support fp16. Issue tracked at:
                # https://github.com/apache/incubator-mxnet/issues/14125
                # topk is used because argmax is slow:
                # https://github.com/apache/incubator-mxnet/issues/11061
                pred_label = ndarray.topk(pred_label.astype('float32', copy=False),
                                          k=1, ret_typ='indices', axis=self.axis)

            # flatten before checking shapes to avoid shape miss match
            pred_label = pred_label.astype('int32', copy=False).reshape((-1,))
            label = label.astype('int32', copy=False).reshape((-1,))
            check_label_shapes(label, pred_label)

            if mask is not None:
                mask = mask.astype('int32', copy=False).reshape((-1,))
                check_label_shapes(label, mask)
                num_correct = ((pred_label == label) * mask).sum().asscalar()
                num_inst = mask.sum().asscalar()
            else:
                num_correct = (pred_label == label).sum().asscalar()
                num_inst = len(label)
            self.sum_metric += num_correct
            self.global_sum_metric += num_correct
            self.num_inst += num_inst
            self.global_num_inst += num_inst
