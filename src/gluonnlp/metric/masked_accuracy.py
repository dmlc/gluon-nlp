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
from mxnet.metric import EvalMetric

__all__ = ['MaskedAccuracy']

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
