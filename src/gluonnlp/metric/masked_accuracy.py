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

import mxnet as mx
from mxnet.metric import register, alias, EvalMetric, check_label_shapes

__all__ = ['MaskedAccuracy']

@register
@alias('masked-acc')
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
    >>> print acc.get()
    ('accuracy', 0.5)
    >>> acc2 = MaskedAccuracy()
    >>> acc2.update(preds=predicts, labels=labels)
    >>> print acc2.get()
    ('accuracy', 0.6666667)
    """
    def __init__(self, axis=1, name='masked-accuracy',
                 output_names=None, label_names=None):
        super(MaskedAccuracy, self).__init__(
            name, axis=axis,
            output_names=output_names, label_names=label_names,
            has_global_stats=True)
        self.axis = axis

    def update(self, labels, preds, masks=None):
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
        num_corrects = []
        num_insts = []
        for label, pred_label, mask in zip(labels, preds, masks):
            if pred_label.shape != label.shape:
                pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.astype('int32', copy=False).reshape((-1,))
            label = label.astype('int32', copy=False).reshape((-1,))
            # flatten before checking shapes to avoid shape miss match
            check_label_shapes(label, pred_label)

            if mask is not None:
                mask = mask.astype('int32', copy=False).reshape((-1,))
                check_label_shapes(label, mask)
                num_correct = ((pred_label == label) * mask).sum()
                num_inst =  mask.sum()
            else:
                num_correct = (pred_label == label).sum()
                num_inst =  len(label)
            num_corrects.append(num_correct)
            num_insts.append(num_inst)
        for num_correct, num_inst in zip(num_corrects, num_insts):
            if isinstance(num_correct, mx.nd.NDArray):
                num_correct = num_correct.asscalar()
            if isinstance(num_inst, mx.nd.NDArray):
                num_inst = num_inst.asscalar()
            self.sum_metric += num_correct
            self.global_sum_metric += num_correct
            self.num_inst += num_inst
            self.global_num_inst += num_inst
