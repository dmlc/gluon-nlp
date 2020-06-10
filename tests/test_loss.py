import mxnet as mx
import numpy as np
import pytest
from numpy.testing import assert_allclose
import scipy.special as sspecial
from gluonnlp.loss import LabelSmoothCrossEntropyLoss
mx.npx.set_np()


@pytest.mark.parametrize('label_shape', [(5, 3), (3,), (2, 3, 2)])
@pytest.mark.parametrize('alpha', [0.0, 0.1])
@pytest.mark.parametrize('from_logits', [True, False])
@pytest.mark.parametrize('hybridize', [True, False])
def test_label_smoothing(label_shape, alpha, from_logits, hybridize):
    def _np_label_smoothing(pred, labels, alpha, from_logits):
        flatten_pred = pred.reshape((-1, pred.shape[-1]))
        flatten_labels = labels.reshape((-1,))
        smoothed_labels = np.full_like(flatten_pred,
                                       fill_value=alpha / flatten_pred.shape[-1])
        smoothed_labels[np.arange(flatten_pred.shape[0]), flatten_labels]\
            = 1 - alpha + alpha / flatten_pred.shape[-1]
        if not from_logits:
            flatten_logits = np.log(sspecial.softmax(flatten_pred, axis=-1))
        else:
            flatten_logits = flatten_pred
        # Calculate cross-entropy
        loss = - (smoothed_labels * flatten_logits).sum(axis=-1)
        return loss.reshape(labels.shape)
    label_num = 5
    loss = LabelSmoothCrossEntropyLoss(num_labels=label_num, alpha=alpha, from_logits=from_logits)
    if hybridize:
        loss.hybridize()
    if from_logits:
        pred = mx.np.random.uniform(-10, -1, label_shape + (label_num,))
    else:
        pred = mx.np.random.normal(0, 1, label_shape + (label_num,))
    labels = mx.np.random.randint(0, label_num, label_shape)
    out = loss(pred, labels)
    np_out = _np_label_smoothing(pred.asnumpy(), labels.asnumpy(), alpha, from_logits)
    assert_allclose(np_out, out.asnumpy(), 1E-4, 1E-4)

