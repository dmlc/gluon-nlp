import pytest
from gluonnlp import initializer
import mxnet as mx
from mxnet.gluon import nn
mx.npx.set_np()


def test_truncnorm_string_alias_works():
    try:
        layer = nn.Dense(prefix="test_layer", in_units=1, units=1, weight_initializer='truncnorm')
        layer.initialize()
    except RuntimeError:
        pytest.fail('Layer couldn\'t be initialized')


def test_truncnorm_all_values_inside_boundaries():
    mean = 0
    std = 0.01
    layer = nn.Dense(prefix="test_layer", in_units=1, units=1000)
    layer.initialize(init=initializer.TruncNorm(mean, std))
    assert (layer.weight.data() <= 2 * std).asnumpy().all()
    assert (layer.weight.data() >= -2 * std).asnumpy().all()


def test_truncnorm_generates_values_with_defined_mean_and_std():
    from scipy import stats

    mean = 10
    std = 5
    layer = nn.Dense(prefix="test_layer", in_units=1, units=100000)
    layer.initialize(init=initializer.TruncNorm(mean, std))
    samples = layer.weight.data().reshape((-1, )).asnumpy()

    p_value = stats.kstest(samples, 'truncnorm', args=(-2, 2, mean, std)).pvalue
    assert p_value > 0.0001
