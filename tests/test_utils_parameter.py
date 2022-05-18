import pytest
import mxnet as mx
import numpy as np
from numpy.testing import assert_allclose
from mxnet.gluon import nn
from gluonnlp.utils.parameter import grad_global_norm, clip_grad_global_norm, AverageSGDTracker
from mxnet.test_utils import assert_almost_equal



def test_average_sgd_tracker():
    samples = [mx.np.random.normal(0, 1, (10, 3)) for _ in range(10)]
    no_moving_avg_param_l = []
    with_moving_avg_param_l = []
    moving_avg_param = None
    net_final_moving_avg_param = None
    for use_moving_avg in [False, True]:
        net = nn.HybridSequential()
        net.add(nn.Dense(10), nn.Dense(3))
        net.initialize(init=mx.init.One())
        net.hybridize()
        trainer = mx.gluon.Trainer(net.collect_params(), 'adam')
        if use_moving_avg:
            model_averager = AverageSGDTracker(net.collect_params())
        for sample in samples:
            out = sample ** 3 + sample
            with mx.autograd.record():
                pred = net(sample)
                loss = ((out - pred) ** 2).mean()
            loss.backward()
            trainer.step(1.0)
            if use_moving_avg:
                model_averager.step()
                print(model_averager.average_params)
            if use_moving_avg:
                with_moving_avg_param_l.append({k: v.data().asnumpy() for k, v in net.collect_params().items()})
            else:
                no_moving_avg_param_l.append({k: v.data().asnumpy() for k, v in net.collect_params().items()})
        if use_moving_avg:
            model_averager.copy_back()
            moving_avg_param = {k: v.asnumpy() for k, v in model_averager.average_params.items()}
            net_final_moving_avg_param = {k: v.data().asnumpy() for k, v in net.collect_params().items()}
    # Match the recorded params
    calculated_moving_param = {k: np.zeros(v.shape) for k, v in no_moving_avg_param_l[0].items()}
    for step, (no_moving_avg_param, with_moving_avg_param) in enumerate(zip(no_moving_avg_param_l,
                                                                            with_moving_avg_param_l)):
        decay = 1.0 / (step + 1)
        assert len(no_moving_avg_param) == len(with_moving_avg_param)
        for k in with_moving_avg_param:
            assert_allclose(no_moving_avg_param[k], with_moving_avg_param[k])
            calculated_moving_param[k] += decay * (with_moving_avg_param[k] - calculated_moving_param[k])
    assert len(moving_avg_param) == len(net_final_moving_avg_param)
    for k in moving_avg_param:
        assert_allclose(moving_avg_param[k], calculated_moving_param[k], 1E-5, 1E-5)
        assert_allclose(moving_avg_param[k], net_final_moving_avg_param[k], 1E-5, 1E-5)


@pytest.mark.parametrize('max_norm,check_isfinite',
                         [(1, True),
                          (1, False),
                          (100, True),
                          (100, False)])
def test_clip_grad_norm(max_norm, check_isfinite):

    def gt_grad_global_norm(parameters):
        ret = 0
        for p in parameters:
            grads = p.list_grad()
            ret += (grads[0].asnumpy() ** 2).sum()
        return np.sqrt(ret)

    devices = [mx.cpu(0), mx.cpu(1)]
    net = mx.gluon.nn.HybridSequential()
    # Create a network with 8 layers
    for _ in range(8):
        net.add(mx.gluon.nn.Dense(1, weight_initializer='ones', bias_initializer='ones'))
    net.initialize(device=devices)
    net.hybridize()
    trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', update_on_kvstore=False)
    for device in devices:
        with mx.autograd.record():
            out = net(mx.np.ones((1, 1), device=device))
        out.backward()
    trainer.allreduce_grads()
    # Cache the original gradient for checking
    original_grad_l = [p.list_grad()[0].asnumpy() for p in net.collect_params().values()]
    norm = grad_global_norm(net.collect_params().values())
    gt_norm = gt_grad_global_norm(net.collect_params().values())
    assert_almost_equal(norm, gt_norm, atol=1e-5)
    with mx.cpu(2):
        norm, ratio, is_finite = clip_grad_global_norm(net.collect_params().values(), max_norm,
                                                       check_isfinite)
    assert_almost_equal(norm, gt_norm, atol=1e-5)
    for p, orig_grad in zip(net.collect_params().values(), original_grad_l):
        for device in devices:
            if max_norm > norm:
                assert_almost_equal(p.grad(device).asnumpy(), orig_grad)
            else:
                ratio = max_norm / norm
                assert_almost_equal(p.grad(device).asnumpy(), orig_grad * ratio)
