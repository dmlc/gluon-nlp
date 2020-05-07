import pytest
import mxnet as mx
from mxnet.gluon import nn
import numpy as np
from numpy.testing import assert_allclose
from gluonnlp.utils.misc import AverageSGDTracker
mx.npx.set_np()


def test_average_sgd_tracker():
    samples = [mx.np.random.normal(0, 1, (10, 3)) for _ in range(10)]
    no_moving_avg_param_l = []
    with_moving_avg_param_l = []
    moving_avg_param = None
    net_final_moving_avg_param = None
    for use_moving_avg in [False, True]:
        net = nn.HybridSequential(prefix='net_')
        with net.name_scope():
            net.add(nn.Dense(10))
            net.add(nn.Dense(3))
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
