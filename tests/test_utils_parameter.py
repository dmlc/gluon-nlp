import pytest
import mxnet as mx
import numpy as np
from gluonnlp.utils.parameter import grad_global_norm, clip_grad_global_norm
from mxnet.test_utils import assert_almost_equal
mx.npx.set_np()


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

    contexts = [mx.cpu(0), mx.cpu(1)]
    net = mx.gluon.nn.HybridSequential()
    # Create a network with 8 layers
    for _ in range(8):
        net.add(mx.gluon.nn.Dense(1, weight_initializer='ones', bias_initializer='ones'))
    net.initialize(ctx=contexts)
    net.hybridize()
    trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', update_on_kvstore=False)
    for ctx in contexts:
        with mx.autograd.record():
            out = net(mx.np.ones((1, 1), ctx=ctx))
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
        for ctx in contexts:
            if max_norm > norm:
                assert_almost_equal(p.grad(ctx).asnumpy(), orig_grad)
            else:
                ratio = max_norm / norm
                assert_almost_equal(p.grad(ctx).asnumpy(), orig_grad * ratio)
