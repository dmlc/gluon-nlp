import mxnet as mx
import numpy as np
from numpy.testing import assert_allclose
from mxnet.gluon import HybridBlock
from mxnet.gluon.data import DataLoader
import itertools
mx.npx.set_np()


def test_const():
    class Foo(HybridBlock):
        def __init__(self, prefix=None, params=None):
            super(Foo, self).__init__(prefix=prefix, params=params)
            self.weight = self.params.get_constant('const', np.ones((10, 10)))

        def hybrid_forward(self, F, x, weight):
            return x, weight.astype(np.float32)

    foo = Foo()
    foo.hybridize()
    foo.initialize()


def test_scalar():
    class Foo(HybridBlock):
        def hybrid_forward(self, F, x):
            return x * x * 2

    foo = Foo()
    foo.hybridize()
    foo.initialize()
    out = foo(mx.np.array(1.0))
    assert_allclose(out.asnumpy(), np.array(2.0))


def test_gluon_nonzero_hybridize():
    class Foo(HybridBlock):
        def __init__(self, prefix=None, params=None):
            super(Foo, self).__init__(prefix=prefix, params=params)

        def hybrid_forward(self, F, x):
            dat = F.np._internal.nonzero(x)
            return dat.sum() + dat

    foo = Foo()
    foo.hybridize()
    out = foo(mx.np.array([1, 0, 2, 0, 3, 0], ctx=mx.gpu()))
    out.wait_to_read()
    out = foo(mx.np.array([0, 0, 0, 0, 0, 0], ctx=mx.gpu()))
    out.wait_to_read()


def test_gluon_boolean_mask():
    class Foo(HybridBlock):
        def hybrid_forward(self, F, data, indices):
            mask = indices < 3
            data = F.npx.reshape(data, (-1, -2), reverse=True)
            mask = F.np.reshape(mask, (-1,))
            sel = F.np._internal.boolean_mask(data, mask)
            return sel
    data = mx.np.random.normal(0, 1, (5, 5, 5, 5, 16))
    indices = mx.np.random.randint(0, 5, (5, 5, 5, 5))
    data.attach_grad()
    indices.attach_grad()
    foo = Foo()
    foo.hybridize()
    with mx.autograd.record():
        out = foo(data, indices)
        out.backward()
    out.wait_to_read()


def test_basic_dataloader():
    def grouper(iterable, n, fillvalue=None):
        """Collect data into fixed-length chunks or blocks"""
        # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
        args = [iter(iterable)] * n
        return itertools.zip_longest(*args, fillvalue=fillvalue)
    ctx_l = [mx.cpu(i) for i in range(8)]
    dataset = [mx.np.ones((10,)) * i for i in range(100000)]
    dataloader = DataLoader(dataset, 2, num_workers=4, prefetch=10)

    for i, data_l in enumerate(grouper(dataloader, len(ctx_l))):
        for data, ctx in zip(data_l, ctx_l):
            if data is None:
                continue
            data = data.as_in_ctx(ctx)
            mx.npx.waitall()
