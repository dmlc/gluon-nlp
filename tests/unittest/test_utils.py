import pytest
import numpy as np
import mxnet as mx
import gluonnlp as nlp

def test_parallel():
    class ParallelNet(nlp.utils.Parallelizable):
        def __init__(self, net, loss):
            self._net = net
            self._loss = loss

        def forward_backward(self, x):
            data, label = x
            with mx.autograd.record():
                out = self._net(data)
                loss = self._loss(out, label)
            loss.backward()
            return loss
    # model
    net = mx.gluon.nn.Dense(2)
    loss = mx.gluon.loss.SoftmaxCELoss()
    ctxs = [mx.cpu(0), mx.cpu(1)]
    net.initialize(ctx=ctxs)
    params = net.collect_params()

    # parallel model
    para_net = ParallelNet(net, loss)
    parallel = nlp.utils.Parallel(len(ctxs), para_net)

    # sample data
    data = mx.nd.random.uniform(shape=(2,5))
    label = mx.nd.array([[0], [1]])
    data_list = mx.gluon.utils.split_and_load(data, ctxs)
    label_list = mx.gluon.utils.split_and_load(label, ctxs)

    # train parallel
    epoch = 2
    params.zero_grad()
    params.setattr('req', 'add')
    parallel_loss = 0
    for i in range(epoch):
        for x, y in zip(data_list, label_list):
            parallel.put((x,y))
        for x, y in zip(data_list, label_list):
            ls = parallel.get()
            parallel_loss += ls.asscalar()

    grads = params['dense0_weight'].list_grad()
    parallel_grads_np = [grad.asnumpy() for grad in grads]

    # train serial
    params.zero_grad()
    params.setattr('req', 'add')
    serial_loss = 0
    for i in range(epoch):
        with mx.autograd.record():
            for x, y in zip(data_list, label_list):
                ls = loss(net(x), y)
                ls.backward()
                serial_loss += ls.asscalar()

    grads = params['dense0_weight'].list_grad()
    serial_grads_np = [grad.asnumpy() for grad in grads]
    assert serial_loss == parallel_loss
    for para_grad, serial_grad in zip(parallel_grads_np, serial_grads_np):
        mx.test_utils.assert_almost_equal(para_grad, serial_grad)
