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
"""Utility functions for parallel processing."""
import threading
try:
    import Queue as queue
except ImportError:
    import queue

__all__ = ['Parallelizable', 'Parallel']

class Parallelizable:
    """ Base class for parallelizable unit of work, which can be invoked by `Parallel`.
    The subclass must implement the `forward_backward` method, and be used
    together with `Parallel`. For example::

        class ParallelNet(Parallelizable):
            def __init__(self):
                self._net = Model()
                self._loss = gluon.loss.SoftmaxCrossEntropyLoss()

            def forward_backward(x):
                data, label = x
                with mx.autograd.record():
                    out = self._net(data)
                    loss = self._loss(out, label)
                loss.backward()
                return loss

        net = ParallelNet()
        ctx = [mx.gpu(0), mx.gpu(1)]
        parallel = Parallel(len(ctx), net)
        # Gluon block is initialized after forwarding the first batch
        initialized = False

        for batch in batches:
            if not initialized:
                # The first batch cannot be forwarded in parallel
                losses = []
                for x in gluon.utils.split_and_load(batch, ctx):
                    losses.append(parallel.forward_backward(x))
                initialized = True
            else:
                for x in gluon.utils.split_and_load(batch, ctx):
                    parallel.put(x)
                losses = [parallel.get() for _ in ctx]
            trainer.step()
    """
    def forward_backward(self, x):
        """ Forward and backward computation. """
        raise NotImplementedError()

class Parallel:
    """ Class for parallel processing with `Parallelizable`s. It invokes a
    `Parallelizable` with multiple Python threads. For example::

        class ParallelNet(Parallelizable):
            def __init__(self):
                self._net = Model()
                self._loss = gluon.loss.SoftmaxCrossEntropyLoss()

            def forward_backward(x):
                data, label = x
                mx.autograd.record():
                    out = self._net(data)
                    loss = self._loss(out, label)
                loss.backward()
                return loss

        net = ParallelNet()
        ctx = [mx.gpu(0), mx.gpu(1)]
        parallel = Parallel(len(ctx), net)

        for batch in batches:
            for data in gluon.utils.split_and_load(batch, ctx):
                parallel.put(data)
            losses = [parallel.get() for _ in ctx]
            trainer.step()

    Parameters
    ----------
    num_workers : int
        Number of worker threads.
    parallelizable :
        Parallelizable net whose `forward` and `backward` methods are invoked
        by multiple worker threads.
    """

    class _StopSignal:
        """ Internal class to signal stop. """
        def __init__(self, msg):
            self._msg = msg

    def __init__(self, num_workers, parallizable):
        self._in_queue = queue.Queue(-1)
        self._out_queue = queue.Queue(-1)
        self._num_workers = num_workers
        self._threads = []
        self._parallizable = parallizable

        def _worker(in_queue, out_queue, parallel):
            while True:
                x = in_queue.get()
                if isinstance(x, self._StopSignal):
                    return
                out = parallel.forward_backward(x)
                out_queue.put(out)

        arg = (self._in_queue, self._out_queue, self._parallizable)
        for _ in range(num_workers):
            thread = threading.Thread(target=_worker, args=arg)
            self._threads.append(thread)
            thread.start()

    def put(self, x):
        """ Assign input `x` to an available worker and invoke
        `parallizable.forward_backward` with x. """
        self._in_queue.put(x)

    def get(self):
        """ Get an output of previous `parallizable.forward_backward` calls.
        This method blocks if none of previous `parallizable.forward_backward`
        calls have return any result. """
        return self._out_queue.get()

    def __del__(self):
        for thread in self._threads:
            if thread.is_alive():
                self._in_queue.put(self._StopSignal("stop"))
        for thread in self._threads:
            thread.join(10)
