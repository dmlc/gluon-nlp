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
"""Utility functions."""

import os
import logging
import inspect
import threading
try:
    import Queue as queue
except ImportError:
    import queue

__all__ = ['logging_config']


def logging_config(folder=None, name=None,
                   level=logging.DEBUG,
                   console_level=logging.INFO,
                   no_console=False):
    """ Config the logging.

    Parameters
    ----------
    folder : str or None
    name : str or None
    level : int
    console_level
    no_console: bool
        Whether to disable the console log
    Returns
    -------
    folder : str
        Folder that the logging file will be saved into.
    """
    if name is None:
        name = inspect.stack()[1][1].split('.')[0]
    if folder is None:
        folder = os.path.join(os.getcwd(), name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # Remove all the current handlers
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + '.log')
    print('All Logs will be saved to {}'.format(logpath))
    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)
    if not no_console:
        # Initialze the console logging
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return folder


class Parallel:
    """ Class for parallel processing with multiple Python threads.

    net = Net()
    loss = Loss()

    def body(data):
        x, y = data
        out = net(x)
        ls = loss(out, y)
        ls.backward()
        return ls

    ctx = [mx.gpu(0), mx.gpu(1)]
    parallel = Parallel(len(ctx), body)

    for batch in batches:
        for data in split_and_load(batch, ctx):
            parallel.put(data)
        trainer.step()
        losses = [parallel.get() for _ in ctx]

    """

    class _StopSignal:
        """ Internal class to signal stop. """
        def __init__(self, msg):
            self._msg = msg

    def __init__(self, num_workers, body):
        self._in_queue = queue.Queue(-1)
        self._out_queue = queue.Queue(-1)
        self._num_workers = num_workers
        self._threads = []
        self._body = body

        def _worker(in_queue, out_queue, body_func):
            while True:
                data = in_queue.get()
                if isinstance(data, self._StopSignal):
                    return
                out = body_func(data)
                out_queue.put(out)

        arg = (self._in_queue, self._out_queue, self._body)
        for _ in range(num_workers):
            thread = threading.Thread(target=_worker, args=arg)
            self._threads.append(thread)
            thread.start()

    def put(self, data):
        self._in_queue.put(data)

    def get(self):
        return self._out_queue.get()

    def __del__(self):
        for thread in self._threads:
            if thread.is_alive():
                self._in_queue.put(self._StopSignal("stop"))
        for thread in self._threads:
            thread.join(10)
