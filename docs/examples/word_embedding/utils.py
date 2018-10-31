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
"""Word Embeddings Training Utilities
=====================================

"""

import logging
import time
from contextlib import contextmanager

import mxnet as mx


def get_context(args):
    if args.gpu is None or args.gpu == '':
        context = [mx.cpu()]
    elif isinstance(args.gpu, int):
        context = [mx.gpu(args.gpu)]
    else:
        context = [mx.gpu(int(i)) for i in args.gpu]
    return context


@contextmanager
def print_time(task):
    start_time = time.time()
    logging.info('Starting to %s', task)
    yield
    logging.info('Finished to {} in {:.2f} seconds'.format(
        task,
        time.time() - start_time))
