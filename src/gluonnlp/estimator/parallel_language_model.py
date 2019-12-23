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

# coding: utf-8
# pylint: disable=wildcard-import, unused-variable
""" Gluon Parallel Languange Model """

from gluonnlp.utils import Parallel, Parallelizable

__all__ = ['ParallelBigRNN']

class ParallelBigRNN(Parallelizable):
    def __init__(self, rnn, loss_fn):
        self._model = rnn
        self._loss = loss_fn

    def forward_backward(self, x):
        X, y, m, s, h = x
        with autograd.record():
            output, hidden, new_target = self._model(X, y, h, s)
            output = output.reshape((-3, -1))
            new_target = new_target.reshape((-1,))
            ls = self._loss(output, new_target) * m.reshape((-1,))
            ls = ls / args.batch_size
            ls.backward()
        return hidden, ls

