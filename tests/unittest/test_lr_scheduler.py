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

from mxnet import gluon

import gluonnlp as nlp


def testReduceLROnPlateau():
    model = gluon.nn.Dense(2)
    model.initialize()
    trainer = gluon.Trainer(model.collect_params(), 'SGD')
    scheduler = nlp.lr_scheduler.ReduceLROnPlateau(trainer,
                                                   'min',
                                                   patience=0,
                                                   factor=0.1)
    base_loss = 0.1
    scheduler.step(base_loss)
    base_lr = scheduler.trainer.learning_rate
    next_loss = 0.11
    scheduler.step(next_loss)
    next_lr = scheduler.trainer.learning_rate
    expected_lr = base_lr * 0.1
    assert expected_lr == next_lr
