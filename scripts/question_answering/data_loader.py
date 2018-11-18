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

# pylint: disable=
r"""
This file contains the class of data loader.
"""
import json
import random

from config import opt


class DataLoader(object):
    r"""
    An implementation of SQuAD data loader.
    """

    def __init__(self, batch_size=32, **kwargs):
        self.batch_size = batch_size
        self.data_path = opt.data_path

        if kwargs['dev_set'] is True:
            self.data_file = opt.processed_dev_file_name
            self._is_dev = True
        else:
            self.data_file = opt.processed_train_file_name
            self._is_dev = False

        self.data = self._load_data()
        self.num_instance = len(self.data)
        self.total_batchs = self.num_instance // self.batch_size

    def random_next_batch(self):
        r"""
        return: List
        --------
            Batchify the dataset in an ordered way.
        """
        i = 0
        while i * self.batch_size < self.num_instance:
            yield self._format_data(random.sample(self.data, self.batch_size))
            i += 1

    def next_batch(self):
        r"""
        return: List
        --------
            Batchify the dataset in random way.
        """
        i = 0
        while i * self.batch_size < self.num_instance:
            yield self._format_data(self.data[i * self.batch_size: (i + 1) * self.batch_size])
            i += 1

    def _format_data(self, data):
        def format_one_instance(instance):
            if self._is_dev:
                return instance
            else:
                return instance[1:7]
        return list(map(format_one_instance, data))

    def _load_data(self):
        with open(self.data_path + self.data_file, 'r') as f:
            line = f.readline()
        return json.loads(line)
