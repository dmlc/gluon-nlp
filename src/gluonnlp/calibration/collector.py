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
# pylint:disable=redefined-outer-name,logging-format-interpolation
"""Bert layer output collector with threshold clipping for calibration"""

import ctypes
from mxnet import ndarray
from mxnet.base import NDArrayHandle, py_str
from mxnet.ndarray import NDArray

class BertLayerCollector:
    """Saves layer output min and max values in a dict with layer names as keys.
    The collected min and max values will be directly used as thresholds for quantization.
    """
    def __init__(self, clip_min=None, clip_max=None, logger=None):
        self.include_layer = lambda name: name.endswith('_output') or \
                                        name.endswith('reshape10_0') or \
                                        name.endswith('_mul0_0') or \
                                        name.endswith('_squeeze0_0')
        self.min_max_dict = {}
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.logger = logger

    def collect(self, name, arr):
        """Callback function for collecting min and max values from an NDArray."""
        name = py_str(name)
        if self.include_layer is not None and not self.include_layer(name):
            return
        handle = ctypes.cast(arr, NDArrayHandle)
        arr = NDArray(handle, writable=False)
        min_range = ndarray.min(arr).asscalar()
        max_range = ndarray.max(arr).asscalar()
        if name.find('gelu0_leakyrelu0') != -1 and max_range > self.clip_max:
            max_range = self.clip_max
        if name.find('layernorm0_layernorm0') != -1 and min_range < self.clip_min:
            min_range = self.clip_min
        if name in self.min_max_dict:
            cur_min_max = self.min_max_dict[name]
            self.min_max_dict[name] = (min(cur_min_max[0], min_range),
                                       max(cur_min_max[1], max_range))
        else:
            self.min_max_dict[name] = (min_range, max_range)
        if self.logger is not None:
            self.logger.info('Collecting layer %s min_range=%f, max_range=%f'
                             % (name, min_range, max_range))
