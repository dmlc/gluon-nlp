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
# pylint:disable=redefined-outer-name,logging-format-interpolation
"""Utility functions for data."""

__all__ = ['mkdir']

import os
import warnings
from .. import _constants as C

def mkdir(dirname):
    """Create directory.

    Parameters
    ----------
    dirname : str
        The name of the target directory to create.
    """
    if C.S3_PREFIX in dirname:
        warnings.warn('%s directory is not created as it contains %s'
                      %(dirname, C.S3_PREFIX))
        return
    dirname = os.path.expanduser(dirname)
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            # errno 17 means the file already exists
            if e.errno != 17:
                raise e
