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

# pylint: disable=abstract-method
"""Helper functions."""

__all__ = ['_str_types', 'numba_njit', 'numba_prange', 'numba_jitclass', 'numba_types']

try:
    _str_types = (str, unicode)
except NameError:  # Python 3
    _str_types = (str, )

try:
    from numba import njit, prange, jitclass, types
    numba_njit = njit(nogil=True)
    numba_prange = prange
    numba_jitclass = jitclass
    numba_types = types
except ImportError:
    # Define numba shims
    def identity(f):
        return f

    def numba_jitclass(spec):
        # pylint: disable=unused-argument
        return identity

    class NumbaTypes(object):
        """Shim for numba.types"""
        class NumbaType(object):
            """Shim for numba.types.type"""
            def __getitem__(self, x):
                # pylint: disable=unused-argument
                pass

            def __call__(self, *args, **kwargs):
                # pylint: disable=unused-argument
                pass

        def __getattr__(self, attr):
            return self.NumbaType()

    numba_types = NumbaTypes()
    numba_njit = identity
    numba_prange = range
