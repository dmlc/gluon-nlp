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
"""Utility functions for version checking."""
import warnings

__all__ = ['check_version']

def check_version(min_version, warning_only=False):
    """Check the version of gluonnlp satisfies the provided minimum version.
    An exception is thrown if the check does not pass.

    Parameters
    ----------
    min_version : str
        Minimum version
    warning_only : bool
        Printing a warning instead of throwing an exception.
    """
    from .. import __version__
    from packaging.version import parse
    bad_version = parse(__version__.replace('.dev', '')) < parse(min_version)
    if bad_version:
        msg = 'Installed GluonNLP version (%s) does not satisfy the ' \
              'minimum required version (%s)'%(__version__, min_version)
        if warning_only:
            warnings.warn(msg)
        else:
            raise AssertionError(msg)
