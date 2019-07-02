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
"""Utility functions for files."""

__all__ = ['mkdir', 'glob']

import os
import warnings
import logging
import tempfile
import glob as _glob
from .. import _constants as C

def glob(url, separator=','):
    """Return a list of paths matching a pathname pattern.

    The pattern may contain simple shell-style wildcards.
    Input may also include multiple patterns, separated by separator.

    Parameters
    ----------
    url : str
        The name of the files
    separator : str, default is ','
        The separator in url to allow multiple patterns in the input
    """
    patterns = [url] if separator is None else url.split(separator)
    result = []
    for pattern in patterns:
        result.extend(_glob.glob(os.path.expanduser(pattern.strip())))
    return result

def mkdir(dirname):
    """Create directory.

    Parameters
    ----------
    dirname : str
        The name of the target directory to create.
    """
    if C.S3_PREFIX in dirname:
        warnings.warn('Directory %s is not created because it contains %s'
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

class _TempFilePath(object):
    """A TempFilePath that provides a path to a temporarily file, and automatically
    cleans up the temp file at exit.
    """
    def __init__(self):
        self.temp_dir = os.path.join(tempfile.gettempdir(), str(hash(os.times())))
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def __enter__(self):
        self.temp_path = os.path.join(self.temp_dir, str(hash(os.times())))
        return self.temp_path

    def __exit__(self, exec_type, exec_value, traceback):
        os.remove(self.temp_path)

def _transfer_file_s3(filename, s3_filename, upload=True):
    """Transfer a file between S3 and local file system."""
    try:
        import boto3
    except ImportError:
        raise ImportError('boto3 is required to support s3 URI. Please install'
                          'boto3 via `pip install boto3`')
    # parse s3 uri
    prefix_len = len(C.S3_PREFIX)
    bucket_idx = s3_filename[prefix_len:].index('/') + prefix_len
    bucket_name = s3_filename[prefix_len:bucket_idx]

    # filename after the bucket, excluding '/'
    key_name = s3_filename[bucket_idx + 1:]

    log_level = logging.getLogger().getEffectiveLevel()
    logging.getLogger().setLevel(logging.INFO)
    # upload to s3
    s3 = boto3.client('s3')
    if upload:
        s3.upload_file(filename, bucket_name, key_name)
    else:
        s3.download_file(bucket_name, key_name, filename)
    logging.getLogger().setLevel(log_level)
