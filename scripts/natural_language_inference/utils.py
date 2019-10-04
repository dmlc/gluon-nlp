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
# Copyright 2018 Mengxiao Lin <linmx0130@gmail.com>.

"""
Utility functions.
"""

import logging

def logging_config(logpath=None,
                   level=logging.DEBUG,
                   console_level=logging.INFO,
                   no_console=False):
    """
    Config the logging.
    """
    logger = logging.getLogger('nli')
    # Remove all the current handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(filename)s:%(funcName)s: %(message)s')

    if logpath is not None:
        print('All Logs will be saved to {}'.format(logpath))
        logfile = logging.FileHandler(logpath, mode='w')
        logfile.setLevel(level)
        logfile.setFormatter(formatter)
        logger.addHandler(logfile)

    if not no_console:
        # Initialze the console logging
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logger.addHandler(logconsole)
