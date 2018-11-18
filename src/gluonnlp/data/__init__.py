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

# pylint: disable=wildcard-import
"""This module includes common utilities such as data readers and counter."""

from . import (batchify, candidate_sampler, conll, corpora, dataloader,
               dataset, question_answering, registry, sampler, sentiment,
               stream, transforms, translation, utils,
               word_embedding_evaluation)
from .candidate_sampler import *
from .conll import *
from .corpora import *
from .dataloader import *
from .dataset import *
from .question_answering import *
from .registry import *
from .sampler import *
from .sentiment import *
from .stream import *
from .transforms import *
from .translation import *
from .utils import *
from .word_embedding_evaluation import *

__all__ = (['batchify'] + utils.__all__ + transforms.__all__ + sampler.__all__
           + dataset.__all__ + corpora.__all__ + sentiment.__all__ +
           word_embedding_evaluation.__all__ + stream.__all__ + conll.__all__ +
           translation.__all__ + registry.__all__ + question_answering.__all__
           + dataloader.__all__)
