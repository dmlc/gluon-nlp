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
"""API to get list of pretrained models"""
__all__ = ['list_datasets']

from . import (batchify, candidate_sampler, conll, corpora, dataloader,
               dataset, question_answering, registry, sampler, sentiment,
               stream, super_glue, transforms, translation, utils,
               word_embedding_evaluation, intent_slot, glue, datasetloader,
               classification, baidu_ernie_data, bert, xlnet, info)
from .corpora import google_billion_word
from .corpora import large_text_compression_benchmark
from .corpora import wikitext
from .conll import *
from .glue import *
from .intent_slot import *
from .question_answering import *
from .sentiment import *
from .super_glue import *
from .translation import *
from .word_embedding_evaluation import *


def list_datasets():
    """Returns the list of datasets
    """
    datasets = (conll.__all__ + glue.__all__ + sentiment.__all__ +
                google_billion_word.__all__ + intent_slot.__all__ +
                large_text_compression_benchmark.__all__ +
                question_answering.__all__ + super_glue.__all__ +
                translation.__all__ + word_embedding_evaluation.__all__)

    return datasets
