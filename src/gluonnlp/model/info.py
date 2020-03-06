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
__all__ = ['list_models']

from . import (attention_cell, bert, bilm_encoder, block,
               convolutional_encoder, elmo, highway, language_model,
               lstmpcellwithclip, parameter, sampled_block,
               seq2seq_encoder_decoder, sequence_sampler, train, transformer,
               utils, info)
from .bert import *
from .bilm_encoder import *
from .elmo import *
from .language_model import *
from .transformer import *

def list_models():
    """Returns the list of pretrained models
    """
    models = (bert.__all__ + bilm_encoder.__all__ + elmo.__all__ +
              language_model.__all__ + transformer.__all__)

    return models
