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
"""Translation datasets."""


__all__ = ['TOY']

import os
from gluonnlp.base import get_home_dir
from gluonnlp.data.translation import _TranslationDataset, _get_pair_key
from gluonnlp.data.registry import register


@register(segment=['train', 'val', 'test'])
class TOY(_TranslationDataset):
    """A Small Translation Dataset for Testing Scripts.

    Parameters
    ----------
    segment : str or list of str, default 'train'
        Dataset segment. Options are 'train', 'val', 'test' or their combinations.
    src_lang : str, default 'en'
        The source language. Option for source and target languages are 'en' <-> 'de'
    tgt_lang : str, default 'de'
        The target language. Option for source and target languages are 'en' <-> 'de'
    root : str, default '$MXNET_HOME/datasets/translation_test'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    """
    def __init__(self, segment='train', src_lang='en', tgt_lang='de',
                 root=os.path.join(get_home_dir(), 'datasets', 'translation_test')):
        self._supported_segments = ['train', 'val', 'test']
        self._archive_file = {_get_pair_key('en', 'de'):
                                  ('translation_test.zip',
                                   '14f6c8e31ac6ec84ce469b4c196d60b4c86a179d')}
        self._data_file = {_get_pair_key('en', 'de'):
                               {'train_en': ('train.en',
                                             'aa7f22b91eb93390fd342a57a81f51f53ed29542'),
                                'train_de': ('train.de',
                                             'f914217ce23ddd8cac07e761a75685c043d4f6d3'),
                                'val_en': ('train.en',
                                           'aa7f22b91eb93390fd342a57a81f51f53ed29542'),
                                'val_de': ('train.de',
                                           'f914217ce23ddd8cac07e761a75685c043d4f6d3'),
                                'test_en': ('train.en',
                                            'aa7f22b91eb93390fd342a57a81f51f53ed29542'),
                                'test_de': ('train.de',
                                            'f914217ce23ddd8cac07e761a75685c043d4f6d3'),
                                'vocab_en': ('vocab.en.json',
                                             'c7c6af4603ea70f0a4af2460a622333fbd014050'),
                                'vocab_de' : ('vocab.de.json',
                                              '5b6f1be36a3e3cb9946b86e5d0fc73d164fda99f')}}
        super(TOY, self).__init__('translation_test', segment=segment, src_lang=src_lang,
                                  tgt_lang=tgt_lang, root=root)
