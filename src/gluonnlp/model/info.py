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
"""Building blocks and utility for models."""
__all__ = ['list_models', 'list_datasets']


def list_models():
    """Returns the list of models in nlp
    """
    models = ['awd_lstm_lm_1150', 'awd_lstm_lm_600',
              'standard_lstm_lm_200', 'standard_lstm_lm_650',
              'standard_lstm_lm_1500', 'big_rnn_lm_2048_512',
              'transformer_en_de_512', 'bert_12_768_12',
              'bert_24_1024_16',
              'elmo_2x1024_128_2048cnn_1xhighway',
              'elmo_2x2048_256_2048cnn_1xhighway',
              'elmo_2x4096_512_2048cnn_2xhighway'
              ]

    return models

def list_datasets():
    """Returns the list of datasets in nlp
    """
    datasets = ['WikiText2', 'WikiText103', 'WikiText2Raw',
                'WikiText103Raw', 'GBWStream', 'IMDB', 'MR',
                'SST_1', 'SST_2', 'SUBJ', 'TREC', 'CR', 'MPQA',
                'WordSim353', 'MEN', 'RadinskyMTurk', 'RareWords',
                'SimLex999', 'SimVerb3500', 'SemEval17Task2',
                'BakerVerb143', 'YangPowersVerb130',
                'GoogleAnalogyTestSet', 'BiggerAnalogyTestSet',
                'CoNLL2000', 'CoNLL2001', 'CoNLL2002', 'CoNLL2004',
                'UniversalDependencies21', 'IWSLT2015', 'WMT2014',
                'WMT2014BPE', 'WMT2016', 'WMT2016BPE', 'ATISDataset',
                'SNIPSDataset', 'SQuAD', 'GlueCoLA', 'GlueSST2',
                'GlueSTSB', 'GlueQQP', 'GlueRTE', 'GlueMNLI',
                'GlueQNLI', 'GlueWNLI', 'GlueMRPC'
                ]

    return datasets
