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
"""Common utilities for the named entity recognition task."""

import argparse
import pickle
from collections import namedtuple

import mxnet as mx
import gluonnlp as nlp

__all__ = ['get_bert_model', 'get_bert_dataset_name', 'get_context',
           'dump_metadata']

BERTModelMetadata = namedtuple('BERTModelMetadata', ['config', 'tag_vocab'])

def _metadata_file_path(checkpoint_prefix):
    """Gets the file path for meta data"""
    return checkpoint_prefix + '_metadata.pkl'


def dump_metadata(config, tag_vocab):
    """Dumps meta-data to the configured path"""
    metadata = BERTModelMetadata(config=config, tag_vocab=tag_vocab)
    with open(_metadata_file_path(config.save_checkpoint_prefix), 'wb') as ofp:
        pickle.dump(metadata, ofp)


def load_metadata(checkpoint_prefix):
    """Loads meta-data to the configured path"""
    with open(_metadata_file_path(checkpoint_prefix), 'rb') as ifp:
        metadata = pickle.load(ifp)
        return metadata.config, metadata.tag_vocab


def get_context(gpu_index):
    """This method gets context of execution"""
    context = None
    if gpu_index is None or gpu_index == '':
        context = mx.cpu()
    if isinstance(gpu_index, int):
        context = mx.gpu(gpu_index)
    return context


def str2bool(v):
    """Utility function for parsing boolean in argparse

    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    :param v: value of the argument
    :return:
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_bert_dataset_name(is_cased):
    """Returns relevant BERT dataset name, depending on whether we are using a cased model.

    Parameters
    ----------
    is_cased: bool
        Whether we are using a cased model.

    Returns
    -------
    str: Named of the BERT dataset.

    """
    if is_cased:
        return 'book_corpus_wiki_en_cased'
    else:
        return 'book_corpus_wiki_en_uncased'


def get_bert_model(bert_model, cased, ctx, dropout_prob):
    """Get pre-trained BERT model."""
    bert_dataset_name = get_bert_dataset_name(cased)

    return nlp.model.get_model(
        name=bert_model,
        dataset_name=bert_dataset_name,
        pretrained=True,
        ctx=ctx,
        use_pooler=False,
        use_decoder=False,
        use_classifier=False,
        dropout=dropout_prob,
        embed_dropout=dropout_prob)
