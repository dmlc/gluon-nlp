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

import os

from . import (batchify, candidate_sampler, conll, corpora, dataloader,
               dataset, question_answering, registry, sampler, sentiment,
               stream, super_glue, transforms, translation, utils,
               word_embedding_evaluation, intent_slot, glue, datasetloader)
from .candidate_sampler import *
from .conll import *
from .glue import *
from .super_glue import *
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
from .utils import _load_pretrained_sentencepiece_tokenizer
from .word_embedding_evaluation import *
from .intent_slot import *
from .datasetloader import *

from ..base import get_home_dir

__all__ = (['batchify', 'get_tokenizer'] + utils.__all__ + transforms.__all__
           + sampler.__all__ + dataset.__all__ + corpora.__all__ + sentiment.__all__
           + word_embedding_evaluation.__all__ + stream.__all__ + conll.__all__
           + translation.__all__ + registry.__all__ + question_answering.__all__
           + dataloader.__all__ + candidate_sampler.__all__ + intent_slot.__all__
           + glue.__all__ + super_glue.__all__
           + datasetloader.__all__)  # pytype: disable=attribute-error


def get_tokenizer(model_name, dataset_name,
                  vocab=None, root=os.path.join(get_home_dir(), 'data'),
                  **kwargs):
    """Returns a pre-defined tokenizer by name.

    Parameters
    ----------
    model_name : str
        Options include 'bert_24_1024_16', 'bert_12_768_12', 'roberta_12_768_12',
        'roberta_24_1024_16' and 'ernie_12_768_12'
    dataset_name : str
        The supported datasets for model_name of either bert_24_1024_16 and
        bert_12_768_12 are 'book_corpus_wiki_en_cased',
        'book_corpus_wiki_en_uncased'.
        For model_name bert_12_768_12 'wiki_cn_cased',
        'wiki_multilingual_uncased', 'wiki_multilingual_cased',
        'scibert_scivocab_uncased', 'scibert_scivocab_cased',
        'scibert_basevocab_uncased','scibert_basevocab_cased',
        'biobert_v1.0_pmc', 'biobert_v1.0_pubmed', 'biobert_v1.0_pubmed_pmc',
        'biobert_v1.1_pubmed',
        'clinicalbert',
        'kobert_news_wiki_ko_cased' are supported.
        For model_name roberta_12_768_12 and roberta_24_1024_16
        'openwebtext_ccnews_stories_books_cased' is supported.
        For model_name ernie_12_768_12
        'baidu_ernie_uncased'.
        is additionally supported.
    vocab : gluonnlp.vocab.BERTVocab or None, default None
        Vocabulary for the dataset. Must be provided if tokenizer is based on
        vocab.
    root : str, default '$MXNET_HOME/models' with MXNET_HOME defaults to '~/.mxnet'
        Location for keeping the model parameters.

    Returns
    -------
    gluonnlp.data.BERTTokenizer or gluonnlp.data.GPT2BPETokenizer or
    gluonnlp.data.SentencepieceTokenizer

    Examples
    --------
    >>> model_name = 'bert_12_768_12'
    >>> dataset_name = 'book_corpus_wiki_en_uncased'
    >>> _, vocab = gluonnlp.model.get_model(model_name,
    ...                                     dataset_name=dataset_name,
    ...                                     pretrained=False, root='./model')
    -etc-
    >>> tokenizer = gluonnlp.data.get_tokenizer(model_name, dataset_name, vocab)
    >>> tokenizer('Habit is second nature.')
    ['habit', 'is', 'second', 'nature', '.']
    """
    model_name, dataset_name = model_name.lower(), dataset_name.lower()
    model_dataset_name = '_'.join([model_name, dataset_name])
    model_dataset_names = {'roberta_12_768_12_openwebtext_ccnews_stories_books_cased':
                           [GPT2BPETokenizer, {'lower': False}],
                           'roberta_24_1024_16_openwebtext_ccnews_stories_books_cased':
                           [GPT2BPETokenizer, {'lower': False}],
                           'bert_12_768_12_book_corpus_wiki_en_cased':
                           [BERTTokenizer, {'lower': False}],
                           'bert_12_768_12_book_corpus_wiki_en_uncased':
                           [BERTTokenizer, {'lower': True}],
                           'bert_12_768_12_openwebtext_book_corpus_wiki_en_uncased':
                           [BERTTokenizer, {'lower': True}],
                           'bert_12_768_12_wiki_multilingual_uncased':
                           [BERTTokenizer, {'lower': False}],
                           'bert_12_768_12_wiki_multilingual_cased':
                           [BERTTokenizer, {'lower': True}],
                           'bert_12_768_12_wiki_cn_cased':
                           [BERTTokenizer, {'lower': False}],
                           'bert_24_1024_16_book_corpus_wiki_en_cased':
                           [BERTTokenizer, {'lower': False}],
                           'bert_24_1024_16_book_corpus_wiki_en_uncased':
                           [BERTTokenizer, {'lower': True}],
                           'bert_12_768_12_scibert_scivocab_uncased':
                           [BERTTokenizer, {'lower': True}],
                           'bert_12_768_12_scibert_scivocab_cased':
                           [BERTTokenizer, {'lower': False}],
                           'bert_12_768_12_scibert_basevocab_uncased':
                           [BERTTokenizer, {'lower': True}],
                           'bert_12_768_12_scibert_basevocab_cased':
                           [BERTTokenizer, {'lower': False}],
                           'bert_12_768_12_biobert_v1.0_pmc_cased':
                           [BERTTokenizer, {'lower': False}],
                           'bert_12_768_12_biobert_v1.0_pubmed_cased':
                           [BERTTokenizer, {'lower': False}],
                           'bert_12_768_12_biobert_v1.0_pubmed_pmc_cased':
                           [BERTTokenizer, {'lower': False}],
                           'bert_12_768_12_biobert_v1.1_pubmed_cased':
                           [BERTTokenizer, {'lower': False}],
                           'bert_12_768_12_clinicalbert_uncased':
                           [BERTTokenizer, {'lower': True}],
                           'bert_12_768_12_kobert_news_wiki_ko_cased':
                           [_load_pretrained_sentencepiece_tokenizer, {'num_best': 0, 'alpha':1.0}],
                           'ernie_12_768_12_baidu_ernie_uncased':
                           [BERTTokenizer, {'lower': True}]}
    if model_dataset_name not in model_dataset_names:
        raise ValueError(
            'Model name %s is not supported. Available options are\n\t%s'%(
                model_dataset_name, '\n\t'.join(sorted(model_dataset_names.keys()))))
    tokenizer_cls, extra_args = model_dataset_names[model_dataset_name]
    kwargs = {**extra_args, **kwargs}
    if tokenizer_cls is BERTTokenizer:
        assert vocab is not None, 'Must specify vocab if loading BERTTokenizer'
        return tokenizer_cls(vocab, **kwargs)
    elif tokenizer_cls is GPT2BPETokenizer:
        return tokenizer_cls(root=root)
    elif tokenizer_cls is _load_pretrained_sentencepiece_tokenizer:
        return tokenizer_cls(dataset_name, root, **kwargs)
    else:
        raise ValueError('Could not get any matched tokenizer interface.')
