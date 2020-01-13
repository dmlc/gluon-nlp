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

# pylint: disable=wildcard-import, arguments-differ
r"""Module for pre-defined NLP models.

This module contains definitions for the following model architectures:
-  `AWD`_

You can construct a model with random weights by calling its constructor. Because NLP models
are tied to vocabularies, you can either specify a dataset name to load and use the vocabulary
of that dataset:

.. code-block:: python

    import gluonnlp as nlp
    awd, vocab = nlp.model.awd_lstm_lm_1150(dataset_name='wikitext-2')

or directly specify a vocabulary object:

.. code-block:: python

    awd, vocab = nlp.model.awd_lstm_lm_1150(None, vocab=custom_vocab)

We provide pre-trained models for all the listed models.
These models can constructed by passing ``pretrained=True``:

.. code-block:: python

    awd, vocab = nlp.model.awd_lstm_lm_1150(dataset_name='wikitext-2'
                                            pretrained=True)

.. _AWD: https://arxiv.org/abs/1404.5997


-  `ELMo`_

You can construct a predefined ELMo model structure:

.. code-block:: python

    import gluonnlp as nlp
    elmo = nlp.model.elmo_2x1024_128_2048cnn_1xhighway(dataset_name='gbw')

You can also get a ELMo model with pretrained parameters:

.. code-block:: python

    import gluonnlp as nlp
    elmo = nlp.model.elmo_2x1024_128_2048cnn_1xhighway(dataset_name='gbw', pretrained=True)

.. _ELMo: https://arxiv.org/pdf/1802.05365.pdf
"""
import os

from . import (attention_cell, bert, bilm_encoder, block,
               convolutional_encoder, elmo, highway, language_model,
               lstmpcellwithclip, parameter, sampled_block,
               seq2seq_encoder_decoder, sequence_sampler, train, transformer,
               utils)
from .attention_cell import *
from .bert import *
from .bilm_encoder import BiLMEncoder
from .block import *
from .convolutional_encoder import *
from .elmo import *
from .highway import *
from .language_model import *
from .lstmpcellwithclip import LSTMPCellWithClip
from .parameter import *
from .sampled_block import *
from .seq2seq_encoder_decoder import *
from .sequence_sampler import *
from .transformer import *
from .translation import *
from .utils import *
from ..base import get_home_dir

__all__ = language_model.__all__ + sequence_sampler.__all__ + attention_cell.__all__ + \
          utils.__all__ + parameter.__all__ + block.__all__ + highway.__all__ + \
          convolutional_encoder.__all__ + sampled_block.__all__ + ['get_model', 'get_tokenizer'] + \
          ['train'] + bilm_encoder.__all__ + lstmpcellwithclip.__all__ + \
          elmo.__all__ + seq2seq_encoder_decoder.__all__ + transformer.__all__ + bert.__all__


def get_model(name, **kwargs):
    """Returns a pre-defined model by name.

    Parameters
    ----------
    name : str
        Name of the model.
    dataset_name : str or None, default None
        The dataset name on which the pre-trained model is trained.
        For language model, options are 'wikitext-2'.
        For ELMo, Options are 'gbw' and '5bw'.
        'gbw' represents 1 Billion Word Language Model Benchmark
        http://www.statmt.org/lm-benchmark/;
        '5bw' represents a dataset of 5.5B tokens consisting of
        Wikipedia (1.9B) and all of the monolingual news crawl data from WMT 2008-2012 (3.6B).
        If specified, then the returned vocabulary is extracted from
        the training set of the dataset.
        If None, then vocab is required, for specifying embedding weight size, and is directly
        returned.
    vocab : gluonnlp.Vocab or None, default None
        Vocabulary object to be used with the language model.
        Required when dataset_name is not specified.
        None Vocabulary object is required with the ELMo model.
    pretrained : bool, default False
        Whether to load the pre-trained weights for model.
    ctx : Context, default CPU
        The context in which to load the pre-trained weights.
    root : str, default '$MXNET_HOME/models' with MXNET_HOME defaults to '~/.mxnet'
        Location for keeping the model parameters.

    Returns
    -------
    gluon.Block, gluonnlp.Vocab, (optional) gluonnlp.Vocab
    """
    models = {'standard_lstm_lm_200' : standard_lstm_lm_200,
              'standard_lstm_lm_650' : standard_lstm_lm_650,
              'standard_lstm_lm_1500': standard_lstm_lm_1500,
              'awd_lstm_lm_1150': awd_lstm_lm_1150,
              'awd_lstm_lm_600': awd_lstm_lm_600,
              'big_rnn_lm_2048_512': big_rnn_lm_2048_512,
              'elmo_2x1024_128_2048cnn_1xhighway': elmo_2x1024_128_2048cnn_1xhighway,
              'elmo_2x2048_256_2048cnn_1xhighway': elmo_2x2048_256_2048cnn_1xhighway,
              'elmo_2x4096_512_2048cnn_2xhighway': elmo_2x4096_512_2048cnn_2xhighway,
              'transformer_en_de_512': transformer_en_de_512,
              'bert_12_768_12'       : bert_12_768_12,
              'bert_24_1024_16'      : bert_24_1024_16,
              'roberta_12_768_12'    : roberta_12_768_12,
              'roberta_24_1024_16'   : roberta_24_1024_16,
              'ernie_12_768_12'      : ernie_12_768_12}
    name = name.lower()
    if name not in models:
        raise ValueError(
            'Model %s is not supported. Available options are\n\t%s'%(
                name, '\n\t'.join(sorted(models.keys()))))
    return models[name](**kwargs)


def get_tokenizer(model_name, dataset_name,
                  vocab=None, root=os.path.join(get_home_dir(), 'models'),
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
    >>> tokenizer = gluonnlp.model.get_tokenizer(model_name, dataset_name, vocab)
    >>> tokenizer('Habit is second nature.')
    ['habit', 'is', 'second', 'nature', '.']
    """
    from ..data.utils import _load_pretrained_sentencepiece_tokenizer  # pylint: disable=import-outside-toplevel
    from ..data import BERTTokenizer, GPT2BPETokenizer  # pylint: disable=import-outside-toplevel

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
