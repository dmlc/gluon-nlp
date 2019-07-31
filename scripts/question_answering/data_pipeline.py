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

# pylint: disable=
"""SQuAD data data preprocessing pipeline."""
import collections
import contextlib
import itertools
import json
import multiprocessing as mp
import os
import re
import time

import nltk
import numpy as np
import tqdm
from mxnet.gluon.data import Dataset

import gluonnlp as nlp
from gluonnlp import data, Vocab
from gluonnlp.data import SQuAD


class SQuADDataPipeline:
    """Main data processing pipeline class, which encapsulate all preprocessing logic. The class
    process the data in multiprocessing mode using Pool. It can save/load the result of processing,
    but since it happens in a single thread, it is usually faster to just process data from scratch.
    """

    def __init__(self, train_para_limit, train_ques_limit, dev_para_limit, dev_ques_limit,
                 ans_limit, char_limit, emb_file_name, num_workers=None, save_load_data=False,
                 data_root_path='./data'):
        """Method that creates a new instance. If an example is longer that provided limits it will
        be truncated for the dev set and filtered out for the training set.

        Parameters
        ----------
        train_para_limit : int
            Maximum characters of a paragraph for training dataset
        train_ques_limit : int
            Maximum characters of a question for training dataset
        dev_para_limit : int
            Maximum characters of a paragraph for dev dataset
        dev_ques_limit
            Maximum characters of a question for dev dataset
        ans_limit : int
            Maximum characters of an answer
        char_limit : int
            Maximum token (word) length of a paragraph, question or answer
        emb_file_name : str
            Glove embedding file name
        num_workers : int, default None
            Number of workers to use for multiprocessing. Default uses all available cores
        data_root_path : str
            Path to store the processed data or load existing processed data, if needed (depends on
            save_load_data flag)
        save_load_data : bool
            Shall save or load data from the ``data_root_path``
        """
        self._train_para_limit = train_para_limit
        self._train_ques_limit = train_ques_limit
        self._dev_para_limit = dev_para_limit
        self._dev_ques_limit = dev_ques_limit
        self._ans_limit = ans_limit
        self._char_limit = char_limit
        self._emb_file_name = emb_file_name
        self._is_cased_embedding = emb_file_name.startswith('glove.840')
        self._num_workers = num_workers
        self._save_load_data = save_load_data
        self._data_root_path = data_root_path

        self._processed_train_data_file_name = 'train_processed.json'
        self._processed_dev_data_file_name = 'dev_processed.json'
        self._word_vocab_file_name = 'word_vocab.bin'
        self._char_vocab_file_name = 'char_vocab.bin'

    def get_processed_data(self, use_spacy=True, shrink_word_vocab=True, squad_data_root=None):
        """Main method to start data processing

        Parameters
        ----------
        use_spacy : bool, default True
            Shall use Spacy as a tokenizer. If not, uses NLTK
        shrink_word_vocab : bool, default True
            When True, only tokens that have embeddings in the embedding file are remained in the
            word_vocab. Otherwise tokens with no embedding also stay
        squad_data_root : str, default None
            Data path to store downloaded original SQuAD data
        Returns
        -------
        train_json_data : dict
            Train JSON data of SQuAD dataset as is to run official evaluation script
        dev_json_data : dict
            Dev JSON data of SQuAD dataset as is to run official evaluation script
        train_examples : SQuADQADataset
            Processed examples to be used for training
        dev_examples : SQuADQADataset
            Processed examples to be used for evaluation
        word_vocab : Vocab
            Word vocabulary
        char_vocab : Vocab
            Char vocabulary

        """
        if self._save_load_data and self._has_processed_data():
            return self._load_processed_data()

        train_dataset = SQuAD(segment='train', root=squad_data_root) \
            if squad_data_root else SQuAD(segment='train')
        dev_dataset = SQuAD(segment='dev', root=squad_data_root) \
            if squad_data_root else SQuAD(segment='dev')

        with contextlib.closing(mp.Pool(processes=self._num_workers)) as pool:
            train_examples, dev_examples = SQuADDataPipeline._tokenize_data(train_dataset,
                                                                            dev_dataset,
                                                                            use_spacy, pool)
            word_vocab, char_vocab = SQuADDataPipeline._get_vocabs(train_examples, dev_examples,
                                                                   self._emb_file_name,
                                                                   self._is_cased_embedding,
                                                                   shrink_word_vocab,
                                                                   pool)

        filter_provider = SQuADDataFilter(self._train_para_limit,
                                          self._train_ques_limit,
                                          self._ans_limit)
        train_examples = list(filter(filter_provider.filter, train_examples))

        train_featurizer = SQuADDataFeaturizer(word_vocab,
                                               char_vocab,
                                               self._train_para_limit,
                                               self._train_ques_limit,
                                               self._char_limit,
                                               self._is_cased_embedding)

        dev_featuarizer = SQuADDataFeaturizer(word_vocab,
                                              char_vocab,
                                              self._dev_para_limit,
                                              self._dev_ques_limit,
                                              self._char_limit,
                                              self._is_cased_embedding)

        train_examples, dev_examples = SQuADDataPipeline._featurize_data(train_examples,
                                                                         dev_examples,
                                                                         train_featurizer,
                                                                         dev_featuarizer)

        if self._save_load_data:
            self._save_processed_data(train_examples, dev_examples, word_vocab, char_vocab)

        return train_dataset._read_data(), dev_dataset._read_data(), \
               SQuADQADataset(train_examples), SQuADQADataset(dev_examples), word_vocab, char_vocab

    @staticmethod
    def _tokenize_data(train_dataset, dev_dataset, use_spacy, pool):
        """Tokenize incoming paragpraphs and questions in incoming datsets using provided
        tokenizer withing the processes of the provided multiprocessing pool

        Parameters
        ----------
        train_dataset : SQuAD
            training dataset
        dev_dataset : SQuAD
            Dev dataset
        use_spacy : bool
            Use Spacy as a tokenizer. Otherwise uses NLTK
        pool : Pool
            Multiprocessing pool to use for the tokenization

        Returns
        -------
        train_examples : List[dict]
            List of tokenized training examples
        dev_examples : List[dict]
            List of tokenized dev examples
        """
        tokenizer = SQuADDataTokenizer(use_spacy)

        tic = time.time()
        print('Train examples [{}] transformation started.'.format(len(train_dataset)))
        train_examples = list(tqdm.tqdm(tokenizer.run_async(pool, train_dataset),
                                        total=len(train_dataset)))
        print('Train examples transformed [{}/{}] in {:.3f} sec'.format(len(train_examples),
                                                                        len(train_dataset),
                                                                        time.time() - tic))
        tic = time.time()
        print('Dev examples [{}] transformation started.'.format(len(dev_dataset)))
        dev_examples = list(tqdm.tqdm(tokenizer.run_async(pool, dev_dataset),
                                      total=len(dev_dataset)))
        print('Dev examples transformed [{}/{}] in {:.3f} sec'.format(len(dev_examples),
                                                                      len(dev_dataset),
                                                                      time.time() - tic))
        return train_examples, dev_examples

    @staticmethod
    def _featurize_data(train_examples, dev_examples, train_featurizer, dev_featuarizer):
        """Create features from incoming datasets by replacing tokens with indices.

        Parameters
        ----------
        train_examples : List[dict]
            Tokenized train examples
        dev_examples : List[dict]
            Tokenized dev examples
        train_featurizer : SQuADDataFeaturizer
            Parametrized featurizer for training examples
        dev_featuarizer : SQuADDataFeaturizer
            Parametrized featurizer for dev examples

        Returns
        -------
        train_ready : List[Tuple]
            Processed train examples. Each tuple consists of question_id, record_index,
            context_tokens_indices, question_tokens_indices, context_chars_indices,
            question_char_indices, start_token_index_of_the_answer, end_token_index_of_the_answer,
            context, context_tokens_spans
        dev_ready : List[Tuple]
            Processed dev examples. Each tuple consists of question_id, record_index,
            context_tokens_indices, question_tokens_indices, context_chars_indices,
            question_char_indices, start_token_index_of_the_answer, end_token_index_of_the_answer,
            context, context_tokens_spans

        """
        tic = time.time()
        print('Train examples [{}] featurization started.'.format(len(train_examples)))
        train_ready = [train_featurizer.build_features(example)
                       for example in tqdm.tqdm(train_examples, total=len(train_examples))]
        print('Train examples featurized [{}] in {:.3f} sec'.format(len(train_examples),
                                                                    time.time() - tic))
        tic = time.time()
        print('Dev examples [{}] featurization started.'.format(len(dev_examples)))
        dev_ready = [dev_featuarizer.build_features(example)
                     for example in tqdm.tqdm(dev_examples, total=len(dev_examples))]
        print('Dev examples featurized [{}] in {:.3f} sec'.format(len(dev_examples),
                                                                  time.time() - tic))
        return train_ready, dev_ready

    @staticmethod
    def _get_vocabs(train_examples, dev_examples, emb_file_name, is_cased_embedding,
                    shrink_word_vocab, pool):
        """Create both word-level and character-level vocabularies. Vocabularies are built using
        data from both train and dev datasets.

        Parameters
        ----------
        train_examples : List[dict]
            Tokenized training examples
        dev_examples : List[dict]
            Tokenized dev examples
        emb_file_name : str
            Glove embedding file name
        is_cased_embedding : bool
            When True, provided embedding file is cased, uncased otherwise
        shrink_word_vocab : bool
            When True, only tokens that have embeddings in the embedding file are remained in the
            word_vocab. Otherwise tokens with no embedding also stay
        pool : Pool
            Multiprocessing pool to use

        Returns
        -------
        word_vocab : Vocab
            Word-level vocabulary
        char_vocab : Vocab
            Char-level vocabulary
        """
        tic = time.time()
        print('Word counters receiving started.')

        word_mapper = SQuADAsyncVocabMapper()
        word_reducer = SQuADAsyncVocabReducer()
        word_mapped = list(
            tqdm.tqdm(word_mapper.run_async(itertools.chain(train_examples, dev_examples), pool),
                      total=len(train_examples) + len(dev_examples)))
        word_partitioned = tqdm.tqdm(SQuADDataPipeline._partition(itertools.chain(*word_mapped)),
                                     total=len(word_mapped))
        word_counts = list(tqdm.tqdm(word_reducer.run_async(word_partitioned, pool),
                                     total=len(word_partitioned)))
        print('Word counters received in {:.3f} sec'.format(time.time() - tic))

        tic = time.time()
        print('Char counters receiving started.')
        char_mapper = SQuADAsyncVocabMapper(iterate_over_example=True)
        char_reducer = SQuADAsyncVocabReducer()
        char_mapped = list(
            tqdm.tqdm(char_mapper.run_async(itertools.chain(train_examples, dev_examples), pool),
                      total=len(train_examples) + len(dev_examples)))
        char_partitioned = SQuADDataPipeline._partition(itertools.chain(*char_mapped))
        char_counts = list(tqdm.tqdm(char_reducer.run_async(char_partitioned, pool),
                                     total=len(char_partitioned)))
        print('Char counters received in {:.3f} sec'.format(time.time() - tic))

        embedding = nlp.embedding.create('glove', source=emb_file_name)

        if is_cased_embedding:
            word_counts = itertools.chain(*[[(item[0], item[1]),
                                             (item[0].lower(), item[1]),
                                             (item[0].capitalize(), item[1]),
                                             (item[0].upper(), item[1])] for item in word_counts])
        else:
            word_counts = [(item[0].lower(), item[1]) for item in word_counts]

        word_vocab = Vocab({item[0]: item[1] for item in word_counts if
                            not shrink_word_vocab or item[0] in embedding.token_to_idx},
                           bos_token=None, eos_token=None)
        word_vocab.set_embedding(embedding)
        char_vocab = Vocab({item[0]: item[1] for item in char_counts},
                           bos_token=None, eos_token=None)

        return word_vocab, char_vocab

    def _has_processed_data(self):
        """Check if the data was processed and stored already

        Returns
        -------
        ret: Boolean
            Is processed data already exists
        """
        return \
            os.path.exists(
                os.path.join(self._data_root_path, self._processed_train_data_file_name)) and \
            os.path.exists(
                os.path.join(self._data_root_path, self._processed_dev_data_file_name)) and \
            os.path.exists(
                os.path.join(self._data_root_path, self._word_vocab_file_name)) and \
            os.path.exists(
                os.path.join(self._data_root_path, self._char_vocab_file_name))

    def _load_processed_data(self):
        """ Load processed data from the disk
        Returns
        -------
        train_examples : List[Tuple]
            Processed train examples. Each tuple consists of question_id, record_index,
            context_tokens_indices, question_tokens_indices, context_chars_indices,
            question_char_indices, start_token_index_of_the_answer, end_token_index_of_the_answer,
            context, context_tokens_spans
        dev_examples : List[Tuple]
            Processed dev examples. Each tuple consists of question_id, record_index,
            context_tokens_indices, question_tokens_indices, context_chars_indices,
            question_char_indices, start_token_index_of_the_answer, end_token_index_of_the_answer,
            context, context_tokens_spans
        word_vocab : Vocab
            Word-level vocabulary
        char_vocab : Vocab
            Char-level vocabulary
        """
        with open(os.path.join(self._data_root_path, self._processed_train_data_file_name),
                  'r') as f:
            train_examples = json.load(f)

        with open(os.path.join(self._data_root_path, self._processed_dev_data_file_name), 'r') as f:
            dev_examples = json.load(f)

        with open(os.path.join(self._data_root_path, self._word_vocab_file_name), 'r') as f:
            word_vocab = Vocab.from_json(json.load(f))

        with open(os.path.join(self._data_root_path, self._char_vocab_file_name), 'r') as f:
            char_vocab = Vocab.from_json(json.load(f))

        return train_examples, dev_examples, word_vocab, char_vocab

    def _save_processed_data(self, train_examples, dev_examples, word_vocab, char_vocab):
        """Save processed data to disk

        Parameters
        ----------
        train_examples : List[Tuple]
            Processed train examples. Each tuple consists of question_id, record_index,
            context_tokens_indices, question_tokens_indices, context_chars_indices,
            question_char_indices, start_token_index_of_the_answer, end_token_index_of_the_answer,
            context, context_tokens_spans
        dev_examples : List[Tuple]
            Processed dev examples. Each tuple consists of question_id, record_index,
            context_tokens_indices, question_tokens_indices, context_chars_indices,
            question_char_indices, start_token_index_of_the_answer, end_token_index_of_the_answer,
            context, context_tokens_spans
        word_vocab : Vocab
            Word-level vocabulary
        char_vocab : Vocab
            Char-level vocabulary
        """
        with open(os.path.join(self._data_root_path, self._processed_train_data_file_name),
                  'w') as f:
            json.dump(train_examples, f)

        with open(os.path.join(self._data_root_path, self._processed_dev_data_file_name), 'w') as f:
            json.dump(dev_examples, f)

        with open(os.path.join(self._data_root_path, self._word_vocab_file_name), 'w') as f:
            f.write(word_vocab.to_json())

        with open(os.path.join(self._data_root_path, self._char_vocab_file_name), 'w') as f:
            f.write(char_vocab.to_json())

    @staticmethod
    def _partition(mapped_values):
        """Groups items with same keys into a single partition

        Parameters
        ----------
        mapped_values : List[Tuple]
            List of mapped (key, value) tuples

        Returns
        -------
        items: List[Tuple]
            List of partitions, where each partition is (key, List[value])
        """
        partitioned_data = collections.defaultdict(list)

        for key, value in mapped_values:
            partitioned_data[key].append(value)

        return partitioned_data.items()


class SQuADDataTokenizer:
    """SQuAD data tokenizer, that encapsulate the splitting logic of each entry of SQuAD dataset"""
    spacy_tokenizer = nlp.data.SpacyTokenizer()

    def __init__(self, use_spacy=True):
        """Init new SQuADDataTokenizer object
        Parameters
        ----------
        use_spacy : bool, default True
            Use Spacy as base tokenizer. Otherwise uses NLTK with some cleansing
        """
        self._use_spacy = use_spacy

    def run_async(self, pool, dataset):
        return pool.imap(self, dataset)

    def __call__(self, example):
        return self.tokenize_one_example(example)

    def tokenize_one_example(self, example):
        """Tokenize a single example

        Parameters
        ----------
        example : Tuple
            A tuple of SQuAD dataset in format (record_index, question_id, question, context,
            answer_list, answer_start)

        Returns
        -------
        ret : dict
            Tokenized example with the following keys: context_tokens, context_chars, ques_tokens,
            ques_chars, y1s, y2s, id, context, spans, record_idx
        """
        index, q_id, question, context, answer_list, answer_start = example

        context = context.replace('\'\'', '\" ').replace(r'``', '\" ')
        context_tokens = SQuADDataTokenizer._word_tokenize_spacy(context) if self._use_spacy else \
            SQuADDataTokenizer._word_tokenize_nltk(context)
        context_chars = [list(token) for token in context_tokens]
        spans = SQuADDataTokenizer._get_token_spans(context, context_tokens)

        ques = question.replace('\'\'', '\" ').replace('``', '\" ')
        ques_tokens = SQuADDataTokenizer._word_tokenize_spacy(ques) if self._use_spacy else \
            SQuADDataTokenizer._word_tokenize_nltk(ques)
        ques_chars = [list(token) for token in ques_tokens]

        y1s, y2s = [], []
        answer_texts = []

        for answer_text, answer_start in zip(answer_list, answer_start):
            answer_end = answer_start + len(answer_text)
            answer_texts.append(answer_text)
            answer_span = []
            for idx, span in enumerate(spans):
                if not (answer_end <= span[0] or answer_start >= span[1]):
                    answer_span.append(idx)
            y1, y2 = answer_span[0], answer_span[-1]
            y1s.append(y1)
            y2s.append(y2)

        result = {'context_tokens': context_tokens, 'context_chars': context_chars,
                  'ques_tokens': ques_tokens, 'ques_chars': ques_chars, 'y1s': y1s,
                  'y2s': y2s, 'id': q_id, 'context': context, 'spans': spans, 'record_idx': index}
        return result

    @staticmethod
    def _word_tokenize_spacy(sent):
        """Default tokenization method that uses Spacy. Called only if not overridden by providing
        base_tokenizer to SQuADDataTokenizer.__init__

        Parameters
        ----------
        sent : str
            A text to tokenize

        Returns
        -------
        tokens : List[str]
            List of tokens
        """
        tokens = SQuADDataTokenizer.spacy_tokenizer(sent)
        return tokens

    @staticmethod
    def _word_tokenize_nltk(sent):
        """Tokenization method that uses NLTK.

        Parameters
        ----------
        sent : str
            A text to tokenize

        Returns
        -------
        tokens : List[str]
            List of tokens
        """
        tokens = []
        splitters = ('-', '\u2212', '\u2014', '\u2013', '/', '~', '"', '\'', '\u201C',
                     '\u2019', '\u201D', '\u2018', '\u00B0')

        sample = sent.replace('\n', ' ').replace(u'\u000A', '').replace(u'\u00A0', '')
        temp_tokens = [token.replace('\'\'', '"').replace('``', '"') for token in
                       nltk.word_tokenize(sample)]

        for token in temp_tokens:
            tokens.extend(re.split('([{}])'.format(''.join(splitters)), token))

        tokens = [token for token in tokens if len(token) > 0]
        return tokens

    @staticmethod
    def _get_token_spans(text, tokens):
        """Create a list of tuples that contains tokens character inidices. By using this output
        it is possible to find character-based indices of token start and end

        Parameters
        ----------
        text : str
            Original text
        tokens : List[str]
            List of tokens of the original text

        Returns
        -------
        ret: List[Tuple]
            List of tuple, where each tuple contains starting character index of the token in the
            text and end character index of the token in the text
        """
        current = 0
        spans = []
        for token in tokens:
            current = text.find(token, current)
            if current < 0:
                print('Token {} cannot be found'.format(token))
                raise Exception()
            spans.append((current, current + len(token)))
            current += len(token)
        return spans


class SQuADDataFilter:
    """Filter an example based on the specified conditions"""

    def __init__(self, para_limit, ques_limit, ans_limit):
        """Init SQuADDataFilter object

        Parameters
        ----------
        para_limit : int
            Maximum allowed length of a paragraph
        ques_limit : int
            Maximum allowed length of a question
        ans_limit : int
            Maximum allowed length of an answer
        """
        self._para_limit = para_limit
        self._ques_limit = ques_limit
        self._ans_limit = ans_limit

    def filter(self, example):
        """Returns if the example should be filtered out or not

        Parameters
        ----------
        example : dict
            A dataset examples with context_tokens, ques_tokens, y1s and y2s keys

        Returns
        -------
        ret : Boolean
            True if an example should remain in the dataset, and False if it should be excluded from
            the dataset
        """
        return len(example['context_tokens']) <= self._para_limit and \
               len(example['ques_tokens']) <= self._ques_limit and \
               (example['y2s'][0] - example['y1s'][0]) <= self._ans_limit


class SQuADAsyncVocabMapper:
    """A multiprocessing implementation of a Mapper for tokens counting"""

    def __init__(self, iterate_over_example=False):
        """Init MapReduce object

        Parameters
        ----------
        iterate_over_example : bool, default False
            Should use examples as is, or iterate over its content
        """
        self._iterate_over_example = iterate_over_example

    def run_async(self, examples, pool):
        """Run async processing over examples

        Parameters
        ----------
        examples : List[dict]
            List of dictionaries with context_tokens and ques_tokens keys
        pool : Pool
            Multiprocessing pool to use

        Returns
        -------
        ret : List[Tuple]
            List of tuples of tokens and counts: (str, int)
        """
        return pool.imap(self, examples)

    def __call__(self, example):
        """Maps examples into distinct tokens

        Parameters
        ----------
        example : dict
            Example to process with context_tokens and ques_tokens keys

        Returns
        -------
        mapped_values : List[Tuple]
            Result of mapping process. Each tuple of (token, count) format
        """
        para_counter = data.count_tokens(example['context_tokens'] if not self._iterate_over_example
                                         else [c for tkn in example['context_tokens'] for c in tkn])
        ques_counter = data.count_tokens(example['ques_tokens'] if not self._iterate_over_example
                                         else [c for tkn in example['ques_tokens'] for c in tkn])
        counter = para_counter + ques_counter
        return list(counter.items())


class SQuADAsyncVocabReducer:
    """A multiprocessing implementation of a Reducing for tokens counting"""

    def run_async(self, items, pool):
        """Run async processing over examples

        Parameters
        ----------
        items : List[Tuple]
            List of tuples of (token, count) structure
        pool : Pool
            Multiprocessing pool to use

        Returns
        -------
        ret : List[Tuple]
            List of tuples of tokens and counts: (str, int)
        """
        return pool.imap(self, items)

    def __call__(self, item):
        """Sums up number of times a token was used

        Parameters
        ----------
        item : Tuple
            A tuple of (token, counts) format

        Returns
        -------
        ret : Tuple
            A tuple of (token, sum_of_counts)

        """
        token, counts = item
        return token, sum(counts)


class SQuADDataFeaturizer:
    """Class that converts tokenized examples into featurized"""

    def __init__(self, word_vocab, char_vocab, para_limit, ques_limit, char_limit,
                 is_cased_embedding):
        """Init SQuADDataFeaturizer object

        Parameters
        ----------
        word_vocab : Vocab
            Word-level vocabulary
        char_vocab : Vocab
            Char-level vocabulary
        para_limit : int
            Maximum characters in a paragraph
        ques_limit : int
            Maximum characters in a question
        char_limit : int
            Maximum characters in a token
        is_cased_embedding: bool
            Is underlying embedding is cased or uncased
        """
        self._para_limit = para_limit
        self._ques_limit = ques_limit
        self._char_limit = char_limit

        self._word_vocab = word_vocab
        self._char_vocab = char_vocab

        self._is_cased_embedding = is_cased_embedding

    def _get_words_emb(self, words):
        """Get embedding for the words

        Parameters
        ----------
        words : list[str]
            Words to embed

        Returns
        -------
        ret : np.array
            Array of embeddings for words
        """

        if not self._is_cased_embedding:
            return self._word_vocab[[word.lower() for word in words]]

        result = np.full([len(words)], fill_value=0, dtype=np.float32)
        word_emb_matrix = np.full([len(words), 4], fill_value=0, dtype=np.float32)

        for i, w in enumerate(words):
            word_emb_matrix[i, :] = self._word_vocab[[w, w.lower(), w.capitalize(), w.upper()]]

        mask = word_emb_matrix != 0
        first_non_zero_embeddings_indices = np.where(mask.any(axis=1), mask.argmax(axis=1), -1)

        for i, index in enumerate(first_non_zero_embeddings_indices):
            result[i] = word_emb_matrix[i, index]

        return result

    def build_features(self, example):
        """Generate features for a given example

        Parameters
        ----------
        example : dict
            A tokenized example of a dataset

        Returns
        -------
        ret : Tuple
            An example with tokens replaced with indices of the following format: question_id,
            record_index, context_tokens_indices, question_tokens_indices, context_chars_indices,
            question_char_indices, start_token_index_of_the_answer, end_token_index_of_the_answer,
            context, context_tokens_spans
        """
        context_idxs = np.full([self._para_limit],
                               fill_value=self._word_vocab[self._word_vocab.padding_token],
                               dtype=np.float32)

        ctx_chars_idxs = np.full([self._para_limit, self._char_limit],
                                 fill_value=self._char_vocab[self._char_vocab.padding_token],
                                 dtype=np.float32)

        ques_idxs = np.full([self._ques_limit],
                            fill_value=self._word_vocab[self._word_vocab.padding_token],
                            dtype=np.float32)

        ques_char_idxs = np.full([self._ques_limit, self._char_limit],
                                 fill_value=self._char_vocab[self._char_vocab.padding_token],
                                 dtype=np.float32)

        context_len = min(len(example['context_tokens']), self._para_limit)
        context_idxs[:context_len] = self._get_words_emb(example['context_tokens'][:context_len])

        ques_len = min(len(example['ques_tokens']), self._ques_limit)
        ques_idxs[:ques_len] = self._get_words_emb(example['ques_tokens'][:ques_len])

        for i in range(0, context_len):
            char_len = min(len(example['context_chars'][i]), self._char_limit)
            ctx_chars_idxs[i, :char_len] = self._char_vocab[example['context_chars'][i][:char_len]]

        for i in range(0, ques_len):
            char_len = min(len(example['ques_chars'][i]), self._char_limit)
            ques_char_idxs[i, :char_len] = self._char_vocab[example['ques_tokens'][i][:char_len]]

        start, end = example['y1s'][-1], example['y2s'][-1]

        record = (example['id'],
                  example['record_idx'],
                  context_idxs,
                  ques_idxs,
                  ctx_chars_idxs,
                  ques_char_idxs,
                  start,
                  end,
                  example['context'],
                  example['spans'])

        return record


class SQuADQADataset(Dataset):
    """Dataset that wraps the featurized examples with standard Gluon API Dataset format. It allows
    to fetch a record by question id for the evaluation"""

    def __init__(self, records):
        super(SQuADQADataset, self).__init__()
        self._data = records
        self._record_idx_to_record = {}

        for record in records:
            self._record_idx_to_record[record[1]] = {'q_id': record[0], 'rec': record}

    def __getitem__(self, idx):
        """Get example by index in the original list

        Parameters
        ----------
        idx : int

        Returns
        -------
        ret : Tuple of question_id, record_index, context_tokens_indices, question_tokens_indices,
            context_chars_indices, question_char_indices, start_token_index_of_the_answer,
            end_token_index_of_the_answer, context, context_tokens_spans
        """
        return self._data[idx]

    def __len__(self):
        """Get the number of the examples in the dataset

        Returns
        -------
        ret : int
            Number of examples of the dataset
        """
        return len(self._data)

    def get_q_id_by_rec_idx(self, rec_idx):
        """Returns a question id associated with provided record index from original SQuAD dataset

        Parameters
        ----------
        rec_idx : int
            Record index in SQuAD dataset

        Returns
        -------
        question_id : str
        """
        return self._record_idx_to_record[rec_idx]['q_id']

    def get_record_by_idx(self, rec_idx):
        """Returns a record associated with provided record index from original SQuAD dataset

        Parameters
        ----------
        rec_idx : int

        Returns
        -------
        ret : Tuple of question_id, record_index, context_tokens_indices, question_tokens_indices,
            context_chars_indices, question_char_indices, start_token_index_of_the_answer,
            end_token_index_of_the_answer, context, context_tokens_spans
        """
        return self._record_idx_to_record[rec_idx]['rec']


class SQuADDataLoaderTransformer:
    """Thin wrapper on SQuADQADataset that removed non-numeric values from the record. The output of
    that transformer can be provided to a DataLoader"""

    def __call__(self, q_id, record_idx, ctx_idxs, ques_idxs, ctx_chars_idxs, ques_char_idxs,
                 start, end, context, spans):
        """Return the same record with non-numeric values removed from the output

        Parameters
        ----------
        q_id : str
            Question Id
        record_idx : int
            Record index
        ctx_idxs : NDArray
            Indices of context tokens
        ques_idxs : NDArray
            Indices of question tokens
        ctx_chars_idxs : NDArray
            Indices of context characters
        ques_char_idxs : NDArray
            Indices of question characters
        start : int
            Start of the answer
        end : int
            End of the answer
        context : str
            Original context string
        spans : List[Tuple]
            List of character indices of each token of the context.

        Returns
        -------
        record_idx : int
            Record index
        ctx_idxs : NDArray
            Indices of context tokens
        ques_idxs : NDArray
            Indices of question tokens
        ctx_chars_idxs : NDArray
            Indices of context characters
        ques_char_idxs : NDArray
            Indices of question characters
        start : int
            Start of the answer
        end : int
            End of the answer
        """
        return record_idx, ctx_idxs, ques_idxs, ctx_chars_idxs, ques_char_idxs, start, end
