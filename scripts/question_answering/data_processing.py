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
"""SQuAD data preprocessing."""

__all__ = ['SQuADTransform', 'VocabProvider']

import pickle
from os.path import isfile
import numpy as np
from mxnet import nd

import gluonnlp as nlp
from gluonnlp import Vocab, data
from gluonnlp.data.batchify import Pad
from .tokenizer import BiDAFTokenizer


class SQuADTransform(object):
    """SQuADTransform class responsible for converting text data into NDArrays that can be later
    feed into DataProvider
    """

    def __init__(self, vocab_provider, question_max_length, context_max_length,
                 max_chars_per_word, embedding_size):
        self._word_vocab = vocab_provider.get_word_level_vocab(embedding_size)
        self._char_vocab = vocab_provider.get_char_level_vocab()
        self._tokenizer = vocab_provider.get_tokenizer()

        self._question_max_length = question_max_length
        self._context_max_length = context_max_length
        self._max_chars_per_word = max_chars_per_word

        self._padder = Pad()

    def __call__(self, record_index, question_id, question, context, answer_list,
                 answer_start_list):
        """
        Method converts text into numeric arrays based on Vocabulary.
        Answers are not processed, as they are not needed in input

        Parameters
        ----------

        record_index: int
            Index of the record
        question_id: str
            Question Id
        question: str
            Question
        context: str
            Context
        answer_list: list[str]
            List of answers
        answer_start_list: list[int]
            List of start indices of answers

        Returns
        -------
        record: Tuple
            A tuple containing record_index [int], question_id [str], question_words_nd [NDArray],
            context_words_nd [NDArray], question_chars_nd [NDArray[, context_chars_nd [NDArray],
            answer_spans [list[Tuple]]
        """
        question_tokens = self._tokenizer(question, lower_case=True)
        context_tokens = self._tokenizer(context, lower_case=True)

        question_words = self._word_vocab[question_tokens[:self._question_max_length]]
        context_words = self._word_vocab[context_tokens[:self._context_max_length]]

        question_chars = [self._char_vocab[[character.lower() for character in word]]
                          for word in question_tokens[:self._question_max_length]]

        context_chars = [self._char_vocab[[character.lower() for character in word]]
                         for word in context_tokens[:self._context_max_length]]

        question_words_nd = self._pad_to_max_word_length(question_words, self._question_max_length)
        question_chars_nd = self._padder(question_chars)
        question_chars_nd = self._pad_to_max_char_length(question_chars_nd,
                                                         self._question_max_length)

        context_words_nd = self._pad_to_max_word_length(context_words, self._context_max_length)
        context_chars_nd = self._padder(context_chars)
        context_chars_nd = self._pad_to_max_char_length(context_chars_nd, self._context_max_length)

        answer_spans = SQuADTransform._get_answer_spans(context, context_tokens, answer_list,
                                                        answer_start_list)

        return (record_index, question_id, question_words_nd, context_words_nd,
                question_chars_nd, context_chars_nd, answer_spans)

    @staticmethod
    def _get_answer_spans(context, context_tokens, answer_list, answer_start_list):
        """Find all answer spans from the context, returning start_index and end_index.
        Each index is a index of a token

        Parameters
        ----------
        context: str
            Context
        context_tokens: list[str]
            Tokenized context
        answer_list: list[str]
            List of answers
        answer_start_list: list[int]
            List of answers start indices

        Returns
        -------
        answer_spans: List[Tuple]
            List of Tuple(answer_start_index answer_end_index) per question
        """
        answer_spans = []
        # SQuAD answers doesn't always match to used tokens in the context. Sometimes there is only
        # a partial match. We use the same method as used in original implementation:
        # 1. Find char index range for all tokens of context
        # 2. Foreach answer
        #   2.1 Find char index range for the answer (not tokenized)
        #   2.2 Find Context token indices which char indices contains answer char indices
        #   2.3. Return first and last token indices
        context_char_indices = SQuADTransform.get_char_indices(context, context_tokens)

        for answer_start_char_index, answer in zip(answer_start_list, answer_list):
            answer_token_indices = []
            answer_end_char_index = answer_start_char_index + len(answer)

            for context_token_index, context_char_span in enumerate(context_char_indices):
                if not (answer_end_char_index <= context_char_span[0] or
                        answer_start_char_index >= context_char_span[1]):
                    answer_token_indices.append(context_token_index)

            if len(answer_token_indices) == 0:
                print('Warning: Answer {} not found for context {}'.format(answer, context))
            else:
                answer_span = (answer_token_indices[0],
                               answer_token_indices[len(answer_token_indices) - 1])
                answer_spans.append(answer_span)

        if len(answer_spans) == 0:
            print('Warning: No answers found for context {}'.format(context_tokens))

        return answer_spans

    @staticmethod
    def get_char_indices(text, text_tokens):
        """Match token with character indices

        Parameters
        ----------
        text: str
            Text
        text_tokens: list[str]
            Tokens of the text

        Returns
        -------
        char_indices_per_token: List[Tuple]
            List of (start_index, end_index) of characters where the position equals to token index
        """
        char_indices_per_token = []
        current_index = 0
        text_lowered = text.lower()

        for token in text_tokens:
            current_index = text_lowered.find(token, current_index)
            char_indices_per_token.append((current_index, current_index + len(token)))
            current_index += len(token)

        return char_indices_per_token

    def _pad_to_max_char_length(self, item, max_item_length):
        """Pads all tokens to maximum size

        Parameters
        ----------
        item: NDArray
            Matrix of indices
        max_item_length: int
            Maximum length of a token

        Returns
        -------
        NDArray
            Padded NDArray
        """
        # expand dimensions to 4 and turn to float32, because nd.pad can work only with 4 dims
        data_expanded = item.reshape(1, 1, item.shape[0], item.shape[1]).astype(np.float32)
        data_padded = nd.pad(data_expanded,
                             mode='constant',
                             pad_width=[0, 0, 0, 0, 0, max_item_length - item.shape[0],
                                        0, self._max_chars_per_word - item.shape[1]],
                             constant_value=0)

        # reshape back to original dimensions with the last dimension of max_item_length
        # We also convert to float32 because it will be necessary later for processing
        data_reshaped_back = data_padded.reshape(max_item_length,
                                                 self._max_chars_per_word).astype(np.float32)
        return data_reshaped_back

    @staticmethod
    def _pad_to_max_word_length(item, max_length):
        """Pads sentences to maximum length

        Parameters
        ----------
        item: NDArray
            Vector of words
        max_length: int
            Maximum length of question/context

        Returns
        -------
        NDArray
            Padded vector of tokens
        """
        data_nd = nd.array(item, dtype=np.float32)
        # expand dimensions to 4 and turn to float32, because nd.pad can work only with 4 dims
        data_expanded = data_nd.reshape(1, 1, 1, data_nd.shape[0])
        data_padded = nd.pad(data_expanded,
                             mode='constant',
                             pad_width=[0, 0, 0, 0, 0, 0, 0, max_length - data_nd.shape[0]],
                             constant_value=0)
        # reshape back to original dimensions with the last dimension of max_length
        # We also convert to float32 because it will be necessary later for processing
        data_reshaped_back = data_padded.reshape(max_length).astype(np.float32)
        return data_reshaped_back


class VocabProvider(object):
    """Provides word level and character level vocabularies
    """

    def __init__(self, datasets, options, tokenizer=BiDAFTokenizer()):
        self._datasets = datasets
        self._options = options
        self._tokenizer = tokenizer

    def get_tokenizer(self):
        """Provides tokenizer used to create vocab

        Returns
        -------
        tokenizer: Tokenizer
            Tokenizer

        """
        return self._tokenizer

    def get_char_level_vocab(self):
        """Provides character level vocabulary

        Returns
        -------
        Vocab
            Character level vocabulary
        """
        if self._options.char_vocab_path and isfile(self._options.char_vocab_path):
            return pickle.load(open(self._options.char_vocab_path, 'rb'))

        all_chars = []
        for dataset in self._datasets:
            all_chars.extend(self._get_all_char_tokens(dataset))

        char_level_vocab = VocabProvider._create_squad_vocab(all_chars)

        if self._options.char_vocab_path:
            pickle.dump(char_level_vocab, open(self._options.char_vocab_path, 'wb'))

        return char_level_vocab

    def get_word_level_vocab(self, embedding_size):
        """Provides word level vocabulary

        Returns
        -------
        Vocab
            Word level vocabulary
        """

        if self._options.word_vocab_path and isfile(self._options.word_vocab_path):
            return pickle.load(open(self._options.word_vocab_path, 'rb'))

        all_words = []
        for dataset in self._datasets:
            all_words.extend(self._get_all_word_tokens(dataset))

        word_level_vocab = VocabProvider._create_squad_vocab(all_words)
        word_level_vocab.set_embedding(
            nlp.embedding.create('glove', source='glove.6B.{}d'.format(embedding_size)))

        count = 0

        for i in range(len(word_level_vocab)):
            if (word_level_vocab.embedding.idx_to_vec[i].sum() != 0).asscalar():
                count += 1

        print('{}/{} words have embeddings'.format(count, len(word_level_vocab)))

        if self._options.word_vocab_path:
            pickle.dump(word_level_vocab, open(self._options.word_vocab_path, 'wb'))

        return word_level_vocab

    @staticmethod
    def _create_squad_vocab(all_tokens):
        """Provides vocabulary based on list of tokens

        Parameters
        ----------

        all_tokens: List[str]
            List of all tokens

        Returns
        -------
        Vocab
            Vocabulary
        """
        counter = data.count_tokens(all_tokens)
        vocab = Vocab(counter)
        return vocab

    def _get_all_word_tokens(self, dataset):
        """Provides all words from context and question of the dataset

        Parameters
        ----------

        dataset: SimpleDataset
            Dataset of SQuAD

        Returns
        -------
        all_tokens: list[str]
            List of all words
        """
        all_tokens = []

        for data_item in dataset:
            all_tokens.extend(self._tokenizer(data_item[2], lower_case=True))
            all_tokens.extend(self._tokenizer(data_item[3], lower_case=True))

        return all_tokens

    def _get_all_char_tokens(self, dataset):
        """Provides all characters from context and question of the dataset

        Parameters
        ----------

        dataset: SimpleDataset
            Dataset of SQuAD

        Returns
        -------
        all_tokens: list[str]
            List of all characters
        """
        all_tokens = []

        for data_item in dataset:
            for character in data_item[2]:
                all_tokens.extend(character.lower())

            for character in data_item[3]:
                all_tokens.extend(character.lower())

        return all_tokens
