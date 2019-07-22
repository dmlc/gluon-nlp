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
"""Data utilities for the named entity recognition task."""

import logging
from collections import namedtuple

import numpy as np
import mxnet as mx
import gluonnlp as nlp

TaggedToken = namedtuple('TaggedToken', ['text', 'tag'])
PredictedToken = namedtuple('PredictedToken', ['text', 'true_tag', 'pred_tag'])

NULL_TAG = 'X'

def bio_bioes(tokens):
    """Convert a list of TaggedTokens in BIO(2) scheme to BIOES scheme.

    Parameters
    ----------
    tokens: List[TaggedToken]
        A list of tokens in BIO(2) scheme

    Returns
    -------
    List[TaggedToken]:
        A list of tokens in BIOES scheme
    """
    ret = []
    for index, token in enumerate(tokens):
        if token.tag == 'O':
            ret.append(token)
        elif token.tag.startswith('B'):
            # if a B-tag is continued by other tokens with the same entity,
            # then it is still a B-tag
            if index + 1 < len(tokens) and tokens[index + 1].tag.startswith('I'):
                ret.append(token)
            else:
                ret.append(TaggedToken(text=token.text, tag='S' + token.tag[1:]))
        elif token.tag.startswith('I'):
            # if an I-tag is continued by other tokens with the same entity,
            # then it is still an I-tag
            if index + 1 < len(tokens) and tokens[index + 1].tag.startswith('I'):
                ret.append(token)
            else:
                ret.append(TaggedToken(text=token.text, tag='E' + token.tag[1:]))
    return ret


def read_bio_as_bio2(data_path):
    """Read CoNLL-formatted text file in BIO scheme in given path as sentences in BIO2 scheme.

    Parameters
    ----------
    data_path: str
        Path of the data file to read

    Returns
    -------
    List[List[TaggedToken]]:
        List of sentences, each of which is a List of TaggedTokens
    """

    with open(data_path, 'r') as ifp:
        sentence_list = []
        current_sentence = []
        prev_tag = 'O'

        for line in ifp:
            if len(line.strip()) > 0:
                word, _, _, tag = line.rstrip().split(' ')
                # convert BIO tag to BIO2 tag
                if tag == 'O':
                    bio2_tag = 'O'
                else:
                    if prev_tag == 'O' or tag[2:] != prev_tag[2:]:
                        bio2_tag = 'B' + tag[1:]
                    else:
                        bio2_tag = tag
                current_sentence.append(TaggedToken(text=word, tag=bio2_tag))
                prev_tag = tag
            else:
                # the sentence was completed if an empty line occurred; flush the current sentence.
                sentence_list.append(current_sentence)
                current_sentence = []
                prev_tag = 'O'

        # check if there is a remaining token. in most CoNLL data files, this does not happen.
        if len(current_sentence) > 0:
            sentence_list.append(current_sentence)
        return sentence_list


def remove_docstart_sentence(sentences):
    """Remove -DOCSTART- sentences in the list of sentences.

    Parameters
    ----------
    sentences: List[List[TaggedToken]]
        List of sentences, each of which is a List of TaggedTokens.
        This list may contain DOCSTART sentences.

    Returns
    -------
        List of sentences, each of which is a List of TaggedTokens.
        This list does not contain DOCSTART sentences.
    """
    ret = []
    for sentence in sentences:
        current_sentence = []
        for token in sentence:
            if token.text != '-DOCSTART-':
                current_sentence.append(token)
        if len(current_sentence) > 0:
            ret.append(current_sentence)
    return ret


def bert_tokenize_sentence(sentence, bert_tokenizer):
    """Apply BERT tokenizer on a tagged sentence to break words into sub-words.
    This function assumes input tags are following IOBES, and outputs IOBES tags.

    Parameters
    ----------
    sentence: List[TaggedToken]
        List of tagged words
    bert_tokenizer: nlp.data.BertTokenizer
        BERT tokenizer

    Returns
    -------
    List[TaggedToken]: list of annotated sub-word tokens
    """
    ret = []
    for token in sentence:
        # break a word into sub-word tokens
        sub_token_texts = bert_tokenizer(token.text)
        # only the first token of a word is going to be tagged
        ret.append(TaggedToken(text=sub_token_texts[0], tag=token.tag))
        ret += [TaggedToken(text=sub_token_text, tag=NULL_TAG)
                for sub_token_text in sub_token_texts[1:]]

    return ret


def load_segment(file_path, bert_tokenizer):
    """Load CoNLL format NER datafile with BIO-scheme tags.

    Tagging scheme is converted into BIOES, and words are tokenized into wordpieces
    using `bert_tokenizer`.

    Parameters
    ----------
    file_path: str
        Path of the file
    bert_tokenizer: nlp.data.BERTTokenizer

    Returns
    -------
    List[List[TaggedToken]]: List of sentences, each of which is the list of `TaggedToken`s.
    """
    logging.info('Loading sentences in %s...', file_path)
    bio2_sentences = remove_docstart_sentence(read_bio_as_bio2(file_path))
    bioes_sentences = [bio_bioes(sentence) for sentence in bio2_sentences]
    subword_sentences = [bert_tokenize_sentence(sentence, bert_tokenizer)
                         for sentence in bioes_sentences]

    logging.info('load %s, its max seq len: %d',
                 file_path, max(len(sentence) for sentence in subword_sentences))

    return subword_sentences


class BERTTaggingDataset:
    """

    Parameters
    ----------
    text_vocab: gluon.nlp.Vocab
        Vocabulary of text tokens/
    train_path: Optional[str]
        Path of the file to locate training data.
    dev_path: Optional[str]
        Path of the file to locate development data.
    test_path: Optional[str]
        Path of the file to locate test data.
    seq_len: int
        Length of the input sequence to BERT.
    is_cased: bool
        Whether to use cased model.
    """

    def __init__(self, text_vocab, train_path, dev_path, test_path, seq_len, is_cased,
                 tag_vocab=None):
        self.text_vocab = text_vocab
        self.seq_len = seq_len

        self.bert_tokenizer = nlp.data.BERTTokenizer(vocab=text_vocab, lower=not is_cased)

        train_sentences = [] if train_path is None else load_segment(train_path,
                                                                     self.bert_tokenizer)
        dev_sentences = [] if dev_path is None else load_segment(dev_path, self.bert_tokenizer)
        test_sentences = [] if test_path is None else load_segment(test_path, self.bert_tokenizer)
        all_sentences = train_sentences + dev_sentences + test_sentences

        if tag_vocab is None:
            logging.info('Indexing tags...')
            tag_counter = nlp.data.count_tokens(token.tag
                                                for sentence in all_sentences for token in sentence)
            self.tag_vocab = nlp.Vocab(tag_counter, padding_token=NULL_TAG,
                                       bos_token=None, eos_token=None, unknown_token=None)
        else:
            self.tag_vocab = tag_vocab
        self.null_tag_index = self.tag_vocab[NULL_TAG]

        if len(test_sentences) > 0:
            logging.info('example test sentences:')
            for i in range(10):
                logging.info(str(test_sentences[i]))

        self.train_inputs = [self._encode_as_input(sentence) for sentence in train_sentences]
        self.dev_inputs = [self._encode_as_input(sentence) for sentence in dev_sentences]
        self.test_inputs = [self._encode_as_input(sentence) for sentence in test_sentences]

        logging.info('tag_vocab: %s', self.tag_vocab)

    def _encode_as_input(self, sentence):
        """Enocde a single sentence into numpy arrays as input to the BERTTagger model.

        Parameters
        ----------
        sentence: List[TaggedToken]
            A sentence as a list of tagged tokens.

        Returns
        -------
        np.array: token text ids (batch_size, seq_len)
        np.array: token types (batch_size, seq_len),
                which is all zero because we have only one sentence for tagging.
        np.array: valid_length (batch_size,) the number of tokens until [SEP] token
        np.array: tag_ids (batch_size, seq_len)
        np.array: flag_nonnull_tag (batch_size, seq_len),
                which is simply tag_ids != self.null_tag_index

        """
        # check whether the given sequence can be fit into `seq_len`.
        assert len(sentence) <= self.seq_len - 2, \
            'the number of tokens {} should not be larger than {} - 2. offending sentence: {}' \
            .format(len(sentence), self.seq_len, sentence)

        text_tokens = ([self.text_vocab.cls_token] + [token.text for token in sentence] +
                       [self.text_vocab.sep_token])
        padded_text_ids = (self.text_vocab.to_indices(text_tokens)
                           + ([self.text_vocab[self.text_vocab.padding_token]]
                              * (self.seq_len - len(text_tokens))))

        tags = [NULL_TAG] + [token.tag for token in sentence] + [NULL_TAG]
        padded_tag_ids = (self.tag_vocab.to_indices(tags)
                          + [self.tag_vocab[NULL_TAG]] * (self.seq_len - len(tags)))

        assert len(text_tokens) == len(tags)
        assert len(padded_text_ids) == len(padded_tag_ids)
        assert len(padded_text_ids) == self.seq_len

        valid_length = len(text_tokens)

        # in sequence tagging problems, only one sentence is given
        token_types = [0] * self.seq_len

        np_tag_ids = np.array(padded_tag_ids, dtype='int32')
        # gluon batchify cannot batchify numpy.bool? :(
        flag_nonnull_tag = (np_tag_ids != self.null_tag_index).astype('int32')

        return (np.array(padded_text_ids, dtype='int32'),
                np.array(token_types, dtype='int32'),
                np.array(valid_length, dtype='int32'),
                np_tag_ids,
                flag_nonnull_tag)

    @staticmethod
    def _get_data_loader(inputs, shuffle, batch_size):
        return mx.gluon.data.DataLoader(inputs, batch_size=batch_size, shuffle=shuffle,
                                        last_batch='keep')

    def get_train_data_loader(self, batch_size):
        return self._get_data_loader(self.train_inputs, shuffle=True, batch_size=batch_size)

    def get_dev_data_loader(self, batch_size):
        return self._get_data_loader(self.dev_inputs, shuffle=False, batch_size=batch_size)

    def get_test_data_loader(self, batch_size):
        return self._get_data_loader(self.test_inputs, shuffle=False, batch_size=batch_size)

    @property
    def num_tag_types(self):
        """Returns the number of unique tags.

        Returns
        -------
        int: number of tag types.
        """
        return len(self.tag_vocab)


def convert_arrays_to_text(text_vocab, tag_vocab,
                           np_text_ids, np_true_tags, np_pred_tags, np_valid_length):
    """Convert numpy array data into text

    Parameters
    ----------
    np_text_ids: token text ids (batch_size, seq_len)
    np_true_tags: tag_ids (batch_size, seq_len)
    np_pred_tags: tag_ids (batch_size, seq_len)
    np.array: valid_length (batch_size,) the number of tokens until [SEP] token

    Returns
    -------
    List[List[PredictedToken]]:

    """
    predictions = []
    for sample_index in range(np_valid_length.shape[0]):
        sample_len = np_valid_length[sample_index]
        entries = []
        for i in range(1, sample_len - 1):
            token_text = text_vocab.idx_to_token[np_text_ids[sample_index, i]]
            true_tag = tag_vocab.idx_to_token[int(np_true_tags[sample_index, i])]
            pred_tag = tag_vocab.idx_to_token[int(np_pred_tags[sample_index, i])]
            # we don't need to predict on NULL tags
            if true_tag == NULL_TAG:
                last_entry = entries[-1]
                entries[-1] = PredictedToken(text=last_entry.text + token_text,
                                             true_tag=last_entry.true_tag,
                                             pred_tag=last_entry.pred_tag)
            else:
                entries.append(PredictedToken(text=token_text,
                                              true_tag=true_tag, pred_tag=pred_tag))

        predictions.append(entries)
    return predictions
