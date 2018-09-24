r"""
This file is taken and modified from R-Net by HKUST-KnowComp
https://github.com/HKUST-KnowComp/R-Net


Make sure that data_process.py and train.json/dev.json/glove.txt files are in the same directory.

And just run `python3 data_process.py` to generate all cached file.

Work flow
---
1. Load train/dev.json and glove file, and build the word2idx/char2idx dict
   and word/char embedding matrix.
2. For each example, encode the word/char represent of context/query, and generate
   the representation of answer span.
3. A simple bucketing with context's length

TODO:
---
replace the data loader/tokenizer/bucket with the gluonnlp existed class.
"""

import json
import random
from collections import Counter

import numpy as np
from tqdm import tqdm

import spacy

nlp = spacy.blank('en')


def word_tokenize(sent):
    r"""
    Tokenize sentence.
    """
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    """
        convert token idx to char idx.
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


def process_file(filename, data_type, word_counter, char_counter):
    r"""
    Process file.
    """
    print('Generating {} examples...'.format(data_type))
    examples = []
    total = 0
    with open(filename, 'r') as fh:
        source = json.load(fh)
        for article in tqdm(source['data']):
            process_one_article(article, examples,
                                word_counter, char_counter, total)
        random.shuffle(examples)
        print('{} questions in total'.format(len(examples)))
    return examples


def process_one_article(article, examples, word_counter, char_counter, total):
    r"""
        Process one article.
    """
    for para in article['paragraphs']:
        context = para['context'].replace(
            '\'\'', '\" ').replace(r'``', '\" ')
        context_tokens = word_tokenize(context)
        context_chars = [list(token) for token in context_tokens]
        spans = convert_idx(context, context_tokens)
        for token in context_tokens:
            word_counter[token] += len(para['qas'])
            for char in token:
                char_counter[char] += len(para['qas'])
        for qa in para['qas']:
            total += 1
            ques = qa['question'].replace(
                '\'\'', '\" ').replace('``', '\" ')
            ques_tokens = word_tokenize(ques)
            ques_chars = [list(token) for token in ques_tokens]
            for token in ques_tokens:
                word_counter[token] += 1
                for char in token:
                    char_counter[char] += 1
            y1s, y2s = [], []
            answer_texts = []
            for answer in qa['answers']:
                answer_text = answer['text']
                answer_start = answer['answer_start']
                answer_end = answer_start + len(answer_text)
                answer_texts.append(answer_text)
                answer_span = []
                for idx, span in enumerate(spans):
                    if not (answer_end <= span[0] or answer_start >= span[1]):
                        answer_span.append(idx)
                y1, y2 = answer_span[0], answer_span[-1]
                y1s.append(y1)
                y2s.append(y2)
            example = {'context_tokens': context_tokens, 'context_chars': context_chars,
                       'ques_tokens': ques_tokens, 'ques_chars': ques_chars, 'y1s': y1s,
                       'y2s': y2s, 'id': qa['id'], 'context': context, 'spans': spans}
            examples.append(example)


def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None):
    r"""
    Gnerate embedding matrix.
    """
    print('Generating {} embedding...'.format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, 'r', encoding='utf-8') as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = ''.join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print('{} / {} tokens have corresponding {} embedding vector'.format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print('{} tokens have corresponding embedding vector'.format(
            len(filtered_elements)))

    NULL = '--NULL--'
    OOV = '--OOV--'
    token2idx_dict = {token: idx for idx,
                      token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def build_features(examples, data_type, word2idx_dict, char2idx_dict):
    r"""
    Generate all features.
    """
    if data_type == 'train':
        para_limit = 400
        ques_limit = 50
        ans_limit = 30
        char_limit = 16
    else:
        para_limit = 1000
        ques_limit = 100
        ans_limit = 30
        char_limit = 16

    def filter_func(example):
        return len(example['context_tokens']) > para_limit or \
            len(example['ques_tokens']) > ques_limit or \
            (example['y2s'][0] - example['y1s'][0]) > ans_limit

    print('Processing {} examples...'.format(data_type))
    total = 0
    total_ = 0
    meta = {}
    record = []
    for example in tqdm(examples):
        total_ += 1

        if data_type == 'train' and filter_func(example):
            continue

        total += 1
        context_idxs = np.zeros([para_limit], dtype=np.int32)
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        for i, token in enumerate(example['context_tokens']):
            context_idxs[i] = _get_word(token)

        for i, token in enumerate(example['ques_tokens']):
            ques_idxs[i] = _get_word(token)

        for i, token in enumerate(example['context_chars']):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[i, j] = _get_char(char)

        for i, token in enumerate(example['ques_chars']):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i, j] = _get_char(char)

        start, end = example['y1s'][-1], example['y2s'][-1]

        record.append([example['id'],
                       context_idxs.tolist(),
                       ques_idxs.tolist(),
                       context_char_idxs.tolist(),
                       ques_char_idxs.tolist(),
                       start,
                       end,
                       example['context'],
                       example['spans']])
    print('Built {} / {} instances of features in total'.format(total, total_))
    meta['total'] = total

    return record


def save(filename, obj, message=None):
    r"""
    Save json file.
    """
    if message is not None:
        print('Saving {}...'.format(message))
        with open(filename, 'w') as fh:
            json.dump(obj, fh)


def prepro():
    r"""
    Main dataprocess function.
    """
    word_counter, char_counter = Counter(), Counter()
    word_emb_file = 'glove.840B.300d.txt'
    char_emb_file = None
    char_emb_size = None
    char_emb_dim = 64
    train_examples = process_file(
        'train-v1.1.json', 'train', word_counter, char_counter)

    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, 'word', emb_file=word_emb_file, size=2.2e6, vec_size=300)
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, 'char', emb_file=char_emb_file, size=char_emb_size, vec_size=char_emb_dim)

    dev_examples = process_file(
        'dev-v1.1.json', 'dev', word_counter, char_counter)

    train_record = build_features(
        train_examples, 'train', word2idx_dict, char2idx_dict)
    dev_record = build_features(
        dev_examples, 'dev', word2idx_dict, char2idx_dict)

    train_sorted_examples = sorted(train_examples, key=lambda x: (
        len(x['context_tokens']), len(x['ques_tokens'])), reverse=True)
    dev_sorted_examples = sorted(dev_examples, key=lambda x: (
        len(x['context_tokens']), len(x['ques_tokens'])), reverse=True)
    train_sorted = build_features(
        train_sorted_examples, 'train', word2idx_dict, char2idx_dict)
    dev_sorted = build_features(
        dev_sorted_examples, 'dev', word2idx_dict, char2idx_dict)
    save('train_record.json', train_record, message='train record')
    save('train_sorted.json', train_sorted, message='train sorted')
    save('dev_record.json', dev_record, message='dev record')
    save('dev_sorted.json', dev_sorted, message='dev sorted')
    save('word_emb.json', word_emb_mat, message='word embedding')
    save('char_emb.json', char_emb_mat, message='char embedding')
    save('train_word_dictionar.json', word2idx_dict, message='word dictionary')
    save('train_char_dictionary.json', char2idx_dict, message='char dictionary')


if __name__ == '__main__':
    prepro()
