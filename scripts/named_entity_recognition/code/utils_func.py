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

# Some general functions in this project
# @author：kenjewu
# @date：2018/12/12


import re
import os
import itertools
import numpy as np
import gluonnlp as nlp


def transform(ifile, ofile):
    '''transform the raw bio to bio2

    Args:
        ifile (str): path of raw file
        ofile (str): path of converted file
    '''

    with open(ifile, 'r') as reader, open(ofile, 'w') as writer:
        for line in itertools.islice(reader, 0, 2):
            writer.write(line)
        prev = 'O'
        for line in itertools.islice(reader, 0, None):
            line = line.strip()
            if len(line) == 0:
                prev = 'O'
                writer.write('\n')
                continue

            tokens = line.split()
            # print tokens
            label = tokens[-1]
            if label != 'O' and label != prev:
                if prev == 'O':
                    label = 'B-' + label[2:]
                elif label[2:] != prev[2:]:
                    label = 'B-' + label[2:]
                else:
                    label = label
            writer.write(" ".join(tokens[:-1]) + " " + label)
            writer.write('\n')
            prev = tokens[-1]


def get_data_from_txt(data_path):
    '''get data from given data_path

    Args:
        data_path (str): path

    Returns:
        sentences (list): [description]
        sentences_poses (list): list of list of pos
        sentences_syns (list): list of list of syn
        sentences_tags (list): list of list of tag
    '''

    with open(data_path, 'r') as fr:

        sentences, sentences_poses, sentences_syns, sentences_tags = [], [], [], []
        sentence, poses, syns, tags = [], [], [], []

        for line in itertools.islice(fr, 2, None):
            if len(line.strip()) != 0:
                word, pos, syntactic, entity_tag = line.rstrip().split(' ')
                sentence.append(word)
                poses.append(pos)
                syns.append(syntactic)
                tags.append(entity_tag)
            else:
                sentences.append(sentence)
                sentences_poses.append(poses)
                sentences_syns.append(syns)
                sentences_tags.append(tags)
                # clear elements
                sentence, poses, syns, tags = [], [], [], []

        return sentences, sentences_poses, sentences_syns, sentences_tags


def bio_bioes(tags):
    '''the function is used to convert
    BIO -> BIOES tagging

    Args:
        tags (list): list of tag

    Raises:
        Exception: [description]

    Returns:
        new_tags (list): list of converted tag (bioes)
    '''
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def check_data_tags_info(tags):
    '''do some statitic check of data

    Args:
        tags (list): list of tag

    Returns:
        some count: count result of every entity
    '''

    O_count, LOC_count, MISC_count, ORG_count, PER_count = 0, 0, 0, 0, 0
    for tag in itertools.chain.from_iterable(tags):
        if tag == 'O':
            O_count += 1
            continue
        suffix = tag.split('-')[-1]
        if suffix == 'LOC':
            LOC_count += 1
        elif suffix == 'MISC':
            MISC_count += 1
        elif suffix == 'ORG':
            ORG_count += 1
        elif suffix == 'PER':
            PER_count += 1

    print(LOC_count + MISC_count + ORG_count + PER_count + O_count)
    print(len(list(itertools.chain.from_iterable(tags))))
    assert (LOC_count + MISC_count + ORG_count + PER_count + O_count) == len(list(itertools.chain.from_iterable(tags)))
    return O_count, LOC_count, MISC_count, ORG_count, PER_count


def to_lower_zero(sentence):
    '''convert str to lower and replace the number with 0

    Args:
        sentence (list): list of list

    Returns:
        new_sentence (list): list of list
    '''

    def transform(x):
        temp = re.sub(r'\d', '0', x.lower())
        # return re.sub(r'-', ' ', temp)
        return temp

    new_sentence = list(map(transform, sentence))
    return new_sentence


def get_data_bioes(data_path):
    '''get bioes data

    Args:
        data_path (str): the path of raw data

    Returns:
        new_sentences (list): list of list
        new_tags (list): list of list
    '''

    sentences, _, _, tags = get_data_from_txt(data_path)

    new_sentences = [to_lower_zero(sentence) for sentence in sentences]
    new_tags = [bio_bioes(tag) for tag in tags]

    return new_sentences, new_tags


def get_data_bio2(data_path):
    '''get bio2 data

    Args:
        data_path (str): the path of raw data

    Returns:
        new_sentences (list): list of list
        new_tags (list): list of list
    '''
    ifile = data_path
    root, filename = os.path.split(data_path)
    ofile = os.path.join(root, filename.split('.')[0] + '_bio2.txt')
    transform(ifile, ofile)
    sentences, _, _, tags = get_data_from_txt(ofile)
    new_sentences = [to_lower_zero(sentence) for sentence in sentences]
    new_tags = tags

    return new_sentences, new_tags


if __name__ == '__main__':
    train_data_path = '../data/eng_train.txt'
    valid_data_path = '../data/eng_testa.txt'
    test_data_path = '../data/eng_testb.txt'

    train_sentences, train_sentences_tags = get_data_bio2(train_data_path)
    valid_sentences, valid_sentences_tags = get_data_bio2(valid_data_path)
    test_sentences, test_sentences_tags = get_data_bio2(test_data_path)
    print(len(train_sentences), len(valid_sentences), len(test_sentences))

    print('word: ')
    t = list(itertools.chain.from_iterable(train_sentences))
    v = list(itertools.chain.from_iterable(valid_sentences))
    t2 = list(itertools.chain.from_iterable(test_sentences))
    print(len(set(t)))
    print(len(set(v)))
    print(len(set(t2)))
    print(len(set(t + v + t2)))
    print(len(set(list(''.join(t+v+t2)))))
    print()

    # do some checking
    train_O_count, train_LOC_count, train_MISC_count, train_ORG_count, train_PER_count = check_data_tags_info(
        train_sentences_tags)
    valid_O_count, valid_LOC_count, valid_MISC_count, valid_ORG_count, valid_PER_count = check_data_tags_info(
        valid_sentences_tags)
    test_O_count, test_LOC_count, test_MISC_count, test_ORG_count, test_PER_count = check_data_tags_info(
        test_sentences_tags)

    print('      |  O |  LOC  |  MISC  |  ORG  |  PER  |')
    print('train:', train_O_count, train_LOC_count, train_MISC_count, train_ORG_count, train_PER_count)
    print('valid:', valid_O_count, valid_LOC_count, valid_MISC_count, valid_ORG_count, valid_PER_count)
    print('test: ', test_O_count, test_LOC_count, test_MISC_count, test_ORG_count, test_PER_count)
