import os
import re

from gluonnlp import data, Vocab
from gluonnlp.data.squad_dataset import SQuAD, SQuADTransform


def test_load_dev_squad():
    dataset = SQuAD(segment="dev", root=os.path.join('tests', 'data', 'squad'))

    for i in range(10):
        data = dataset[i]
        print("{}: {}\n{}\n{}\n\n".format(data[0], data[1], data[2], data[3]))


def test_load_vocabs():
    dataset = SQuAD(segment="dev", root=os.path.join('tests', 'data', 'squad'))
    assert dataset.word_vocab is not None
    assert dataset.char_vocab is not None


def test_transform_to_numpy():
    # got these numbers by random
    question_max_length = 96
    context_max_length = 512

    dataset = SQuAD(segment="dev", root=os.path.join('tests', 'data', 'squad'))
    transformer = SQuADTransform(dataset.word_vocab, dataset.char_vocab,
                                 question_max_length, context_max_length)

    transformed_record = transformer(*dataset[0])


def test_create_char_vocab():
    _create_squad_vocab(iter, 'tests/data/squad/char_vocab.json')


def test_create_word_vocab():
    def simple_tokenize(source_str, token_delim=' ', seq_delim='\n'):
        return list(filter(None, re.split(token_delim + '|' + seq_delim, source_str)))

    _create_squad_vocab(simple_tokenize, 'tests/data/squad/word_vocab.json')


def _create_squad_vocab(tokenization_fn, output_path):
    dataset_train = SQuAD(segment="train", root=os.path.join('tests', 'data', 'squad'))
    all_tokens = []

    for i in range(len(dataset_train)):
        data_item = dataset_train[i]

        # we don't add data_item[0] because it contains question Id
        all_tokens.extend(tokenization_fn(data_item[1]))
        all_tokens.extend(tokenization_fn(data_item[2]))

        for answer in data_item[3]:
            all_tokens.extend(tokenization_fn(answer[1]))

    counter = data.count_tokens(all_tokens)
    vocab = Vocab(counter)
    vocab_json = vocab.to_json()

    with open(output_path, 'w') as outfile:
        outfile.write(vocab_json)