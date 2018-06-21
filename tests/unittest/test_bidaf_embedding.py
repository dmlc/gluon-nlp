import numpy as np

from mxnet import nd, init

import gluonnlp as nlp
from gluonnlp.model.block import CharacterLevelCNNEmbedding, PredefinedEmbedding, BiDAFEmbedding


def test_create_char_embedding():
    input_batch = _get_input_batch()
    vocab = nlp.Vocab(nlp.data.count_tokens(list(_get_character_set(input_batch)), to_lower=True))
    input_batch_nd = nd.array(_replace_chars_with_indices(input_batch, vocab), dtype=np.int)

    embedding = CharacterLevelCNNEmbedding(channels=[100], kernel_sizes=[5], padding=0,
                                           vocab_size=len(vocab), char_embedding_size=8)
    embedding.initialize(init.Xavier(magnitude=2.24))
    out = embedding(input_batch_nd)

    # The output should batch_size (2, because 2 examples in input), seq_len (5, because 5 words
    # in each example), channels (100, because output embedding is of size 100)
    assert out is not None
    assert out.shape == (len(input_batch), len(input_batch[0].split(' ')), 100)


def test_create_predefined_embedding_layer():
    input_batch = _get_input_batch()

    glove_embedding = nlp.embedding.create('glove', source='glove.6B.100d')
    embedding = PredefinedEmbedding(glove_embedding)
    embedding.initialize(init.Xavier(magnitude=2.24))

    tokens = _get_token_set(input_batch)

    vocab = nlp.Vocab(nlp.data.count_tokens(tokens))
    vocab.set_embedding(glove_embedding)

    input_batch_nd = nd.array(_replace_words_with_indices(input_batch, vocab), dtype=np.int)
    out = embedding(input_batch_nd)

    # The output should batch_size (2, because 2 examples in input), seq_len (5, because 5 words
    # in each example), channels (100, because we use glove embedding of size 100)
    assert out is not None
    assert out.shape == (len(input_batch), len(input_batch[0].split(' ')), 100)


def test_create_bidaf_embedding_layer():
    input_batch = _get_input_batch()
    counter = nlp.data.count_tokens(list(_get_character_set(input_batch)), to_lower=True)
    char_vocab = nlp.Vocab(counter)

    glove_embedding = nlp.embedding.create('glove', source='glove.6B.100d')

    embedding = BiDAFEmbedding(glove_embedding, channels=[100], kernel_sizes=[5], padding=0,
                               char_vocab_size=len(char_vocab), char_embedding_size=8)
    embedding.initialize(init.Xavier(magnitude=2.24))

    tokens = _get_token_set(input_batch)
    word_vocab = nlp.Vocab(nlp.data.count_tokens(tokens))
    word_vocab.set_embedding(glove_embedding)

    char_level_batch = nd.array(_replace_chars_with_indices(input_batch, char_vocab), dtype=np.int)
    word_level_batch = nd.array(_replace_words_with_indices(input_batch, word_vocab), dtype=np.int)

    out = embedding(char_level_batch, word_level_batch)

    # The output should batch_size (2, because 2 examples in input), seq_len (5, because 5 words
    # in each example), channels (200, because we use concat by channels dimensions: 2*100 = 200)
    assert out is not None
    assert out.shape == (len(input_batch), len(input_batch[0].split(' ')), 200)


def _get_input_batch():
    # for simplicity of tests, length of words and length of sentences are same, but it will be
    # different for real question and context - padding to the same length would be needed
    # The max size of a word for both question and context is 16 characters
    return ['query11111111111 1234567890123456 1234567890123456 1234567890123456 1234567890123456',
            'query22222222222 1234567890123456 1234567890123456 1234567890123456 1234567890123456']


def _get_token_set(input_batch):
    token_list = []

    for sentence in input_batch:
        token_list.extend([word for word in sentence.split(' ')])

    return token_list


def _get_character_set(input_batch):
    chars_list = []

    for sentence in input_batch:
        word_chars = [list(iter(word)) for word in sentence.split(' ')]

        for word in word_chars:
            chars_list.extend([char for char in word])

    return set(chars_list)


def _replace_words_with_indices(text, vocab):
    result = []

    for sentence in text:
        sentence_array = [vocab.to_indices(word) for word in sentence.split(' ')]
        result.append(sentence_array)

    return result


def _replace_chars_with_indices(text, vocab):
    result = []

    for sentence in text:
        sentence_array = []

        for word in sentence.split(' '):
            word_array = vocab.to_indices(list(iter(word)))
            sentence_array.append(word_array)

        result.append(sentence_array)

    return result
