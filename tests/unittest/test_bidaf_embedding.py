import re

from mxnet.gluon import nn
from mxnet import nd, init

import gluonnlp as nlp
from gluonnlp.model.block import CharacterLevelCNNEmbedding, PredefinedEmbedding


def test_create_char_embedding():
    input = 'my custom string'
    vocab = nlp.Vocab(nlp.data.count_tokens(input.__iter__(), to_lower=True))

    char_vocab_size = len(vocab)
    char_emb_size = 8

    net = nn.HybridSequential()
    with net.name_scope():
        net.add(CharacterLevelCNNEmbedding(channels=[100], kernel_sizes=[5], padding=-1,
                                           vocab_size=char_vocab_size))

    net.initialize(init.Xavier(magnitude=2.24))

    indices = nd.array([vocab(list(input.__iter__())[0:10])])
    print(indices)
    out = net.forward(indices)
    print(out)


def test_create_word_embedding_layer():
    glove_embedding = nlp.embedding.create('glove', source='glove.6B.100d')
    input = 'my custom string'

    net = nn.HybridSequential()
    with net.name_scope():
        net.add(PredefinedEmbedding(glove_embedding))

    net.initialize(init.Xavier(magnitude=2.24))
    tokens = _simple_tokenize(input)
    counter = nlp.data.count_tokens(tokens)
    vocab = nlp.Vocab(counter)
    vocab.set_embedding(glove_embedding)

    indices = nd.array(vocab(tokens))

    out = net.forward(indices)
    print(out)


def _simple_tokenize(source_str, token_delim=' ', seq_delim='\n'):
    return list(filter(None, re.split(token_delim + '|' + seq_delim, source_str)))
