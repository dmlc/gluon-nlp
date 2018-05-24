import re

from mxnet.gluon import nn
from mxnet import nd, init

import gluonnlp as nlp
from gluonnlp.model.block import CharacterLevelCNNEmbedding, PredefinedEmbedding


def test_create_char_embedding():
    input_text = 'my custom string'
    vocab = nlp.Vocab(nlp.data.count_tokens(list(iter(input_text)), to_lower=True))

    net = nn.HybridSequential()
    with net.name_scope():
        net.add(CharacterLevelCNNEmbedding(channels=[100], kernel_sizes=[5], padding=-1,
                                           vocab_size=len(vocab)))

    net.initialize(init.Xavier(magnitude=2.24))

    indices = nd.array([vocab(list(iter(input_text))[0:10])])
    print(indices)
    out = net.forward(indices)
    print(out)


def test_create_word_embedding_layer():
    glove_embedding = nlp.embedding.create('glove', source='glove.6B.100d')
    input_text = 'my custom string'

    net = nn.HybridSequential()
    with net.name_scope():
        net.add(PredefinedEmbedding(glove_embedding))

    net.initialize(init.Xavier(magnitude=2.24))
    tokens = _simple_tokenize(input_text)
    counter = nlp.data.count_tokens(tokens)
    vocab = nlp.Vocab(counter)
    vocab.set_embedding(glove_embedding)

    indices = nd.array(vocab(tokens))

    out = net.forward(indices)
    print(out)


def _simple_tokenize(source_str, token_delim=' ', seq_delim='\n'):
    return list(filter(None, re.split(token_delim + '|' + seq_delim, source_str)))
