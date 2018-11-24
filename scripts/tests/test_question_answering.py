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
from __future__ import unicode_literals
import pytest

import mxnet as mx
from mxnet import init, nd, autograd, gluon
from mxnet.gluon import Trainer, nn
from mxnet.gluon.data import DataLoader, SimpleDataset
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from argparse import Namespace

import gluonnlp as nlp
from gluonnlp.data import SQuAD

from ..question_answering.attention_flow import AttentionFlow
from ..question_answering.bidaf import BidirectionalAttentionFlow
from ..question_answering.data_processing import SQuADTransform, VocabProvider
from ..question_answering.utils import PolyakAveraging
from ..question_answering.question_answering import *
from ..question_answering.similarity_function import LinearSimilarity
from ..question_answering.tokenizer import BiDAFTokenizer
from ..question_answering.train_question_answering import get_record_per_answer_span

batch_size = 5
question_max_length = 30
context_max_length = 400
max_chars_per_word = 16
embedding_size = 100


@pytest.mark.serial
@pytest.mark.remote_required
def test_transform_to_nd_array():
    dataset = SQuAD(segment='dev', root='tests/data/squad')
    vocab_provider = VocabProvider([dataset], get_args(batch_size))
    transformer = SQuADTransform(vocab_provider, question_max_length,
                                 context_max_length, max_chars_per_word, embedding_size)
    record = dataset[0]

    transformed_record = transformer(*record)
    assert transformed_record is not None
    assert len(transformed_record) == 7


@pytest.mark.serial
@pytest.mark.remote_required
def test_data_loader_able_to_read():
    dataset = SQuAD(segment='dev', root='tests/data/squad')
    vocab_provider = VocabProvider([dataset], get_args(batch_size))
    transformer = SQuADTransform(vocab_provider, question_max_length,
                                 context_max_length, max_chars_per_word, embedding_size)
    record = dataset[0]

    processed_dataset = SimpleDataset([transformer(*record)])
    loadable_data = SimpleDataset([(r[0], r[2], r[3], r[4], r[5], r[6]) for r in processed_dataset])
    dataloader = DataLoader(loadable_data, batch_size=1)

    for data in dataloader:
        record_index, question_words, context_words, question_chars, context_chars, answers = data

        assert record_index is not None
        assert question_words is not None
        assert context_words is not None
        assert question_chars is not None
        assert context_chars is not None
        assert answers is not None


@pytest.mark.serial
@pytest.mark.remote_required
def test_load_vocabs():
    dataset = SQuAD(segment='dev', root='tests/data/squad')
    vocab_provider = VocabProvider([dataset], get_args(batch_size))

    assert vocab_provider.get_word_level_vocab(embedding_size) is not None
    assert vocab_provider.get_char_level_vocab() is not None


def test_bidaf_embedding():
    dataset = SQuAD(segment='dev', root='tests/data/squad')
    vocab_provider = VocabProvider([dataset], get_args(batch_size))
    transformer = SQuADTransform(vocab_provider, question_max_length,
                                 context_max_length, max_chars_per_word, embedding_size)

    # for performance reason, process only batch_size # of records
    processed_dataset = SimpleDataset([transformer(*record) for i, record in enumerate(dataset)
                                       if i < batch_size])

    # need to remove question id before feeding the data to data loader
    loadable_data, dataloader = get_record_per_answer_span(processed_dataset, get_args(batch_size))

    word_vocab = vocab_provider.get_word_level_vocab(embedding_size)
    word_vocab.set_embedding(nlp.embedding.create('glove', source='glove.6B.100d'))
    char_vocab = vocab_provider.get_char_level_vocab()

    embedding = BiDAFEmbedding(word_vocab=word_vocab,
                               char_vocab=char_vocab,
                               max_seq_len=question_max_length)
    embedding.initialize(init.Xavier(magnitude=2.24), ctx=mx.cpu_pinned())
    embedding.hybridize(static_alloc=True)

    trainer = Trainer(embedding.collect_params(), 'sgd', {'learning_rate': 0.1})

    for i, (data, label) in enumerate(dataloader):
        with autograd.record():
            record_index, q_words, ctx_words, q_chars, ctx_chars = data
            # passing only question_words_nd and question_chars_nd batch
            out = embedding(q_words, q_chars)
            assert out is not None

        out.backward()
        trainer.step(batch_size)
        break


def test_attention_layer():
    ctx_fake_data = nd.random.uniform(shape=(batch_size, context_max_length, 2 * embedding_size))

    q_fake_data = nd.random.uniform(shape=(batch_size, question_max_length, 2 * embedding_size))

    ctx_fake_mask = nd.ones(shape=(batch_size, context_max_length))
    q_fake_mask = nd.ones(shape=(batch_size, question_max_length))

    matrix_attention = AttentionFlow(LinearSimilarity(array_1_dim=6 * embedding_size,
                                                      array_2_dim=1,
                                                      combination='x,y,x*y'),
                                     context_max_length,
                                     question_max_length)

    layer = BidirectionalAttentionFlow(context_max_length,
                                       question_max_length)

    matrix_attention.initialize()
    layer.initialize()

    with autograd.record():
        passage_question_similarity = matrix_attention(ctx_fake_data, q_fake_data).reshape(
            shape=(batch_size, context_max_length, question_max_length))

        output = layer(passage_question_similarity, ctx_fake_data, q_fake_data,
                       q_fake_mask, ctx_fake_mask)

    assert output.shape == (batch_size, context_max_length, 8 * embedding_size)


def test_output_layer():
    # The output layer receive 2 inputs: the output of Modeling layer (context_max_length,
    # batch_size, 2 * embedding_size) and the output of Attention flow layer
    # (batch_size, context_max_length, 8 * embedding_size)

    # The modeling layer returns data in TNC format
    modeling_output = nd.random.uniform(shape=(context_max_length, batch_size, 2 * embedding_size))
    # The layer assumes that attention is already return data in TNC format
    attention_output = nd.random.uniform(shape=(context_max_length, batch_size, 8 * embedding_size))
    ctx_mask = nd.ones(shape=(batch_size, context_max_length))

    layer = BiDAFOutputLayer()
    # The model doesn't need to know the hidden states, so I don't hold variables for the states
    layer.initialize()
    layer.hybridize(static_alloc=True)

    with autograd.record():
        output = layer(attention_output, modeling_output, ctx_mask)

    # We expect final numbers as batch_size x 2 (first start index, second end index)
    assert output[0].shape == (batch_size, 400) and output[1].shape == (batch_size, 400)


def test_bidaf_model():
    options = get_args(batch_size)
    ctx = [mx.cpu(0)]

    dataset = SQuAD(segment='dev', root='tests/data/squad')
    vocab_provider = VocabProvider([dataset], options)
    transformer = SQuADTransform(vocab_provider, question_max_length,
                                 context_max_length, max_chars_per_word, embedding_size)

    # for performance reason, process only batch_size # of records
    processed_dataset = SimpleDataset([transformer(*record) for i, record in enumerate(dataset)
                                       if i < options.batch_size * len(ctx)])

    # need to remove question id before feeding the data to data loader
    train_dataset, train_dataloader = get_record_per_answer_span(processed_dataset, options)

    word_vocab = vocab_provider.get_word_level_vocab(embedding_size)
    word_vocab.set_embedding(nlp.embedding.create('glove', source='glove.6B.100d'))
    char_vocab = vocab_provider.get_char_level_vocab()

    net = BiDAFModel(word_vocab=word_vocab,
                     char_vocab=char_vocab,
                     options=options)

    net.initialize(init.Xavier(magnitude=2.24))
    net.hybridize(static_alloc=True)

    loss_function = SoftmaxCrossEntropyLoss()
    trainer = Trainer(net.collect_params(), 'adadelta', {'learning_rate': 0.5})

    for i, (data, label) in enumerate(train_dataloader):
        record_index, q_words, ctx_words, q_chars, ctx_chars = data

        record_index = gluon.utils.split_and_load(record_index, ctx, even_split=False)
        q_words = gluon.utils.split_and_load(q_words, ctx, even_split=False)
        ctx_words = gluon.utils.split_and_load(ctx_words, ctx, even_split=False)
        q_chars = gluon.utils.split_and_load(q_chars, ctx, even_split=False)
        ctx_chars = gluon.utils.split_and_load(ctx_chars, ctx, even_split=False)
        label = gluon.utils.split_and_load(label, ctx, even_split=False)

        losses = []

        for ri, qw, cw, qc, cc, l in zip(record_index, q_words, ctx_words,
                                         q_chars, ctx_chars, label):
            with autograd.record():
                begin, end = net(qw, cw, qc, cc)
                begin_end = l.split(axis=1, num_outputs=2, squeeze_axis=1)
                loss = loss_function(begin, begin_end[0]) + loss_function(end, begin_end[1])
                losses.append(loss)

        for loss in losses:
            loss.backward()

        trainer.step(options.batch_size)
        break


def test_get_answer_spans_exact_match():
    tokenizer = BiDAFTokenizer()

    context = 'to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a ' \
              'direct line that connects through 3 statues and the Gold Dome), is a simple, ' \
              'modern stone statue of Mary.'
    context_tokens = tokenizer(context)

    answer_start_index = 3
    answer = 'Saint Bernadette Soubirous'

    result = SQuADTransform._get_answer_spans(context, context_tokens,
                                              [answer], [answer_start_index])

    assert result == [(1, 3)]


def test_get_answer_spans_partial_match():
    tokenizer = BiDAFTokenizer()

    context = 'In addition, trucks will be allowed to enter India\'s capital only after 11 p.m., ' \
              'two hours later than the existing restriction'
    context_tokens = tokenizer(context)

    answer_start_index = 72
    answer = '11 p.m'

    result = SQuADTransform._get_answer_spans(context, context_tokens,
                                              [answer], [answer_start_index])

    assert result == [(15, 16)]


def test_get_answer_spans_unicode():
    tokenizer = BiDAFTokenizer()

    context = 'Back in Warsaw that year, Chopin heard Niccolò Paganini play'
    context_tokens = tokenizer(context, lower_case=True)

    answer_start_index = 39
    answer = 'Niccolò Paganini'

    result = SQuADTransform._get_answer_spans(context, context_tokens,
                                              [answer], [answer_start_index])

    assert result == [(8, 9)]


def test_get_answer_spans_after_comma():
    tokenizer = BiDAFTokenizer()

    context = 'Chopin\'s successes as a composer and performer opened the door to western ' \
              'Europe for him, and on 2 November 1830, he set out,'
    context_tokens = tokenizer(context, lower_case=True)

    answer_start_index = 108
    answer = '1830'

    result = SQuADTransform._get_answer_spans(context, context_tokens,
                                              [answer], [answer_start_index])

    assert result == [(22, 22)]


def test_get_answer_spans_after_quotes():
    tokenizer = BiDAFTokenizer()

    context = 'In the film Knute Rockne, All American, Knute Rockne (played by Pat O\'Brien) ' \
              'delivers the famous "Win one for the Gipper" speech, at which point the ' \
              'background music swells with the "Notre Dame Victory March". George Gipp was ' \
              'played by Ronald Reagan, whose nickname "The Gipper" was derived from this role. ' \
              'This scene was parodied in the movie Airplane! with the same background music, ' \
              'only this time honoring George Zipp, one of Ted Striker\'s former comrades. ' \
              'The song also was prominent in the movie Rudy, with Sean Astin as Daniel "Rudy" ' \
              'Ruettiger, who harbored dreams of playing football at the University of Notre ' \
              'Dame despite significant obstacles.'
    context_tokens = tokenizer(context, lower_case=True)

    answer_start_char_index = 267
    answer = 'The Gipper'

    result = SQuADTransform._get_answer_spans(context, context_tokens,
                                              [answer], [answer_start_char_index])

    assert result == [(58, 59)]


def test_get_char_indices():
    context = 'to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a ' \
              'direct line that connects through 3 statues and the Gold Dome), is a simple, ' \
              'modern stone statue of Mary.'
    tokenizer = BiDAFTokenizer()
    context_tokens = tokenizer(context, lower_case=True)

    result = SQuADTransform.get_char_indices(context, context_tokens)
    assert len(result) == len(context_tokens)


def test_tokenizer_split_new_lines():
    context = 'that are of equal energy\u2014i.e., degenerate\u2014is a     configuration ' \
              'termed a spin triplet state. Hence, the ground state of the O\n2 molecule is ' \
              'referred to as triplet oxygen'

    tokenizer = BiDAFTokenizer()
    context_tokens = tokenizer(context, lower_case=True)

    assert len(context_tokens) == 35


def test_polyak_averaging():
    net = nn.HybridSequential()
    net.add(nn.Dense(5), nn.Dense(3), nn.Dense(2))
    net.initialize(init.Xavier())
    net.hybridize()

    ema = None
    loss_fn = SoftmaxCrossEntropyLoss()
    trainer = Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})

    train_data = mx.random.uniform(-0.1, 0.1, shape=(5, 10))
    train_label = mx.nd.array([0, 1, 1, 0, 1])

    for i in range(3):
        with autograd.record():
            o = net(train_data)
            loss = loss_fn(o, train_label)

        if i == 0:
            ema = PolyakAveraging(net.collect_params(), decay=0.999)

        loss.backward()
        trainer.step(5)
        ema.update()

    assert ema.get_params() is not None


def get_args(batch_size_arg):
    options = Namespace()
    options.gpu = None
    options.ctx_embedding_num_layers = 2
    options.embedding_size = 100
    options.dropout = 0.2
    options.ctx_embedding_num_layers = 2
    options.highway_num_layers = 2
    options.modeling_num_layers = 2
    options.output_num_layers = 2
    options.batch_size = batch_size_arg
    options.ctx_max_len = context_max_length
    options.q_max_len = question_max_length
    options.word_max_len = max_chars_per_word
    options.epochs = 12
    options.save_dir = "output/"
    options.filter_long_context = False
    options.word_vocab_path = ""
    options.char_vocab_path = ""
    options.train_unk_token = False

    return options
