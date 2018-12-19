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

# Get the data and build a data pipeline.
# Then build a model based on the given hyperparameters.
# Finally train and verify model performance.
# @author：kenjewu
# @date：2018/12/12

import argparse
import os
import logging
import sys
sys.path.append('..')

import itertools
import pickle

import mxnet as mx
import numpy as np
import gluonnlp as nlp
import train_helper as th

from mxnet import autograd, gluon, init, nd
from mxnet.gluon.data import ArrayDataset, DataLoader

from algorithms.cnn_bilstm_crf import CNN_BILSTM_CRF
from utils_func import get_data_bio2, get_data_bioes


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()."""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


CTX = try_gpu()


def build_dataset(sentences, sentences_tags, word_vocab, char_vocab, tag_vocab, max_char_len):
    '''build dataset for model

    Args:
        sentences (list): list of list of word
        sentences_tags (list): list of list of tag
        word_vocab (nlp.Vocab): vocab of all words in train dataset
        char_vocab (nlp.Vocab): vocab of all chars in train dataset
        tag_vocab (nlp.Vocab): vocab of all tags in train dataset
        max_char_len (int): max length of character that one word should has

    Returns:
        nlp.data.SimpleDataset: a dataset that include word_idx, char_idx, tag_idx
    '''

    char_paded_value = char_vocab[char_vocab.padding_token]
    pad_chars = nlp.data.PadSequence(max_char_len, pad_val=char_paded_value, clip=True)

    per_chars_of_sentences = []
    for sent in sentences:
        per_chars = []
        for word in sent:
            per_chars.append(pad_chars(char_vocab[list(word)]))
        per_chars_of_sentences.append(per_chars)
    sentences = [word_vocab[sent] for sent in sentences]
    tags = [tag_vocab[tags] for tags in sentences_tags]

    # dataset [[x,x,x], [[c,c], [c,c], [c,c]], [t, t, t]]
    dataset = gluon.data.SimpleDataset([[sent, per_chars, tag] for sent, per_chars,
                                        tag in zip(sentences, per_chars_of_sentences, tags) if len(sent) > 0])

    return dataset


def data_pipeline(train_data_path, valid_data_path, test_data_path,
                  embedding, word_vocab_path, char_vocab_path,
                  tag_vocab_path, batch_size, max_char_len, logger):
    '''
    Get the data and build a data pipeline to prepare for the training model.

    Args:
        train_data_path (str): path of train data
        valid_data_path (str): path of valid data
        test_data_path (str): path of test data
        embedding (str): the category of pre-trained embedding, ['glove', 'senna', 'random']
        word_vocab_path (str): path of word vocab
        char_vocab_path (str): path of char vocab
        tag_vocab_path (str): path of tag vocab
        batch_size (int): batch size
        max_char_len (int): The maximum number of characters each word has
        logger (logger): logging some data

    Returns:
        char_vocab (nlp.Vocab) : Vocab of characters
        word_vocab (nlp.Vocab) : Vocab of words
        tag_vocab (nlp.Vocab) : Vocab of tags
        train_dataloader (gluon.data.DataLoader) : dataloader of train data
        valid_dataloader (gluon.data.DataLoader) : dataloader of valid data
        test_dataloader (gluon.data.DataLoader) : dataloader of test data
        [type]: [description]
    '''

    # get data and transform to bio2 format
    train_sentences, train_sentences_tags = get_data_bio2(train_data_path)
    valid_sentences, valid_sentences_tags = get_data_bio2(valid_data_path)
    test_sentences, test_sentences_tags = get_data_bio2(test_data_path)

    logger.info('train_sentences length: %d, valid_sentences length: %d, test_sentences length: %d'
                % (len(train_sentences), len(valid_sentences), len(test_sentences)))

    # build word_vocab, char_vocab, tag_vocab
    all_sentences = train_sentences + valid_sentences + test_sentences
    all_tags = train_sentences_tags + valid_sentences_tags + test_sentences_tags
    counter = nlp.data.count_tokens(list(itertools.chain.from_iterable(all_sentences)))
    word_vocab = nlp.Vocab(counter, max_size=50000)

    char_counter = nlp.data.count_tokens(list(''.join([''.join(sentence) for sentence in all_sentences])))
    char_vocab = nlp.Vocab(char_counter)

    tag_counter = nlp.data.count_tokens(list(itertools.chain.from_iterable(all_tags)))
    tag_vocab = nlp.Vocab(tag_counter, padding_token='O')

    # attach pre-trained embedding for word_vocab
    if embedding == 'glove':
        glove100_emb = nlp.embedding.GloVe(source='glove.6B.100d')
        logger.info('Attach the glove.6B.100d to word_vocab.')
        word_vocab.set_embedding(glove100_emb)
    elif embedding == 'senna':
        logger.warning('Not find senna embedding!')
    else:
        logger.info('Random embedding!')

    logger.info('tag to idx: {0}'.format(tag_vocab.token_to_idx))
    logger.info('word_vocab length: %d', len(word_vocab))
    logger.info('char_vocab length: %d', len(char_vocab))
    logger.info('tag_vocab length: %d', len(tag_vocab))

    with open(word_vocab_path, 'wb') as fw1, open(char_vocab_path, 'wb') as fw2, open(tag_vocab_path, 'wb') as fw3:
        pickle.dump(word_vocab, fw1)
        pickle.dump(char_vocab, fw2)
        pickle.dump(tag_vocab, fw3)

    # Temporarily do not cut the length, directly fill each batch of data with the maximum sentence length of each batch
    # length_clip = nlp.data.ClipSequence(200)
    # sentences = [length_clip(sent) for sent in train_sentences]
    # labels = [length_clip(label) for label in train_labels]

    # build train, valid, test dataset.
    train_dataset = build_dataset(train_sentences, train_sentences_tags,
                                  word_vocab, char_vocab, tag_vocab, max_char_len)
    valid_dataset = build_dataset(valid_sentences, valid_sentences_tags,
                                  word_vocab, char_vocab, tag_vocab, max_char_len)
    test_dataset = build_dataset(test_sentences, test_sentences_tags, word_vocab, char_vocab, tag_vocab, max_char_len)

    train_dataset_lengths = [len(data[0]) for data in train_dataset]

    # Bucketing and  Dataloader
    word_pad_value = word_vocab[word_vocab.padding_token]
    char_pad_value = char_vocab[char_vocab.padding_token]
    tag_pad_value = tag_vocab[tag_vocab.padding_token]

    batchify_fn = nlp.data.batchify.Tuple(
        nlp.data.batchify.Pad(pad_val=word_pad_value),
        nlp.data.batchify.Pad(pad_val=char_pad_value),
        nlp.data.batchify.Pad(pad_val=tag_pad_value))
    batch_sampler = nlp.data.sampler.FixedBucketSampler(train_dataset_lengths, batch_size=batch_size, num_buckets=20,
                                                        ratio=0.3, shuffle=True)
    train_dataloader = gluon.data.DataLoader(train_dataset, batch_sampler=batch_sampler, batchify_fn=batchify_fn)
    valid_dataloader = gluon.data.DataLoader(valid_dataset, batch_size=batch_size,
                                             shuffle=False, batchify_fn=batchify_fn)
    test_dataloader = gluon.data.DataLoader(test_dataset, batch_size=1, shuffle=False, batchify_fn=batchify_fn)

    logger.info(batch_sampler.stats())

    return char_vocab, word_vocab, tag_vocab, train_dataloader, valid_dataloader, test_dataloader


def build_model(nchars, nchar_embed, nwords, nword_embed, nfilters,
                kernel_size, nhiddens, nlayers, ntag_space, emb_drop_prob,
                rnn_drop_prob, out_drop_prob, tag2idx):
    '''Model based on a given hyperparameter

    Args:
        nchars (int): size of character vocab
        nchar_embed (int): dim of character vector
        nwords (int): size of word vocab
        nword_embed (int): dim of word vector
        nfilters (int): the number of filters
        kernel_size (int): wind of conv
        nhiddens (int): the number of lstm hidden units
        nlayers (int): the number of lstm layers
        ntag_space (int): the dim of tag space
        emb_drop_prob (float): drop prob of embedding
        rnn_drop_prob (tuple or list of int): lstm input drop prob and cell drop prob
        out_drop_prob (float): dorp prob of output
        tag2idx (dict): tag --> idx

    Returns:
        model (Block): CharacterCNN + BiLSTM + CRF for NER
    '''

    model = CNN_BILSTM_CRF(nchars, nchar_embed, nwords, nword_embed, nfilters, kernel_size,
                           nhiddens, nlayers, ntag_space, emb_drop_prob, rnn_drop_prob, out_drop_prob,
                           tag2idx,)
    # use Xavier initializer
    model.initialize(init=init.Xavier(), ctx=CTX)
    return model


def train_and_valid(model, optim_name, train_dataloader, valid_dataloader, test_dataloader,
                    nepochs, lr, clip, word_vocab, char_vocab, label_vocab, lr_decay_step,
                    lr_decay_rate, logger):
    '''Train the model on the training set and evaluate the model on the validation set and test set

    Args:
        model (Block): CharacterCNN + BiLSTM + CRF for NER
        optim_name (str): name of optimize operator
        train_dataloader (gluon.data.DataLoader) : dataloader of train data
        valid_dataloader (gluon.data.DataLoader) : dataloader of valid data
        test_dataloader (gluon.data.DataLoader) : dataloader of test data
        nepochs (int): the number of epoch
        lr (float): learning_rate
        clip (float): clip value
        word_vocab (nlp.Vocab): vocab of all words in train dataset
        char_vocab (nlp.Vocab): vocab of all chars in train dataset
        tag_vocab (nlp.Vocab): vocab of all tags in train dataset
        lr_decay_step (int): Interval steps for learning rate decay
        lr_decay_rate (float): rate for learning rate decay
        logger (logging): logging some data
    '''

    # user the crf loss
    loss = model.crf_layer.neg_log_likelihood

    # choose trainer
    if optim_name.lower() == 'adam':
        logger.info('use adam to optimize params')
        trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr})
    elif optim_name.lower() == 'sgd':
        logger.info('use sgd to optimize params')
        trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9})
    # train and evaluate
    th.train(train_dataloader, valid_dataloader, test_dataloader, model, loss, trainer,
             CTX, nepochs, word_vocab, char_vocab, label_vocab, clip, lr, lr_decay_step,
             lr_decay_rate, logger)


def main(args):
    # Parsing parameters
    train_data_path = args.train_data_path
    valid_data_path = args.valid_data_path
    test_data_path = args.test_data_path
    word_vocab_path = args.word_vocab_path
    char_vocab_path = args.char_vocab_path
    tag_vocab_path = args.tag_vocab_path

    embedding = args.embedding
    char_len_per_word = args.char_len_per_word
    nchar_embed = args.nchar_embed
    nword_embed = args.nword_embed
    nfilters = args.nfilters
    kernel_size = args.kernel_size
    nhiddens = args.nhiddens
    nlayers = args.nlayers
    ntag_space = args.ntag_space

    emb_drop_prob = args.emb_drop_prob
    out_drop_prob = args.out_drop_prob
    rnn_drop_prob = args.rnn_drop_prob

    nepochs = args.nepochs
    lr = args.lr
    batch_size = args.batch_size
    clip = args.clip
    lr_decay_step = args.lr_decay_step
    lr_decay_rate = args.lr_decay_rate

    optim_name = args.optim_name
    log_path = args.log_path
    if not os.path.exists(os.path.split(log_path)[0]):
        os.makedirs(os.path.split(log_path)[0])

    # Configuring logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fhandler = logging.FileHandler(log_path, mode='w')
    shandler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    shandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.addHandler(shandler)

    # prepare data
    char_vocab, word_vocab, tag_vocab, \
        train_dataloader, valid_dataloader, test_dataloader = data_pipeline(train_data_path, valid_data_path,
                                                                            test_data_path,
                                                                            embedding,
                                                                            word_vocab_path,
                                                                            char_vocab_path,
                                                                            tag_vocab_path,
                                                                            batch_size, char_len_per_word,
                                                                            logger)

    # build model
    nchars, nwords, tag2idx = len(char_vocab), len(word_vocab), tag_vocab._token_to_idx
    model = build_model(nchars, nchar_embed, nwords, nword_embed, nfilters, kernel_size,
                        nhiddens, nlayers, ntag_space, emb_drop_prob, rnn_drop_prob, out_drop_prob,
                        tag2idx,)

    # use pretrained embedding
    if embedding != 'random':
        # print('setting embedding for layer')
        model.word_embedding_layer.weight.set_data(word_vocab.embedding.idx_to_vec)
        # fix or finetune
        # model.word_embedding_layer.collect_params().setattr('grad_req', 'null')

    # train and evaluate
    train_and_valid(model, optim_name, train_dataloader, valid_dataloader, test_dataloader, nepochs,
                    lr, clip, word_vocab, char_vocab, tag_vocab, lr_decay_step, lr_decay_rate, logger)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('train model')

    parser.add_argument('--train', dest='train_data_path', action='store')
    parser.add_argument('--valid', dest='valid_data_path', action='store')
    parser.add_argument('--test', dest='test_data_path', action='store')
    parser.add_argument('--wvp', dest='word_vocab_path', action='store')
    parser.add_argument('--cvp', dest='char_vocab_path', action='store')
    parser.add_argument('--tvp', dest='tag_vocab_path', action='store')

    parser.add_argument('--embedding', choices=['glove', 'senna', 'random'],
                        help='Embedding for words', required=True)
    parser.add_argument('--clpw', type=int, dest='char_len_per_word', default=12, action='store')
    parser.add_argument('--nce', type=int, dest='nchar_embed', default=30, action='store')
    parser.add_argument('--nwe', type=int, dest='nword_embed', default=300, action='store')
    parser.add_argument('--nf', type=int, dest='nfilters', default=30, action='store')
    parser.add_argument('--ks', type=int, dest='kernel_size', default=3, action='store')
    parser.add_argument('--nhiddens', type=int, dest='nhiddens', default=128, action='store')
    parser.add_argument('--nlayers', type=int, dest='nlayers', default=1, action='store')
    parser.add_argument('--nts', type=int, dest='ntag_space', default=128, action='store')

    parser.add_argument('--edp', type=float, dest='emb_drop_prob', default=0.33, action='store')
    parser.add_argument('--odp', type=float, dest='out_drop_prob', default=0.33, action='store')
    parser.add_argument('--rdp', type=float, dest='rnn_drop_prob', nargs=2, required=True)

    parser.add_argument('--nepochs', type=int, dest='nepochs', default=100, action='store')
    parser.add_argument('--lr', type=float, dest='lr', default=0.015, action='store')
    parser.add_argument('--bc', type=int, dest='batch_size', default=16, action='store')
    parser.add_argument('--clip', type=float, dest='clip', default=None, action='store')
    parser.add_argument('--lds', type=int, dest='lr_decay_step', default=1, action='store')
    parser.add_argument('--ldr', type=float, dest='lr_decay_rate', default=0.1, action='store')

    parser.add_argument('--op_name', type=str, dest='optim_name', default='sgd', action='store')
    parser.add_argument('--lp', type=str, dest='log_path', action='store')

    args = parser.parse_args()

    main(args)
