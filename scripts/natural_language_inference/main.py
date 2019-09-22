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
# Copyright 2018 Mengxiao Lin <linmx0130@gmail.com>.
# pylint: disable=redefined-outer-name,logging-format-interpolation

"""
Decomposable Attention Models for Natural Language Inference
============================================================

This script reproduces results in [Parikh et al., 2016]  with the Gluon NLP Toolkit.

@article{parikh2016decomposable,
  title={A decomposable attention model for natural language inference},
  author={Parikh, Ankur P and T{\"a}ckstr{\"o}m, Oscar and Das, Dipanjan and Uszkoreit, Jakob},
  journal={arXiv preprint arXiv:1606.01933},
  year={2016}
}
"""

import os
import argparse
import json
import logging
import numpy as np

import mxnet as mx
from mxnet import gluon, autograd
import gluonnlp as nlp

from decomposable_attention import DecomposableAttentionModel
from esim import ESIMModel
from dataset import read_dataset, prepare_data_loader, build_vocab
from utils import logging_config

logger = logging.getLogger('nli')

nlp.utils.check_version('0.7.0')

def parse_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU id (-1 means CPU)')
    parser.add_argument('--train-file', default='snli_1.0/snli_1.0_train.txt',
                        help='training set file')
    parser.add_argument('--test-file', default='snli_1.0/snli_1.0_dev.txt',
                        help='validation set file')
    parser.add_argument('--max-num-examples', type=int, default=-1,
                        help='maximum number of examples to load (for debugging)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--print-interval', type=int, default=20,
                        help='the interval of two print')
    parser.add_argument('--model', choices=['da', 'esim'], default=None, required=True,
                        help='which model to use')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                        help='train or test')
    parser.add_argument('--lr', type=float, default=0.025,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=300,
                        help='maximum number of epochs to train')
    parser.add_argument('--embedding', default='glove',
                        help='word embedding type')
    parser.add_argument('--fix-embedding', action='store_true',
                        help='whether to fix pretrained word embedding')
    parser.add_argument('--embedding-source', default='glove.840B.300d',
                        help='embedding file source')
    parser.add_argument('--embedding-size', type=int, default=300,
                        help='size of pretrained word embedding')
    parser.add_argument('--hidden-size', type=int, default=200,
                        help='hidden layer size')
    parser.add_argument('--output-dir', default='./output',
                        help='directory for all experiment output')
    parser.add_argument('--model-dir', default='./output',
                        help='directory to load model')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='dropout rate')
    parser.add_argument('--optimizer', choices=['adam', 'adagrad'], default='adagrad',
                        help='optimization method')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='l2 regularization weight')
    parser.add_argument('--intra-attention', action='store_true',
                        help='use intra-sentence attention')

    return parser.parse_args()

def train_model(model, train_data_loader, val_data_loader, embedding, ctx, args):
    """
    Train model and validate/save every epoch.
    """
    logger.info(vars(args))

    # Initialization
    model.hybridize()
    model.collect_params().initialize(mx.init.Normal(0.01), ctx=ctx)
    model.word_emb.weight.set_data(embedding.idx_to_vec)
    # Fix word embedding
    if args.fix_embedding:
        model.word_emb.weight.grad_req = 'null'

    loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(model.collect_params(), args.optimizer,
                            {'learning_rate': args.lr,
                             'wd': args.weight_decay,
                             'clip_gradient': 5})

    checkpoints_dir = os.path.join(args.output_dir, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    best_val_acc = 0.
    for epoch_id in range(args.epochs):
        avg_loss = 0.
        avg_acc = 0.
        for batch_id, example in enumerate(train_data_loader):
            s1, s2, label = example
            s1 = s1.as_in_context(ctx)
            s2 = s2.as_in_context(ctx)
            label = label.as_in_context(ctx)

            with autograd.record():
                output = model(s1, s2)
                loss = loss_func(output, label).mean()
            loss.backward()
            trainer.step(1)
            avg_loss += loss.sum().asscalar()

            pred = output.argmax(axis=1)
            acc = (pred == label.astype(np.float32)).mean()
            avg_acc += acc.asscalar()

            if (batch_id + 1) % args.print_interval == 0:
                avg_loss /= args.print_interval
                avg_acc /= args.print_interval
                logger.info('[Epoch {} Batch {}/{}] loss={:.4f}, acc={:.4f}'
                            .format(epoch_id, batch_id + 1, len(train_data_loader),
                                    avg_loss, avg_acc))
                avg_loss = 0.
                avg_acc = 0.

        # Validation
        val_loss, val_acc = test_model(model, val_data_loader, loss_func, ctx)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'valid_best.params')
            model.save_parameters(checkpoint_path)
        logger.info('[Epoch {}] valid loss={:.4f}, valid acc={:.4f}, best valid acc={:.4f}'
                    .format(epoch_id, val_loss, val_acc, best_val_acc))

        # Save checkpoint of last epoch
        checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'last.params')
        model.save_parameters(checkpoint_path)

def test_model(model, data_loader, loss_func, ctx):
    """
    Test model.
    """
    acc = 0.
    loss = 0.
    for _, example in enumerate(data_loader):
        s1, s2, label = example
        s1 = s1.as_in_context(ctx)
        s2 = s2.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = model(s1, s2)
        loss += loss_func(output, label).mean().asscalar()
        pred = output.argmax(axis=1)
        acc += (pred == label.astype(np.float32)).mean().asscalar()
    acc /= len(data_loader)
    loss /= len(data_loader)
    return loss, acc

def build_model(args, vocab):
    if args.model == 'da':
        model = DecomposableAttentionModel(len(vocab), args.embedding_size, args.hidden_size,
                                           args.dropout, args.intra_attention)
    elif args.model == 'esim':
        model = ESIMModel(len(vocab), 3, args.embedding_size, args.hidden_size,
                          args.dropout)
    return model

def main(args):
    """
    Entry point: train or test.
    """
    json.dump(vars(args), open(os.path.join(args.output_dir, 'config.json'), 'w'))

    if args.gpu_id == -1:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu_id)

    mx.random.seed(args.seed, ctx=ctx)

    if args.mode == 'train':
        train_dataset = read_dataset(args, 'train_file')
        val_dataset = read_dataset(args, 'test_file')

        vocab_path = os.path.join(args.output_dir, 'vocab.jsons')
        if os.path.exists(vocab_path):
            vocab = nlp.Vocab.from_json(open(vocab_path).read())
        else:
            vocab = build_vocab(train_dataset)
            with open(vocab_path, 'w') as fout:
                fout.write(vocab.to_json())
        glove = nlp.embedding.create(args.embedding, source=args.embedding_source)
        vocab.set_embedding(glove)

        train_data_loader = prepare_data_loader(args, train_dataset, vocab)
        val_data_loader = prepare_data_loader(args, val_dataset, vocab, test=True)

        model = build_model(args, vocab)
        train_model(model, train_data_loader, val_data_loader, vocab.embedding, ctx, args)
    elif args.mode == 'test':
        model_args = argparse.Namespace(**json.load(
            open(os.path.join(args.model_dir, 'config.json'))))
        vocab = nlp.Vocab.from_json(
            open(os.path.join(args.model_dir, 'vocab.jsons')).read())
        val_dataset = read_dataset(args, 'test_file')
        val_data_loader = prepare_data_loader(args, val_dataset, vocab, test=True)
        model = build_model(model_args, vocab)
        model.load_parameters(os.path.join(
            args.model_dir, 'checkpoints', 'valid_best.params'), ctx=ctx)
        loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
        logger.info('Test on {}'.format(args.test_file))
        loss, acc = test_model(model, val_data_loader, loss_func, ctx)
        logger.info('loss={:.4f} acc={:.4f}'.format(loss, acc))

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging_config(os.path.join(args.output_dir, 'main.log'))

    main(args)
