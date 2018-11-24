# pylint: disable=E1101,R0914
"""
main.py
Main of NLI script in gluon-nlp. Intra-sentence attention model.

Copyright 2018 Mengxiao Lin <linmx0130@gmail.com>
"""

import os
import argparse
import json
import logging
import numpy as np

import gluonnlp as nlp
import mxnet as mx
from mxnet import gluon, autograd, nd

from decomposable_attention import NLIModel
from dataset import read_dataset, prepare_data_loader, build_vocab
from utils import logging_config

logger = logging.getLogger('nli')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
                        help='use CPU')
    parser.add_argument('--train-file',
                        help='training set file', default='snli_1.0/snli_1.0_train.txt')
    parser.add_argument('--test-file',
                        help='validation set file', default='snli_1.0/snli_1.0_dev.txt')
    parser.add_argument('--max-num-examples', type=int, default=-1,
                        help='maximum number of examples to load (for debugging)')
    parser.add_argument('--batch-size',
                        help='batch size', default=32, type=int)
    parser.add_argument('--print-interval', default=20, type=int,
                        help='the interval of two print')
    parser.add_argument('--model',
                        help='model file to test, only for test mode', default=None)
    parser.add_argument('--mode',
                        help='train or test', default='train')
    parser.add_argument('--lr',
                        help='learning rate', default=0.025, type=float)
    parser.add_argument('--epochs', type=int, default=30,
                        help='maximum number of epochs to train')
    parser.add_argument('--embedding',
                        help='word embedding type',
                        default='glove')
    parser.add_argument('--embedding_source',
                        help='embedding file soure',
                        default='glove.6B.300d')
    parser.add_argument('--embedding_size',
                        help='size of embedding. Change it when using new embedding file!',
                        default=300, type=int)
    parser.add_argument('--hidden-size', type=int, default=200,
                        help='hidden layer size')
    parser.add_argument('--output-dir', default='./output',
                        help='directory for all experiment output')
    parser.add_argument('--model-dir', default='./output',
                        help='directory to load model')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')

    return parser.parse_args()

def train_model(model, train_data_loader, val_data_loader, embedding, ctx, args):
    model.hybridize()

    # Initialziation
    model.collect_params().initialize(mx.init.Uniform(0.1), ctx=ctx)
    model.word_emb.weight.set_data(embedding.idx_to_vec)

    loss_func = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(model.collect_params(), 'adagrad',
                            {'learning_rate': args.lr})

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
    acc = 0.
    loss = 0.
    for batch_id, example in enumerate(data_loader):
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

def main(args):
    json.dump(vars(args), open(os.path.join(args.output_dir, 'config.json'), 'w'))

    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu()

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

        model = NLIModel(len(vocab), args.embedding_size, args.hidden_size)
        train_model(model, train_data_loader, val_data_loader, vocab.embedding, ctx, args)
    elif args.mode == 'test':
        model_args = argparse.Namespace(**json.load(open(os.path.join(args.model_dir, 'config.json'))))
        vocab = nlp.Vocab.from_json(open(os.path.join(args.model_dir, 'vocab.jsons')).read())
        val_dataset = read_dataset(args, 'test_file')
        val_data_loader = prepare_data_loader(args, val_dataset, vocab, test=True)
        model = NLIModel(len(vocab), model_args.embedding_size, model_args.hidden_size)
        model.load_parameters(os.path.join(args.model_dir, 'checkpoints', 'valid_best.params'), ctx=ctx)
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
