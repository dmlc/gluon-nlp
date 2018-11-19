# pylint: disable=E1101,R0914
"""
main.py
Main of NLI script in gluon-nlp. Intra-sentence attention model.

Copyright 2018 Mengxiao Lin <linmx0130@gmail.com>
"""

import os
import argparse
import json
import numpy as np
import gluonnlp as nlp
import gluonnlp.data.batchify as btf
import mxnet as mx
from mxnet import gluon, autograd, nd
from decomposable_attention import DecomposableAttention, IntraSentenceAttention
from dataset import read_dataset, prepare_data_loader, build_vocab
from utils import tokenize_and_index

LABEL_TO_IDX = {'neutral': 0, 'contradiction': 1, 'entailment': 2}

class NLIModel(gluon.HybridBlock):
    """
    A Decomposable Attention Model for Natural Language Inference
    using intra-sentence attention.
    Arxiv paper: https://arxiv.org/pdf/1606.01933.pdf
    """
    def __init__(self, vocab_size, word_embed_size, hidden_size, **kwargs):
        super(NLIModel, self).__init__(**kwargs)
        self.word_embed_size = word_embed_size
        self.hidden_size = hidden_size
        with self.name_scope():
            self.word_emb = mx.gluon.nn.Embedding(vocab_size, word_embed_size)
            self.lin_proj = gluon.nn.Dense(hidden_size, in_units=word_embed_size, activation='relu', flatten=False)
            self.intra_attention = IntraSentenceAttention(hidden_size, hidden_size)
            self.model = DecomposableAttention(hidden_size*2, hidden_size, 3)

    def hybrid_forward(self, F, sentence1, sentence2):
        """
        Model forward definition
        """
        feature1 = self.lin_proj(self.word_emb(sentence1))
        feature1 = F.concat(feature1, self.intra_attention(feature1), dim=-1)
        feature2 = self.lin_proj(self.word_emb(sentence2))
        feature2 = F.concat(feature2, self.intra_attention(feature2), dim=-1)
        pred = self.model(feature1, feature2)
        return pred

def parse_args():
    """
    Parsing arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
                        help='use CPU')
    parser.add_argument('--data_root',
                        help='root of data file', default='./data')
    parser.add_argument('--train_set',
                        help='training set file', default='snli_1.0/snli_1.0_train.txt')
    parser.add_argument('--dev_set',
                        help='development set file', default='snli_1.0/snli_1.0_dev.txt')
    parser.add_argument('--max-num-examples', type=int, default=-1,
                        help='maximum number of examples to load (for debugging)')
    parser.add_argument('--batch_size',
                        help='batch size', default=32, type=int)
    parser.add_argument('--print_period',
                        help='the interval of two print', default=20, type=int)
    parser.add_argument('--checkpoints',
                        help='path to save checkpoints', default='checkpoints')
    parser.add_argument('--model',
                        help='model file to test, only for test mode', default=None)
    parser.add_argument('--mode',
                        help='train or test', default='train')
    parser.add_argument('--lr',
                        help='learning rate', default=0.025, type=float)
    parser.add_argument('--weight_decay',
                        help='weight decay', default=0.0001, type=float)
    parser.add_argument('--maximum_iter',
                        help='maximum number of iterations to train',
                        default=200, type=float)
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

    return parser.parse_args()

def train_model(model, data_loader, embedding, ctx, args):
    """
    Train the model.
    """
    model.hybridize()

    # Initialziation
    model.collect_params().initialize(
        mx.init.Xavier(), force_reinit=True, ctx=ctx)
    model.word_emb.weight.set_data(embedding.idx_to_vec)

    celoss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(model.collect_params(), 'adagrad',
                            {'learning_rate': args.lr, 'wd': args.weight_decay})
    print_period = args.print_period
    if not os.path.exists(args.checkpoints):
        os.makedirs(args.checkpoints)
    model.save_parameters(
        os.path.join(args.checkpoints, 'epoch-0.gluonmodel'))

    for epoch in range(1, args.maximum_iter + 1):
        counter = print_period
        acc_loss = nd.array([0.], ctx=ctx)
        acc_acc = nd.array([0.], ctx=ctx)
        for batch_id, example in enumerate(data_loader):
            s1, s2, label = example
            s1 = s1.as_in_context(ctx)
            s2 = s2.as_in_context(ctx)
            label = label.as_in_context(ctx)

            with autograd.record():
                output = model(s1, s2)
                cur_loss = celoss(output, label).mean()
            pred = output.argmax(axis=1)
            cur_acc = (pred == label.astype(np.float32)).mean()
            cur_loss.backward()
            trainer.step(args.batch_size)
            acc_loss += cur_loss
            acc_acc += cur_acc
            counter -= 1
            if counter == 0:
                acc_loss /= print_period
                acc_acc /= print_period
                print('Epoch ', epoch, 'Loss=', acc_loss.asscalar(), 'Acc=', acc_acc.asscalar())
                counter = print_period
                acc_loss = nd.array([0.], ctx=ctx)
                acc_acc = nd.array([0.], ctx=ctx)
        checkpoints_path = os.path.join(args.checkpoints, 'epoch-%d.gluonmodel' % epoch)
        model.save_parameters(checkpoints_path)
        print('Epoch ', epoch, ' saved to ', checkpoints_path)

def test_network(model, test_set, embedding, ctx):
    """
    Test a network.
    """
    acc = nd.array([0.], ctx=ctx)
    counter = 0
    emb_layer = mx.gluon.nn.Embedding(len(embedding.idx_to_token),
                                      embedding.idx_to_vec[0].size)
    emb_layer.initialize()
    emb_layer.weight.set_data(embedding.idx_to_vec)
    padder = nlp.data.batchify.Pad()
    for item in test_set:
        sen1 = []
        sen2 = []
        label = []
        feature1 = tokenize_and_index(item.sentence1, embedding)
        sen1.append(feature1)
        feature2 = tokenize_and_index(item.sentence2, embedding)
        sen2.append(feature2)
        label.append(LABEL_TO_IDX[item.gold_label])

        sen1 = emb_layer(padder(sen1)).as_in_context(ctx)
        sen2 = emb_layer(padder(sen2)).as_in_context(ctx)
        label = nd.array(label, dtype=int).as_in_context(ctx)
        with autograd.predict_mode():
            yhat = model(sen1, sen2)
            pred = yhat.argmax(axis=1)
        cur_acc = (pred == label.astype(np.float32)).sum()
        acc += cur_acc
        counter += 1
    acc = acc / counter
    print('Acc=', acc.asscalar())

def main(args):
    train_dataset = read_dataset(args, 'train_set')
    dev_dataset = read_dataset(args, 'dev_set')

    vocab = build_vocab(train_dataset)
    glove = nlp.embedding.create(args.embedding, source=args.embedding_source)
    vocab.set_embedding(glove)

    train_data_loader = prepare_data_loader(args, train_dataset, vocab)
    dev_data_loader = prepare_data_loader(args, dev_dataset, vocab, test=True)

    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu()

    model = NLIModel(len(vocab), args.embedding_size, args.hidden_size)

    if args.mode == 'train':
        train_model(model, train_data_loader, vocab.embedding, ctx, args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    #if ARGS.mode == 'train':
    #    TRAIN_SET = prepare_dataset(ARGS, 'train_set')
    #    NET = NLIModel(ARGS.embedding_size, 200)
    #    train_model(NET, TRAIN_SET, GLOVE, mx.gpu(), ARGS)
    #elif ARGS.mode == 'test':
    #    TEST_SET = prepare_dataset(ARGS, 'dev_set')
    #    NET = NLIModel(ARGS.embedding_size, 200)
    #    CTX = mx.gpu()
    #    NET.load_parameters(ARGS.model, ctx=CTX)
    #    test_network(NET, TEST_SET, GLOVE, CTX)
    #else:
    #    raise NotImplementedError()
