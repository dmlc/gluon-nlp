# pylint: disable=E1101,R0914
"""
main_infra.py
Main of NLI script in gluon-nlp. Infra-sentence attention model.

Copyright 2018 Mengxiao Lin <linmx0130@gmail.com>
"""

import os
import argparse
import numpy as np
import gluonnlp as nlp
import mxnet as mx
from mxnet import gluon, autograd, nd
from decomposable_atten import DecomposableAtten, IntraSentenceAtten
from nlidataset import NLIDataset
from utils import tokenize_and_index

LABEL_TO_IDX = {'neutral': 0, 'contradiction': 1, 'entailment': 2}

class Network(gluon.Block):
    """
    Network of Infra-attention NLI.
    """
    def __init__(self, word_embed_size, hidden_size, **kwargs):
        super(Network, self).__init__(**kwargs)
        self.word_embed_size = word_embed_size
        self.hidden_size = hidden_size
        with self.name_scope():
            self.lin_proj = gluon.nn.Dense(hidden_size, in_units=word_embed_size, activation='relu')
            self.intra_atten = IntraSentenceAtten(hidden_size, hidden_size)
            self.model = DecomposableAtten(hidden_size*2, hidden_size, 3)

    def forward(self, *args):
        """
        Model forward definition
        """
        sentence1 = args[0]
        sentence2 = args[1]
        batch_size1, length1, dimension1 = sentence1.shape
        batch_size2, length2, dimension2 = sentence2.shape
        assert batch_size1 == batch_size2
        assert dimension1 == dimension2
        batch_size = batch_size1
        dimension = dimension1
        feature1 = self.lin_proj(
            sentence1.reshape(batch_size * length1, dimension)
            ).reshape(batch_size, length1, self.hidden_size)
        feature1 = mx.nd.concat(feature1, self.intra_atten(feature1), dim=-1)
        feature2 = self.lin_proj(
            sentence2.reshape(batch_size * length2, dimension)
            ).reshape(batch_size, length2, self.hidden_size)
        feature2 = mx.nd.concat(feature2, self.intra_atten(feature2), dim=-1)
        pred = self.model(feature1, feature2)
        return pred

def parse_args():
    """
    Parsing arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',
                        help='root of data file', default='./data')
    parser.add_argument('--train_set',
                        help='training set file', default='snli_1.0/snli_1.0_train.txt')
    parser.add_argument('--dev_set',
                        help='development set file', default='snli_1.0/snli_1.0_dev.txt')
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

    return parser.parse_args()

def train_network(model, train_set, embedding, ctx, args):
    """
    Train the network.
    """
    model.collect_params().initialize(
        mx.init.Xavier(), force_reinit=True, ctx=ctx)
    celoss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(model.collect_params(), 'adagrad',
                            {'learning_rate': args.lr, 'wd': args.weight_decay})
    batch_size = args.batch_size
    print_period = args.print_period
    if not os.path.exists(args.checkpoints):
        os.mkdir(args.checkpoints)
    emb_layer = mx.gluon.nn.Embedding(len(embedding.idx_to_token),
                                      embedding.idx_to_vec[0].size)
    emb_layer.initialize()
    emb_layer.weight.set_data(embedding.idx_to_vec)
    padder = nlp.data.batchify.Pad()
    model.save_parameters(
        os.path.join(args.checkpoints, 'epoch-0.gluonmodel'))
    for epoch in range(1, args.maximum_iter + 1):
        access_key = list(range(len(train_set)))
        np.random.shuffle(access_key)
        idx = 0
        counter = print_period
        acc_loss = nd.array([0.], ctx=ctx)
        acc_acc = nd.array([0.], ctx=ctx)
        while idx + batch_size <= len(access_key):
            sen1 = []
            sen2 = []
            label = []
            for i in range(batch_size):
                item = train_set[access_key[idx + i]]
                sen1.append(tokenize_and_index(item.sentence1, embedding))
                sen2.append(tokenize_and_index(item.sentence2, embedding))
                label.append(LABEL_TO_IDX[item.gold_label])
            idx += batch_size
            sen1 = emb_layer(padder(sen1)).as_in_context(ctx)
            sen2 = emb_layer(padder(sen2)).as_in_context(ctx)
            label = nd.array(label, dtype=int).as_in_context(ctx)
            with autograd.record():
                yhat = model(sen1, sen2)
                cur_loss = celoss(yhat, label).mean()
            pred = yhat.argmax(axis=1)
            cur_acc = (pred == label.astype(np.float32)).mean()
            cur_loss.backward()
            trainer.step(batch_size)
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

def prepare_dataset(args, dataset):
    """
    Preparing datasets and push them into a list.
    Illegal data items are removed.
    """
    train_set = NLIDataset(os.path.join(args.data_root, vars(args)[dataset]))
    clean_train_set = []
    for item in train_set:
        if item.gold_label in LABEL_TO_IDX:
            clean_train_set.append(item)
    return clean_train_set

if __name__ == '__main__':
    ARGS = parse_args()
    GLOVE = nlp.embedding.create(ARGS.embedding, source=ARGS.embedding_source)
    if ARGS.mode == 'train':
        TRAIN_SET = prepare_dataset(ARGS, 'train_set')
        NET = Network(ARGS.embedding_size, 200)
        train_network(NET, TRAIN_SET, GLOVE, mx.gpu(), ARGS)
    elif ARGS.mode == 'test':
        TEST_SET = prepare_dataset(ARGS, 'dev_set')
        NET = Network(ARGS.embedding_size, 200)
        CTX = mx.gpu()
        NET.load_parameters(ARGS.model, ctx=CTX)
        test_network(NET, TEST_SET, GLOVE, CTX)
    else:
        raise NotImplementedError()
