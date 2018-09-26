import mxnet as mx
from mxnet import gluon, autograd, nd
from decomposable_atten import DecomposableAtten
from nlidataset import NLIDataset, NLIDataItem
import gluonnlp as nlp
import argparse
import os
from utils import *
import numpy as np

label_to_idx = {'neutral': 0, 'contradiction': 1, 'entailment': 2}

class Network(gluon.Block):
    def __init__(self, word_embed_size, hidden_size, **kwargs):
        super(Network, self).__init__(**kwargs)
        self.word_embed_size = word_embed_size
        self.hidden_size = hidden_size
        with self.name_scope():
            self.lin_proj = gluon.nn.Dense(hidden_size, in_units=word_embed_size, activation="relu")
            self.model = DecomposableAtten(hidden_size, hidden_size, 3)

    def forward(self, sentence1, sentence2):
        B, L1, D = sentence1.shape
        B, L2, D = sentence2.shape
        # s1, s2 = sentence1, sentence2
        s1 = self.lin_proj(sentence1.reshape(B * L1, D)).reshape(B, L1, self.hidden_size)
        s2 = self.lin_proj(sentence2.reshape(B * L2, D)).reshape(B, L2, self.hidden_size)
        pred = self.model(s1, s2)
        return pred

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help="root of data file", default="./data")
    parser.add_argument('--train_set', help="training set file", default="snli_1.0/snli_1.0_train.txt")
    parser.add_argument('--dev_set', help="development set file", default="snli_1.0/snli_1.0_dev.txt")
    parser.add_argument('--batch_size', help="batch_size", default=32, type=int)
    parser.add_argument('--print_period', help="the interval of two print", default=20, type=int)
    parser.add_argument('--checkpoints', help="path to save checkpoints", default="checkpoints")
    parser.add_argument('--model', help="model file to test, only for test mode", default=None)
    parser.add_argument('--mode', help="train or test", default="train")
    return parser.parse_args()

def train_network(model, train_set, embedding, ctx, args):
    model.collect_params().initialize(mx.init.Xavier(), force_reinit=True, ctx=ctx)
    celoss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(model.collect_params(), 'adagrad',
            {'learning_rate': 0.01, 'wd': 0.0001})
    acc_loss = nd.array([0,])
    acc_acc = nd.array([0,])
    batch_size = args.batch_size
    print_period = args.print_period
    if not os.path.exists(args.checkpoints):
        os.mkdir(args.checkpoints)

    checkpoints_path = os.path.join(args.checkpoints, "epoch-0.gluonmodel")
    model.save_parameters(checkpoints_path)
    for epoch in range(1, 200):
        access_key = list(range(len(train_set)))
        np.random.shuffle(access_key)        
        idx = 0
        counter = print_period
        while idx + batch_size <= len(access_key):
            sen1 = []
            sen2 = []
            label = []
            for i in range(batch_size):
                item = train_set[access_key[idx + i]]
                s1 = fetch_embedding_of_sentence(item.sentence1, embedding)
                sen1.append(s1)
                s2 = fetch_embedding_of_sentence(item.sentence2, embedding)
                sen2.append(s2)
                label.append(label_to_idx[item.gold_label])
            idx += batch_size

            sen1 = pad_sentences(sen1).as_in_context(ctx)
            sen2 = pad_sentences(sen2).as_in_context(ctx)
            label = nd.array(label, dtype=int).as_in_context(ctx)
            with autograd.record():
                yhat = model(sen1, sen2)
                L = celoss(yhat, label).mean()
            pred = yhat.argmax(axis=1)
            cur_acc = (pred == label.astype(np.float32)).mean()
            L.backward()
            trainer.step(batch_size)
            acc_loss = acc_loss * 0.95 + 0.05 * L.as_in_context(acc_loss.context)
            acc_acc = acc_acc * 0.95 + 0.05 * cur_acc.as_in_context(acc_loss.context)
            counter -= 1
            if counter == 0:
                print("Epoch ", epoch, "Loss=",acc_loss.asscalar(), "Acc=", acc_acc.asscalar())
                counter = print_period
        checkpoints_path = os.path.join(args.checkpoints, "epoch-%d.gluonmodel" % epoch)
        model.save_parameters(checkpoints_path)
        print("Epoch ", epoch, " saved to ", checkpoints_path)

def test_network(model, test_set, embedding, ctx, args):
    acc = nd.array([0.], ctx=ctx)
    counter = 0 
    for idx in range(len(test_set)):
        sen1 = []
        sen2 = []
        label = []
        item = test_set[idx]
        s1 = fetch_embedding_of_sentence(item.sentence1, embedding)
        sen1.append(s1)
        s2 = fetch_embedding_of_sentence(item.sentence2, embedding)
        sen2.append(s2)
        label.append(label_to_idx[item.gold_label])

        sen1 = pad_sentences(sen1).as_in_context(ctx)
        sen2 = pad_sentences(sen2).as_in_context(ctx)
        label = nd.array(label, dtype=int).as_in_context(ctx)
        with autograd.predict_mode():
            yhat = model(sen1, sen2)
            pred = yhat.argmax(axis=1)
        cur_acc = (pred == label.astype(np.float32)).sum()
        acc += cur_acc
        counter += 1
    acc = acc / counter
    print("Acc=", acc.asscalar())

def prepare_dataset(args, dataset):
    train_set = NLIDataset(os.path.join(args.data_root, vars(args)[dataset]))
    clean_train_set = []
    for item in train_set:
        if item.gold_label in label_to_idx:
            clean_train_set.append(item)
    return clean_train_set

if __name__=="__main__":    
    glove = nlp.embedding.create('glove', source='glove.6B.300d')
    args = parse_args()
    if args.mode == "train":
        clean_train_set = prepare_dataset(args, 'train_set')
        ctx = mx.gpu()
        model = Network(300, 200)    
        train_network(model, clean_train_set, glove, ctx, args)
    elif args.mode == "test":
        test_set = prepare_dataset(args, 'dev_set')
        ctx = mx.gpu()
        model = Network(300, 200)
        model.load_parameters(args.model, ctx=ctx)
        test_network(model, test_set, glove, ctx, args)
    else:
        raise NotImplementedError()
