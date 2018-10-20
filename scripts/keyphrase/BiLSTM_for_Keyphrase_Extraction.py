# coding:utf-8
'''
params to reproduce the results in log files:
glove-50: embedding_dim=50, logging_path="./log/log_glove_50.txt",model_path='./model/glove_50.model',hidden=100,lstm_dropout=0.1,
          learning_rate=0.001,epochs=50,seq_len=500,batch_size=100,dropout=0.3
glove-100: embedding_dim=50, logging_path="./log/log_glove_50.txt",model_path='./model/glove_50.model',hidden=100,lstm_dropout=0.1,
          learning_rate=0.001,epochs=100,seq_len=500,batch_size=100,dropout=0.3
glove-200: embedding_dim=50, logging_path="./log/log_glove_50.txt",model_path='./model/glove_50.model',hidden=150,lstm_dropout=0.25,
          learning_rate=0.001,epochs=100,seq_len=500,batch_size=100,dropout=0.3
glove-300: embedding_dim=50, logging_path="./log/log_glove_50.txt",model_path='./model/glove_50.model',hidden=150,lstm_dropout=0.1,
          learning_rate=0.001,epochs=50,seq_len=500,batch_size=100,dropout=0.3      
'''
import os, io, sys, re
from collections import Counter
import mxnet as mx
import numpy as np
from mxnet.contrib import text
from mxnet import gluon, autograd,nd, ndarray
import gluonnlp as nlp
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--embedding_dim', type=int, default=100, help='glove embedding dim')
parser.add_argument('--logging_path', default="./log/log_glove_100.txt", help='logging file path')
parser.add_argument('--model_path', default='./model/glove_100.model', help='saving model in model_path')
parser.add_argument('--hidden', type=int, default=300, help='hidden units in bilstm')
parser.add_argument('--lstm_dropout', type=float, default=0.5, help='dropout applied to lstm layers ')
parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
parser.add_argument('--epochs', type=int, default=50, help='training epochs')
parser.add_argument('--seq_len', type=int, default=500, help='max length of sequences')
parser.add_argument('--batch_size', type=int, default=100, help='batch size')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout applied to fully connected layers')
args = parser.parse_args()
print(args)

# logging config
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler(args.logging_path)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# data preprocessing
def data_process():
    directory = "./inspec/all"
    articles = os.listdir(directory)
    all_data = []
    fullText = []

    text_articles = []
    for article in articles:
        if article.endswith(".abstr"):
            text_articles.append(article)
    text_articles.sort()

    keyp_articles = []
    for article in articles:
        if article.endswith('.uncontr'):
            keyp_articles.append(article)
    keyp_articles.sort()

    for article_ID in range(len(text_articles)):
        a = text_articles[article_ID].split('.')[0]
        b = keyp_articles[article_ID].split('.')[0]
        if a==b:
            articleFile = io.open(directory + "/" + text_articles[article_ID], 'r')
            text = articleFile.read().strip()
            text = re.sub('[!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~]+', '', str(text))
            text = text.replace('\\n',' ').replace('\\t',' ').replace('\\r',' ')
            words1 = text.split()
            words = [x.lower() for x in words1]
            fullText += words

            keyphraseFile = io.open(directory + "/" + keyp_articles[article_ID], 'r')
            keyphrases1 = keyphraseFile.read().strip().replace('; ',' ')
            keyphrases = [x.lower() for x in keyphrases1.split()]

            tag = []
            for i in range(len(words)):
                if words[i] not in keyphrases:
                    tag.append(0)  # NO_KP
                elif words[i] in keyphrases and words[i-1] not in keyphrases:
                    tag.append(1)  # BEGIN_KP
                else:
                    tag.append(2)  # INSIDE_KP

            all_data.append((words, tag))

    return all_data, fullText
all_data, fullText = data_process()

# add external glove word embedding
counter = nlp.data.count_tokens(fullText)
print(len(counter))

print('---glove embedding----')
my_vocab = nlp.Vocab(counter)
my_embedding = nlp.embedding.create('glove', source='glove.6B.'+ str(args.embedding_dim) + 'd')
my_vocab.set_embedding(my_embedding)
inputdim, outputemb = my_vocab.embedding.idx_to_vec.shape
print(inputdim,outputemb)  # (18199,emb_dim)

# word to idx
dataset = []
tag = []
seq_len = args.seq_len
for i,(words, label) in enumerate(all_data):
    index = [my_embedding.token_to_idx[x] for x in words]
    if len(index)<seq_len:  # padding 0
        index += [0]*(seq_len-len(index))
        label += [0]*(seq_len-len(label))
    else:
        index = index[:seq_len]
        label = label[:seq_len]
    dataset.append(index)
    tag.append(label)
dataset = np.array(dataset)
tag = np.array(tag)

# dataset
train_data = gluon.data.ArrayDataset(dataset[:1000], tag[:1000])
val_data = gluon.data.ArrayDataset(dataset[1000:1500], tag[1000:1500])
test_data = gluon.data.ArrayDataset(dataset[1500:], tag[1500:])

batch_size = args.batch_size
train_iter = gluon.data.DataLoader(train_data, batch_size=batch_size)
val_iter = gluon.data.DataLoader(val_data, batch_size=batch_size)
test_iter = gluon.data.DataLoader(test_data, batch_size=batch_size)


def evaluate_accu(data_iter, net):
    acc = 0.
    for i, (data, label) in enumerate(data_iter):
        output = net(data)
        output = output.reshape((-1, 3))
        acc += accuracy(output, label)
    return acc / (i + 1)


def accuracy(output, label):
    pred = mx.nd.argmax(output, axis=1).asnumpy().flatten()
    label = label.asnumpy().flatten()

    cnt = 0
    for i in range(len(pred)):
        if pred[i] == label[i]:
            cnt += 1
    accu = 1.0 * cnt / len(pred)
    return accu


def train(hidden, lstm_dropout, learning_rate, epochs):
    # train net
    logger.info('----Start training-----')
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Embedding(inputdim, outputemb))
        net.add(gluon.nn.Dropout(args.dropout))
        net.add(gluon.rnn.LSTM(hidden_size=hidden//2, num_layers=1, layout='NTC', bidirectional=True, dropout=lstm_dropout))
        net.add(gluon.nn.Dense(3, flatten=False))
    logger.info(net)

    net.collect_params().initialize(mx.init.Normal(sigma=0.1))
    net[0].weight.set_data(my_vocab.embedding.idx_to_vec)

    softmax_cross_entropy = mx.gluon.loss.SoftmaxCrossEntropyLoss(from_logits=False)

    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate})

    for e in range(epochs):
        train_loss = 0.
        train_acc = 0.
        for i, (data, label) in enumerate(train_iter):
            with autograd.record():
                output = net(data)
                output = output.reshape((-1, 3))
                label = label.reshape((-1, 1))
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size=batch_size)

            train_loss += nd.mean(loss).asscalar()
            train_acc += accuracy(output, label)
        val_accuracy = evaluate_accu(val_iter, net)
        logger.info("Epoch {}. Current Loss: {}. train accu: {}. val accu: {}." \
              .format(e, train_loss / len(train_iter), train_acc / len(train_iter), val_accuracy))

    model_path = args.model_path
    net.save_params(model_path)
    return model_path


def evaluate(test_iter, net2):
    correct = 0
    extract = 0
    standard = 0
    for i, (data, label) in enumerate(test_iter):
        output = net2(data).reshape((-1,3))
        pred = nd.argmax(output,axis=1).asnumpy().flatten()
        label = label.asnumpy().flatten()

        pred2 = [str(int(x)) for x in pred]
        label2 = [str(x) for x in label]

        predstr = ''.join(pred2).replace('0',' ').split()
        labelstr = ''.join(label2).replace('0',' ').split()

        extract += len(predstr)
        standard += len(labelstr)

        i = 0
        while i < len(label):
            if label[i] != 0:
                while i < len(label) and label[i] != 0 and pred[i] == label[i]:
                    i += 1
                if i < len(label) and label[i] == pred[i] == 0 or i == len(label):
                    correct += 1
            i += 1
    precision = 1.0 * correct/extract
    recall = 1.0 * correct/standard
    f1 = 2*precision*recall/(precision+recall)
    return precision, recall, f1,correct,extract,standard


def test(model_path, hidden, lstm_dropout):
    logger.info('----Start testing-----')
    net2 = gluon.nn.Sequential()
    with net2.name_scope():
        net2.add(gluon.nn.Embedding(inputdim, outputemb))
        net2.add(gluon.nn.Dropout(args.dropout))
        net2.add(gluon.rnn.LSTM(hidden_size=hidden//2, num_layers=1,layout='NTC',bidirectional=True, dropout=lstm_dropout))
        net2.add(gluon.nn.Dense(3,flatten=False))
    logger.info(net2)

    net2.load_params(args.model_path, mx.cpu())

    # precision= correct/extract, recall = correct/standard, f1 = 2*p*r/(p+r)
    precision, recall, f1, correct, extract, standard = evaluate(test_iter, net2)
    logger.info("precision {}. recall {}. f1 {}. ".format(precision, recall, f1))
    logger.info("correct {}. extract {}. standard {}. ".format(correct, extract, standard))


if __name__=='__main__':
    model_path = train(args.hidden, args.lstm_dropout, args.learning_rate, args.epochs)
    test(model_path,args.hidden,args.lstm_dropout)
