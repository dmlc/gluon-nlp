import numpy as np
import mxnet as mx
import argparse, math
from mxnet.gluon import nn, rnn
from mxnet import gluon, autograd
import os
import time
import pickle

from model import LanguageModel
from adaptive_softmax import *


parser = argparse.ArgumentParser(description='Benchmark for Adaptive Softmax')
parser.add_argument('--adaptive_softmax', type=bool, default=True,
        help=('Whether use Adaptive Softmax or not , '
            'False for common full softmax'))
parser.add_argument('--train_file_path', type=str,
                    default='./data/text8.train.pkl')
parser.add_argument('--test_file_path', type=str,
                    default='./data/text8.test.pkl')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--bptt', type=int, default=20)
parser.add_argument('--hidden_size', type=int, default=512)
parser.add_argument('--clip_global_norm_rate', type=float, default=0.25)
parser.add_argument('--drop_rate', type=float, default=0.25)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--learning_rate_decay', type=float, default=1e-5)
parser.add_argument('--pass_num', type=int, default=5)
parser.add_argument('--emsize', type=int, default=512,
                    help=('size of word embeddings'))
parser.add_argument('--nlayers', type=int, default=1,
                    help=('number of layers'))
parser.add_argument('--log_interval', type=int, default=200,
                    help=('report interval'))

parser.add_argument('--cutoff', type=str, default="2000 10000")

args = parser.parse_args()

adaptive_softmax = args.adaptive_softmax
batch_size = args.batch_size
bptt = args.bptt
nhid = args.hidden_size
clip = args.clip_global_norm_rate
dropout = args.drop_rate
lr = args.learning_rate
wd = args.weight_decay
lrd = args.learning_rate_decay
epochs = args.pass_num
emsize = args.emsize
nlayers = args.nlayers
log_interval = args.log_interval

cutoff = []
for element in args.cutoff.split():
    cutoff.append(int(element))

context = mx.gpu(0)

#load the data
train_file_path = args.train_file_path
test_file_path = args.test_file_path

with open(train_file_path, 'rb') as f:
    data = pickle.load(f)

input = data['input']
label = data['label']
input = mx.nd.array(input)
label = mx.nd.array(label)

vocab = len(data['worddic'])

with open(test_file_path, 'rb') as f:
    data_test = pickle.load(f)

input_test = data_test['input']
label_test = data_test['label']
input_test = mx.nd.array(input_test)
label_test = mx.nd.array(label_test)

ntokens = vocab

model = LanguageModel(vocab_size=ntokens, num_embed=emsize, num_hidden=nhid, num_layers=nlayers, dropout=dropout,
                       adaptive_softmax=adaptive_softmax, cutoff=cutoff)
model.initialize(mx.init.Xavier(), ctx=context)
trainer = gluon.Trainer(model.collect_params(), 'AdaGrad',
                        {'learning_rate': lr,
                         'wd': wd})

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden

def eval(data_source, target_source):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)
    loss = gluon.loss.SoftmaxCrossEntropyLoss(from_logits=True)
    for (data, target) in zip(data_source, target_source):
        data = data.as_in_context(context).T
        target = target.as_in_context(context).T.reshape((-1, 1))
        hidden = detach(hidden)
        if adaptive_softmax:
            prob, hidden = model.log_prob(data, hidden)
            L = loss(prob.as_in_context(context), target)
        else:
            nnloss, hidden = model(data, hidden, target)
            L = nnloss
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal

def train():
    for epoch in range(epochs):
        start_time = time.time()
        total_L = 0.0
        hidden = model.begin_state(func=mx.nd.zeros, batch_size=batch_size, ctx=context)
        i = 0
        
        
        lr_i = lr / (1 + lrd * epoch)
        trainer.set_learning_rate(lr_i)
        
        for (data, target) in zip(input, label):
            data = data.as_in_context(context).T
            target = target.as_in_context(context).T.reshape((-1, 1))
            hidden = detach(hidden)
            with autograd.record():
                nnloss, hidden = model(data, hidden, target)
                L = nnloss
                L.backward()
            
            grads = [p.grad(context) for p in model.collect_params().values()]
            gluon.utils.clip_global_norm(grads, clip)

            trainer.step(1, ignore_stale_grad=True)
            total_L += L.asscalar()
            
            
            i+=1
            if i % log_interval == 0 and i > 0:
                cur_L = total_L / log_interval
                print('[Epoch %d Batch %d] loss %.2f, ppl %.2f'%(
                    epoch, i, cur_L, math.exp(cur_L)))
                total_L = 0.0
        
        test_L = eval(input_test, label_test)

        print('[Epoch %d] time cost %.2fs, test loss %.2f, test ppl %.2f'%(
            epoch, time.time()-start_time, test_L, np.exp(test_L)))

train()






