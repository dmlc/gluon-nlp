import os
import math
import shutil,time
import numpy as np
import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from mxnet import init
from mxnet import nd
from mxnet.gluon.data import vision
from mxnet.gluon import nn
from mxnet import nd
from mxnet.gluon.model_zoo import vision as models
####################
list_max_sequence_length = [256,256,256,1024,256]
list_n_classes = [2,5,4,10,14]
list_vocab_size = [394385,356312,42783,361926,227863]
list_task = ['amazon_polarity','amazon_full','ag','yahoo','dbpedia']
max_epoch = 20
emb_size = 128
region_size = 7
batch_size = 16
learning_rate = 0.0001
#####################
#####################
base_path = 'data/'
print_step = 200
ctx = mx.gpu(0)
index = 3 # which task
n_classes = list_n_classes[index]
vocab_size = list_vocab_size[index]
max_sequence_length = list_max_sequence_length[index]
task_path = list_task[index]+'/'
####################
class Net(nn.HybridBlock):
    def __init__(self):
        super(Net, self).__init__()
        with self.name_scope():
            self.embedding = nn.Embedding(vocab_size,region_size*emb_size)
            self.embedding_region = nn.Embedding(vocab_size,emb_size)
            self.max_pool = nn.GlobalMaxPool1D()
            self.dense = nn.Dense(n_classes)
            self.dense1 = nn.Dense(max_sequence_length*2,activation='relu')
            self.dense2 = nn.Dense(1)
    def hybrid_forward(self, F, seq):
        region_radius = region_size//2
        aligned_seq = list(map(lambda i: F.slice(seq, begin=[None, i-region_radius], end=[None, i-region_radius+region_size]).asnumpy(), \
                    range(region_radius, seq.shape[1] - region_radius)))
        aligned_seq = nd.array(aligned_seq)
        region_aligned_seq = aligned_seq.transpose((1, 0, 2))
        region_aligned_emb = self.embedding_region(region_aligned_seq).reshape((batch_size,-1,region_size,emb_size))
        trimed_seq = seq[:, region_radius: seq.shape[1] - region_radius]
        context_unit = self.embedding(trimed_seq).reshape((batch_size,-1,region_size,emb_size))
        projected_emb = region_aligned_emb * context_unit
        feature = self.max_pool(projected_emb.transpose((0,1,3,2)).reshape((batch_size,-1,region_size))).reshape((batch_size,-1,emb_size))
        trimed_seq = seq[:,region_radius:seq.shape[1]-region_radius]
        mask = F.greater(trimed_seq,0).reshape((batch_size,-1,1))
        feature = mask*feature
        feature = feature.reshape((-1,emb_size))
        feature = self.dense(feature).reshape((batch_size,-1,n_classes)).transpose((0,2,1)).reshape((batch_size*n_classes,-1))
        #accumulation
        feature = F.expand_dims(feature,axis = 1)
        residual =  F.sum(feature,axis=2).reshape((batch_size,n_classes))
        res = self.dense2(self.dense1(feature)).reshape(batch_size*n_classes,1,-1).reshape((batch_size,n_classes))
        return res+residual
def read_data(path, slot_indexes, slots_lengthes, delim=';', pad=0, type_dict=None):
    n_slots = len(slot_indexes)
    slots = [[] for _ in range(n_slots)]
    if not type_dict:
        type_dict = {}
    with open(path, 'r') as fin:
        for i, line in enumerate(fin):
            items = line.strip().split(delim)    
            i += 1
            if i % 10000 == 1:
                print('read %d lines' % i)
            raw = []
            for index in slot_indexes:
                slot_value = items[index].split()
                tp = type_dict.get(index, int)
                raw.append([tp(x) for x in slot_value])
            for index in range(len(raw)):
                slots[index].append(pad_and_trunc(raw[index],slots_lengthes[index],pad=pad,sequence=slots_lengthes[index]>1))
    return slots
def batch_iter(data, batch_size, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield batch_num * 100.0 / num_batches_per_epoch,shuffled_data[start_index:end_index]
def pad_and_trunc(data, length, pad=0, sequence=False):
    if pad < 0:
        return data
    if sequence: 
        data.insert(0, pad)
        data.insert(0, pad)
        data.insert(0, pad)
        data.insert(0, pad)
    if len(data) > length:
        return data[:length]
    while len(data) < length:
        data.append(pad)
    return data
def load_data(path):
    print('Loading data...')
    indexes = [0,1]
    lengths = [1,max_sequence_length]
    print('Loading train...')
    train_path = base_path+task_path+'train.csv.id'
    labels_train, sequence_train = read_data(train_path, indexes, lengths)
    print('Loading test...')
    test_path = base_path+task_path+'test.csv.id'
    labels_test, sequence_test = read_data(test_path, indexes, lengths)
    return list(zip(sequence_test, labels_test)),list(zip(sequence_train, labels_train))
def evaluate(data):
    acc_test = mx.metric.Accuracy()
    test_loss = 0.0
    cnt = 0
    for epoch_percent, batch_slots in batch_iter(data,batch_size,shuffle=False):
        batch_sequence, batch_label = zip(*batch_slots)
        batch_sequence = nd.array(batch_sequence,ctx)
        batch_label = nd.array(batch_label,ctx)
        output = net(batch_sequence)
        loss = SCE(output,batch_label)
        acc_test.update(preds=[output],labels=[batch_label])
        test_loss += nd.mean(loss).asscalar()
        cnt = cnt+1
    return acc_test.get()[1],test_loss/cnt
net = Net()
SCE = mx.gluon.loss.SoftmaxCrossEntropyLoss()
net.initialize(init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(),'adam', {'learning_rate': learning_rate})
data_test,data_train = load_data(task_path)
best_acc,global_step,train_loss,train_acc = 0,0,0,0
ctime = time.time()
print(ctx,list_task[index])
for epoch in range(max_epoch):
    for epoch_percent, batch_slots in batch_iter(data_train,batch_size,shuffle=True):
        acc = mx.metric.Accuracy()
        batch_sequence, batch_label = zip(*batch_slots)
        global_step = global_step + 1
        batch_sequence = nd.array(batch_sequence,ctx)
        batch_label = nd.array(batch_label,ctx)
        with autograd.record():
            output = net(batch_sequence)
            loss = SCE(output,batch_label)
        loss.backward()
        trainer.step(batch_size)
        acc.update(preds=[output],labels=[batch_label])
        train_acc += acc.get()[1]
        train_loss += nd.mean(loss).asscalar()
        if global_step%print_step==0:
            print('%.4f %%'%epoch_percent,'train_loss:',train_loss/print_step,' train_acc:',train_acc/print_step,'time:',time.time()-ctime)
            ctime = time.time()
            train_loss,train_acc = 0,0
        if global_step%10000==0:
            test_acc,test_loss = evaluate(data_test)
            if test_acc>best_acc:
                net.save_parameters('params/TextEXAM_mlp_param_'+list_task[index])
            print('epoch %d '%(epoch+1),'acc = %.4f,loss = %.4f'%(test_acc,test_loss))
    test_acc,test_loss = evaluate(data_test)
    if test_acc>best_acc:
        net.save_parameters('params/TextEXAM_mlp_param_'+list_task[index])
    print('epoch %d done'%(epoch+1),'acc = %.4f,loss = %.4f'%(test_acc,test_loss))   
