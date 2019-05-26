import os
import math
import shutil,time
import numpy as np
import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd,gluon,autograd,init
from dataloader import *
#####################################
###     variable need to modify   ###
#####################################
data_path = 'data/'
print_step = 200
ctx = mx.gpu(2)
isContextWord = True # choose ContextWord or WordContext
index = 3 # which task, see list list_task in line31 
#####################################

#####################################
###        hyper parameters       ###
#####################################
emb_size = 128
region_size = 7
region_radius = region_size//2
batch_size = 16
max_epoch = 20
learning_rate = 0.0001
list_max_sequence_length = [1024,256,256,256,1024,256]
list_n_classes = [5,2,5,4,10,14]
list_vocab_size = [124273,394385,356312,42783,361926,227863]
list_task = ['yelp_full','amazon_polarity','amazon_full','ag','yahoo','dbpedia']
n_classes = list_n_classes[index]
vocab_size = list_vocab_size[index]
max_sequence_length = list_max_sequence_length[index]
task_path = list_task[index]+'/'
####################################
class ContextWordNet(nn.HybridBlock):
    def __init__(self):
        super(ContextWordNet, self).__init__()
        with self.name_scope():
            self.embedding = nn.Embedding(vocab_size,emb_size)
            self.embedding_region = nn.Embedding(vocab_size*region_size,emb_size)
            self.max_pool = nn.GlobalMaxPool1D()
            self.dense = nn.Dense(n_classes)
    def hybrid_forward(self, F,aligned_seq,trimed_seq,mask):
        region_aligned_unit = self.embedding_region(aligned_seq)
        word_emb = self.embedding(trimed_seq).expand_dims(axis=2).broadcast_axes(axis=2,size=7)
        projected_emb = region_aligned_unit * word_emb
        feature = self.max_pool(projected_emb.transpose((0,1,3,2)).reshape((batch_size,-1,region_size))).reshape((batch_size,-1,emb_size))
        feature = feature*mask
        res = F.sum(feature, axis=1).reshape((batch_size,emb_size))
        res = self.dense(res)
        return res
class WordContextNet(nn.HybridBlock):
    def __init__(self):
        super(WordContextNet, self).__init__()
        with self.name_scope():
            self.embedding = nn.Embedding(vocab_size,region_size*emb_size)
            self.embedding_region = nn.Embedding(vocab_size,emb_size)
            self.max_pool = nn.GlobalMaxPool1D()
            self.dense = nn.Dense(n_classes)
    def hybrid_forward(self, F,aligned_seq,trimed_seq,mask):
        region_aligned_seq = aligned_seq.transpose((1, 0, 2))
        region_aligned_emb = self.embedding_region(region_aligned_seq).reshape((batch_size,-1,region_size,emb_size))
        context_unit = self.embedding(trimed_seq).reshape((batch_size,-1,region_size,emb_size))
        projected_emb = region_aligned_emb * context_unit
        feature = self.max_pool(projected_emb.transpose((0,1,3,2)).reshape((batch_size,-1,region_size))).reshape((batch_size,-1,emb_size))
        feature = feature*mask
        res = F.sum(feature, axis=1).reshape((batch_size,emb_size))
        res = self.dense(res)
        return res
def accuracy(output,label,batch_size):
    out = nd.argmax(output,axis=1)
    res = nd.sum(nd.equal(out.reshape((-1,1)),label))/batch_size
    return res
def batch_process(seq,isContextWord,ctx):
    seq = np.array(seq)
    aligned_seq = np.zeros((max_sequence_length - 2*region_radius,batch_size,region_size))
    for i in range(region_radius, max_sequence_length - region_radius):
        aligned_seq[i-region_radius] = seq[:,i-region_radius:i-region_radius+region_size]
    if isContextWord:
        unit_id_bias = np.array([i * vocab_size for i in range(region_size)])
        aligned_seq = aligned_seq.transpose((1,0,2))+unit_id_bias
    aligned_seq = nd.array(aligned_seq,ctx)
    batch_sequence = nd.array(seq,ctx)
    trimed_seq = batch_sequence[:, region_radius: max_sequence_length - region_radius]
    mask = nd.broadcast_axes(nd.greater(trimed_seq,0).reshape((batch_size,-1,1)),axis=2,size=128)
    return aligned_seq,nd.array(trimed_seq,ctx),mask
def evaluate(data,batch_size):
    test_loss = 0.0
    acc_test = 0.0
    cnt = 0
    for epoch_percent, batch_slots in batch_iter(data,batch_size,shuffle=False):
        batch_sequence, batch_label = zip(*batch_slots)
        batch_label = nd.array(batch_label,ctx)
        aligned_seq,trimed_seq,mask = batch_process(batch_sequence,isContextWord,ctx)
        output = net(aligned_seq,trimed_seq,mask)
        loss = SCE(output,batch_label)
        acc_test += accuracy(output,batch_label,batch_size)
        test_loss += nd.mean(loss)
        cnt = cnt+1
    return acc_test.asscalar()/cnt,test_loss.asscalar()/cnt

net = ContextWordNet() if isContextWord else WordContextNet()
SCE = mx.gluon.loss.SoftmaxCrossEntropyLoss()
net.initialize(init.Xavier(), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(),'adam', {'learning_rate': learning_rate})
data_test,data_train = load_data(data_path+task_path,max_sequence_length)
best_acc,global_step,train_loss,train_acc = 0,0,0,0
net.hybridize()
ctime = time.time()
print(ctx,list_task[index])
for epoch in range(max_epoch):
    for epoch_percent, batch_slots in batch_iter(data_train,batch_size,shuffle=True):
        batch_sequence, batch_label = zip(*batch_slots)
        global_step = global_step + 1
        batch_label = nd.array(batch_label,ctx)
        aligned_seq,trimed_seq,mask = batch_process(batch_sequence,isContextWord,ctx)
        with autograd.record():
            output = net(aligned_seq,trimed_seq,mask)
            loss = SCE(output,batch_label)
        loss.backward()
        trainer.step(batch_size)
        train_acc += accuracy(output,batch_label,batch_size)
        train_loss += nd.mean(loss)
        if global_step%print_step==0:
            print('%.4f %%'%epoch_percent,'train_loss:',train_loss.asscalar()/print_step,' train_acc:',train_acc.asscalar()/print_step,'time:',time.time()-ctime)
            train_loss,train_acc = 0,0
            ctime = time.time()
    test_acc,test_loss = evaluate(data_test,batch_size)
    if test_acc>best_acc:
        best_acc = test_acc
        net.save_parameters('params/regionemb_'+list_task[index])
    print('epoch %d done'%(epoch+1),'acc = %.4f,loss = %.4f'%(test_acc,test_loss))    
