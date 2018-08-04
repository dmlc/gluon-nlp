import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn, rnn

class Adaptivesoftmax(gluon.Block):
    def __init__(self, input_size, cutoff, reduce_factor=4):
        super(Adaptivesoftmax, self).__init__()
        
        self.input_size = input_size
        self.cutoff = cutoff
        self.output_size = cutoff[0] + len(cutoff) - 1

        self.head = nn.Dense(units=self.output_size, in_units=input_size, flatten=False)
        self.tail = nn.Sequential()
        
        for i in range(len(cutoff) - 1):
            if reduce_factor == 1:
                seq = nn.Dense(units=(cutoff[i + 1] - cutoff[i]), in_units=input_size, flatten=False)

            else:
                seq = nn.Sequential()
                seq.add(nn.Dense(units=(input_size // reduce_factor ** i), 
                                 in_units=input_size, flatten=False))
                seq.add(nn.Dense(units=(cutoff[i + 1] - cutoff[i]), 
                                 in_units=(input_size // reduce_factor ** i), flatten=False))

            self.tail.add(seq)
        
    def set_target(self, target):
        self.id = []
        target = target.asnumpy()

        for i in range(len(self.cutoff) - 1):
            mask_1 = (target >= self.cutoff[i])
            mask_2 = (target <= self.cutoff[i + 1])
            mask = mask_1 * mask_2
            
            mask = mask.reshape((mask.shape[1],mask.shape[0]))
            if True in mask:
                self.id.append(np.where(mask[0])[0])

            else:
                self.id.append(None)
                
    def remap_target(self, target):
        target = target.asnumpy()
        new_target = []
        new_target.append(np.copy(target))

        for i in range(len(self.cutoff) - 1):
            mask_1 = (target >= self.cutoff[i])
            mask_2 = (target <= self.cutoff[i + 1])
            mask = mask_1 * mask_2            
            new_target[0][mask] = self.cutoff[0] + i

            if True in mask:
                new_target.append(target[mask] - self.cutoff[i])

            else:
                new_target.append(None)

        return new_target
        
                       
    def forward(self, input, target):
        output_head = self.head(input)
        nnloss = 0
        self.target = target
        context = input.context

        if self.target is not None:
            self.set_target(self.target)
            self.target = self.remap_target(self.target)
            
        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        nnloss = nnloss + mx.nd.sum(loss(output_head, mx.nd.array(self.target[0]).as_in_context(context))) 
        
        for i in range(len(self.id)):
            if self.id[i] is not None:
                id_select = np.array(self.id[i])
                output_tail = self.tail[i](input[id_select])
                nnloss = nnloss + mx.nd.sum(loss(output_tail,  mx.nd.array(self.target[i+1]).as_in_context(context)))          
        
        nnloss = nnloss / (len(target))    
        
        return nnloss     
    
    def log_prob(self, input):  
        head_out = self.head(input)
        target_size = head_out.shape[0]
        prob = mx.nd.zeros((target_size, self.cutoff[-1]))
            
        lsm_head = mx.nd.log_softmax(head_out, axis=1)
        prob[:, : self.cutoff[0]] = lsm_head[:, : self.cutoff[0]]
        
        for i in range(len(self.tail)):
            split = lsm_head[:, self.cutoff[0] + i]
            split = split.expand_dims(1)
            tail_out = self.tail[i](input)
            if i==10:
                print (tail_out[0])
                print ('tail loss:', mx.nd.log_softmax(tail_out, axis=1)[-1][3395])
            lsm_tail = mx.nd.log_softmax(tail_out, axis=1) + split.broadcast_to((tail_out.shape[0], tail_out.shape[1]))
            prob[:, self.cutoff[i] : self.cutoff[i + 1]] = lsm_tail
        
        return prob                 
