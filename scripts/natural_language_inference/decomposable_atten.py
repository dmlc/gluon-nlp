import mxnet
from mxnet import nd, autograd, gluon
import mxnet.gluon.nn as nn

class IntraSentenceAtten(gluon.Block):
    def __init__(self, inp_size: int, hidden_size: int, max_length=10, **kwargs):
        super(IntraSentenceAtten, self).__init__(**kwargs)
        self.max_length = max_length
        self.hidden_size = hidden_size
        with self.name_scope():
            self.f = nn.Sequential()
            self.f.add(nn.Dense(hidden_size, in_units=inp_size, activation='relu'))
            self.f.add(nn.Dense(hidden_size, in_units=hidden_size, activation='relu'))
            self.f.add(nn.Dense(hidden_size, in_units=hidden_size))

    def forward(self, a):
        B, L, H = a.shape
        tilde_a = self.f(a.reshape(B*L, H)).reshape(B, L, self.hidden_size)  # shape = [B, L1, H]
        e = nd.linalg.gemm2(A=tilde_a, B=tilde_a.transpose([0, 2, 1]))
        alpha = nd.linalg.gemm2(nd.softmax(e), tilde_a)
        return alpha


class DecomposableAtten(gluon.Block):
    def __init__(self, inp_size: int, hidden_size: int, num_class:int, **kwargs):
        super(DecomposableAtten, self).__init__(**kwargs)
        with self.name_scope():
            # attention function
            self.f = nn.Sequential()
            self.f.add(nn.Dense(hidden_size, in_units=inp_size, activation='relu'))
            self.f.add(nn.Dense(hidden_size, in_units=hidden_size, activation='relu'))
            self.f.add(nn.Dense(hidden_size, in_units=hidden_size))
            
            # compare function
            self.g = nn.Sequential()
            self.g.add(nn.Dense(hidden_size, in_units=hidden_size * 2, activation='relu'))
            self.g.add(nn.Dense(hidden_size, in_units=hidden_size, activation='relu'))
            self.g.add(nn.Dense(hidden_size, in_units=hidden_size))

            # predictor
            self.h = nn.Sequential()
            self.h.add(nn.Dense(hidden_size, in_units=hidden_size * 2, activation='relu'))
            self.h.add(nn.Dense(hidden_size, in_units=hidden_size, activation='relu'))
            self.h.add(nn.Dense(num_class, in_units=hidden_size))
        # extract features
        self.hidden_size = hidden_size
        self.inp_size = inp_size

    def forward(self, a, b):
        B, L1, H = a.shape
        B2, L2, H2 = b.shape
        assert B == B2
        assert H == H2
        assert H == self.inp_size

        # extract features
        tilde_a = self.f(a.reshape(B*L1, H)).reshape(B, L1, self.hidden_size)  # shape = [B, L1, H]
        tilde_b = self.f(b.reshape(B*L2, H)).reshape(B, L2, self.hidden_size)  # shape = [B, L2, H]
        
        # attention
        # e shape = [B, L1 , L2]
        e = nd.linalg.gemm2(A=tilde_a, B=tilde_b.transpose([0, 2, 1]))
        # beta: b align to a, [B, L1, H]
        beta = nd.linalg.gemm2(nd.softmax(e), tilde_b)
        # alpha: a align to b, [B, L2, H]
        alpha = nd.linalg.gemm2(
            nd.softmax(e.transpose([0, 2, 1])),
            tilde_a)
        
        # compare
        v1 = self.g(nd.concat(tilde_a, beta, dim=2).reshape(B*L1, self.hidden_size*2)).reshape(B, L1, self.hidden_size)
        v2 = self.g(nd.concat(tilde_b, alpha, dim=2).reshape(B*L2, self.hidden_size*2)).reshape(B, L2, self.hidden_size)
        
        # predict
        v1 = v1.mean(axis=1)
        v2 = v2.mean(axis=1)
        yhat = self.h(nd.concat(v1,v2, dim=1))
        
        return yhat
