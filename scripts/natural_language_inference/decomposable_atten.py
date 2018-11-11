# pylint: disable=C0103,E1101,R0914
"""
decomposable_atten.py

Part of NLI script of gluon-nlp. Implementation of Decomposable Attention.
Copyright 2018 Mengxiao Lin <linmx0130@gmail.com>.
"""

from mxnet import nd, gluon
from mxnet.gluon import nn

class IntraSentenceAtten(gluon.Block):
    """
    Intra Sentence Attantion block.
    """
    def __init__(self, inp_size: int, hidden_size: int, max_length=10, **kwargs):
        super(IntraSentenceAtten, self).__init__(**kwargs)
        self.max_length = max_length
        self.hidden_size = hidden_size
        with self.name_scope():
            self.func = nn.Sequential()
            self.func.add(nn.Dense(hidden_size, in_units=inp_size, activation='relu'))
            self.func.add(nn.Dense(hidden_size, in_units=hidden_size, activation='relu'))
            self.func.add(nn.Dense(hidden_size, in_units=hidden_size))

    def forward(self, *args):
        feature_a = args[0]
        batch_size, length, inp_size = feature_a.shape
        tilde_a = self.func(
            feature_a.reshape(batch_size * length, inp_size)).reshape(
                batch_size, length, self.hidden_size)
        e_matrix = nd.linalg.gemm2(A=tilde_a, B=tilde_a.transpose([0, 2, 1]))
        alpha = nd.linalg.gemm2(nd.softmax(e_matrix), tilde_a)
        return alpha

class DecomposableAtten(gluon.Block):
    """
    Decomposable Attention block.
    """
    def __init__(self, inp_size: int, hidden_size: int, num_class: int, **kwargs):
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

    def forward(self, *args):
        """
        Forward of Decomposable Attention layer
        """
        a, b = args[0], args[1]
        batch_size1, length1, hidden_size1 = a.shape
        batch_size2, length2, hidden_size2 = b.shape
        assert batch_size1 == batch_size2
        assert hidden_size1 == hidden_size2
        assert hidden_size1 == self.inp_size
        hidden_size = hidden_size1
        batch_size = batch_size1
        # extract features
        tilde_a = self.f(
            a.reshape(batch_size * length1, hidden_size)).reshape(
                batch_size, length1, self.hidden_size)  # shape = [B, L1, H]
        tilde_b = self.f(
            b.reshape(batch_size * length2, hidden_size)).reshape(
                batch_size, length2, self.hidden_size)  # shape = [B, L2, H]
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
        feature1 = self.g(nd.concat(tilde_a, beta, dim=2).reshape(
            batch_size * length1, self.hidden_size * 2)).reshape(
                batch_size, length1, self.hidden_size)
        feature2 = self.g(nd.concat(tilde_b, alpha, dim=2).reshape(
            batch_size * length2, self.hidden_size * 2)).reshape(
                batch_size, length2, self.hidden_size)
        feature1 = feature1.mean(axis=1)
        feature2 = feature2.mean(axis=1)
        yhat = self.h(nd.concat(feature1, feature2, dim=1))
        return yhat
