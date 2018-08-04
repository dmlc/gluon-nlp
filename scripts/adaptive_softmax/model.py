from mxnet.gluon import nn, rnn
import mxnet as mx
from adaptive_softmax import *

class LanguageModel(gluon.Block):
    def __init__(self, vocab_size, num_embed, num_hidden, num_layers, dropout=0.0,
            adaptive_softmax=True, context=mx.gpu(0), cutoff=[2000], **kwargs):
        super(LanguageModel, self).__init__(**kwargs)
        
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(vocab_size, num_embed,
                                        weight_initializer=mx.init.Uniform(0.1))

            self.rnn = rnn.LSTM(num_hidden, num_layers, dropout=dropout,
                                    input_size=num_embed)

        if adaptive_softmax:
            self.linear = Adaptivesoftmax(num_hidden, [*cutoff, vocab_size + 1])
        else:
            self.linear = nn.Dense(units=vocab_size, in_units=num_hidden, flatten=False)
            
        self.adaptive_softmax = adaptive_softmax

        self.num_layers = num_layers
        self.num_hidden = num_hidden
        
    def forward(self, input, hidden, target=None, training=True):
        embed = self.encoder(input)
        embed = self.drop(embed)

        output, hidden = self.rnn(embed, hidden)
        output = self.drop(output)

        if self.adaptive_softmax:
            self.linear.set_target(target)
            nnloss = self.linear(output.reshape((output.shape[0] * output.shape[1], output.shape[2])), target)
        
        if not self.adaptive_softmax:
            output = self.linear(output.reshape((output.shape[0] * output.shape[1], output.shape[2])))
            loss = gluon.loss.SoftmaxCrossEntropyLoss()
            nnloss =  mx.nd.sum(loss(output, target))
            nnloss = nnloss / (len(target))
        
        return nnloss, hidden
         
    def log_prob(self, input, hidden):
        embed = self.encoder(input)
        output, hidden = self.rnn(embed, hidden)
        prob = self.linear.log_prob(output.reshape((output.shape[0] * output.shape[1], output.shape[2])))

        return prob, hidden            

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
