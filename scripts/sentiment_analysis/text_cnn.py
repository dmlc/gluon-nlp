"""textCNN model."""

import gluonnlp as nlp
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import HybridBlock

class SentimentNet(HybridBlock):
    """Network for sentiment analysis."""

    def __init__(self, dropout, embed_size=300, vocab_size=100, prefix=None,
                 params=None, model_mode='multichannel', output_size=2,
                 num_filters=(100, 100, 100), ngram_filter_sizes=(3, 4, 5)):
        super(SentimentNet, self).__init__(prefix=prefix, params=params)
        self.model_mode = model_mode
        with self.name_scope():
            self.embedding = gluon.nn.Embedding(vocab_size, embed_size)
            if self.model_mode == 'multichannel':
                self.embedding_extend = gluon.nn.Embedding(vocab_size, embed_size)
                embed_size *= 2
            self.encoder = nlp.model.ConvolutionalEncoder(embed_size=embed_size,
                                                          num_filters=num_filters,
                                                          ngram_filter_sizes=ngram_filter_sizes,
                                                          conv_layer_activation='relu',
                                                          num_highway=None)
            self.output = gluon.nn.HybridSequential()
            with self.output.name_scope():
                self.output.add(gluon.nn.Dropout(dropout))
                self.output.add(gluon.nn.Dense(output_size, flatten=False))

    def hybrid_forward(self, F, data): # pylint: disable=arguments-differ
        if self.model_mode == 'multichannel':
            embedded = F.concat(self.embedding(data), self.embedding_extend(data), dim=2)
        else:
            embedded = self.embedding(data)
        encoded = self.encoder(embedded)  # Shape(T, N, C)
        out = self.output(encoded)
        return out

def model(dropout, vocab, model_mode, output_size):
    """Construct the model."""

    textCNN = SentimentNet(dropout=dropout, vocab_size=len(vocab), model_mode=model_mode,\
                       output_size=output_size)
    textCNN.hybridize()
    return textCNN

def init(textCNN, vocab, model_mode, context, lr):
    """Initialize parameters."""

    textCNN.initialize(mx.init.Xavier(), ctx=context, force_reinit=True)
    if model_mode != 'rand':
        textCNN.embedding.weight.set_data(vocab.embedding.idx_to_vec)
    if model_mode == 'multichannel':
        textCNN.embedding_extend.weight.set_data(vocab.embedding.idx_to_vec)
    if model_mode == 'static' or model_mode == 'multichannel':
        # Parameters of textCNN.embedding are not updated during training.
        textCNN.embedding.collect_params().setattr('grad_req', 'null')
    trainer = gluon.Trainer(textCNN.collect_params(), 'adam', {'learning_rate': lr})
    return textCNN, trainer
