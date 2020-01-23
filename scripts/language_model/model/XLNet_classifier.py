"""Model for sentence (pair) classification task/ regression with XLnet.
"""
from mxnet.gluon import Block
from mxnet.gluon import nn
import mxnet as mx


class XLNetClassifier(Block):
    """XLNet Classifier
    """
    def __init__(self, xl, units=768, num_classes=2, dropout=0.0,
                 prefix=None, params=None):
        super(XLNetClassifier, self).__init__(prefix=prefix, params=params)
        self.xlnet = xl
        self._units = units
        with self.name_scope():
            self.classifier = nn.HybridSequential(prefix=prefix)
            if dropout:
                self.classifier.add(nn.Dropout(rate=dropout))
            self.classifier.add(nn.Dense(units=num_classes, flatten=False))
            self.pooler = nn.Dense(units=units, flatten=False, activation='tanh', prefix=prefix)

    def __call__(self, inputs, token_types, valid_length=None, mems=None):
        # pylint: disable=arguments-differ
        """Generate the unnormalized score for the given the input sequences.

        Parameters
        ----------
        inputs : NDArray or Symbol, shape (batch_size, seq_length)
            Input words for the sequences.
        token_types : NDArray or Symbol, shape (batch_size, seq_length)
            Token types for the sequences, used to indicate whether the word belongs to the
            first sentence or the second one.
        valid_length : NDArray or Symbol, or None, shape (batch_size)
            Valid length of the sequence. This is used to mask the padded tokens.

        Returns
        -------
        outputs : NDArray or Symbol
            Shape (batch_size, num_classes)
        """
        return super(XLNetClassifier, self).__call__(inputs, token_types, valid_length, mems)

    def _apply_pooling(self, sequence, valid_length):
        """Generate the representation given the inputs.

        This is used for pre-training or fine-tuning a XLNet model.
        """
        F = mx.ndarray
        index = F.contrib.arange_like(sequence, axis=0, ctx=sequence.context).expand_dims(1)
        valid_length_rs = valid_length.reshape((-1, 1)) - 1
        gather_index = F.concat(index, valid_length_rs).T
        cls_states = F.gather_nd(sequence, gather_index)
        return self.pooler(cls_states)

    def _padding_mask(self, inputs, valid_length):
        F = mx.ndarray
        valid_length = valid_length.astype(inputs.dtype)
        steps = F.contrib.arange_like(inputs, axis=1)
        ones = F.ones_like(steps)
        mask = F.broadcast_lesser(F.reshape(steps, shape=(1, -1)),
                                  F.reshape(valid_length, shape=(-1, 1)))
        mask = F.broadcast_mul(F.expand_dims(mask, axis=1),
                               F.broadcast_mul(ones, F.reshape(ones, shape=(-1, 1))))
        return mask

    def forward(self, inputs, token_types, valid_length=None, mems=None):
        # pylint: disable=arguments-differ
        """Generate the unnormalized score for the given the input sequences.

        Parameters
        ----------
        inputs : NDArray or Symbol, shape (batch_size, seq_length)
            Input words for the sequences.
        token_types : NDArray or Symbol, shape (batch_size, seq_length)
            Token types for the sequences, used to indicate whether the word belongs to the
            first sentence or the second one.
        valid_length : NDArray or None, shape (batch_size)
            Valid length of the sequence. This is used to mask the padded tokens.

        Returns
        -------
        outputs : NDArray
            Shape (batch_size, num_classes)
        """
        attention_mask = self._padding_mask(inputs, valid_length).astype('float32')
        output, _ = self.xlnet(inputs, token_types, mems, attention_mask)
        output = self._apply_pooling(output, valid_length.astype('float32'))
        pooler_out = self.pooler(output)
        return self.classifier(pooler_out)
