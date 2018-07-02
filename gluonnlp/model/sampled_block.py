# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Blocks for sampled losses."""
__all__ = ['SampledLogits', 'SparseSampledLogits']

from mxnet import ndarray
from mxnet.gluon import Block, HybridBlock


class _SampledLogits(HybridBlock):
    """A helper Block for calculating sampled logits.

    Parameters
    ----------
    num_classes: int
        Number of possible classes.
    num_sampled: int
        Number of classes randomly sampled for each batch.
    in_unit: int
        Dimensionality of the input space.
    remove_accidental_hits: bool
        Whether to remove "accidental hits" when a sampled candidate is equal to
        one of the true classes.
    """
    def __init__(self, num_classes, num_sampled, in_unit, remove_accidental_hits,
                 prefix=None, params=None):
        super(_SampledLogits, self).__init__(prefix=prefix, params=params)
        self._num_classes = num_classes
        self._num_sampled = num_sampled
        self._in_unit = in_unit
        self._remove_accidental_hits = remove_accidental_hits

    def hybrid_forward(self, F, x, sampled_candidates, expected_count_sampled,
                       expected_count_true, label, w_all, b_all):
        """Forward computation."""
        # (num_sampled, in_unit)
        w_sampled = w_all.slice(begin=(0, 0), end=(self._num_sampled, None))
        w_true = w_all.slice(begin=(self._num_sampled, 0), end=(None, None))
        b_sampled = b_all.slice(begin=(0,), end=(self._num_sampled,))
        b_true = b_all.slice(begin=(self._num_sampled,), end=(None,))
        # true
        # (batch_size, 1)
        x = x.reshape((-1, self._in_unit))
        logits_true = (w_true * x).sum(axis=1) + b_true
        # samples
        # (batch_size, num_sampled)
        b_sampled = F.reshape(b_sampled, (-1,))
        logits_sampled = F.FullyConnected(x, weight=w_sampled, bias=b_sampled,
                                          num_hidden=self._num_sampled)

        # remove accidental hits
        if self._remove_accidental_hits:
            label_vec = F.reshape(label, (-1, 1))
            sample_vec = F.reshape(sampled_candidates, (1, -1))
            mask = F.broadcast_equal(label_vec, sample_vec) * -1e37
            logits_sampled = logits_sampled + mask

        expected_count_sampled = F.reshape(expected_count_sampled,
                                           shape=(1, self._num_sampled))
        expected_count_true = expected_count_true.reshape((-1,))
        logits_true = logits_true - F.log(expected_count_true)
        logits_true = logits_true.reshape((-1, 1))
        logits_sampled = F.broadcast_sub(logits_sampled, F.log(expected_count_sampled))

        # logits and new_labels
        # (batch_size, 1+num_sampled)
        logits = F.concat(logits_true, logits_sampled, dim=1)
        return logits, F.zeros_like(label)

    def __repr__(self):
        s = '{name}({mapping})'
        mapping = '{0} -> {1}, with {2} samples'.format(self._in_unit, self._num_classes,
                                                        self._num_sampled)
        return s.format(name=self.__class__.__name__,
                        mapping=mapping,
                        **self.__dict__)

class SampledLogits(HybridBlock):
    """Block that computes sampled output training logits and labels suitable for
    sampled softmax loss or noise contrastive estimation loss.

    Please use `loss.SoftmaxCrossEntropyLoss` for sampled softmax loss, and
    `loss.SigmoidBinaryCrossEntropyLoss` for nce loss.

    The block is designed for distributed training with extremely large
    number of classes. Both weight and gradient w.r.t. weight are `RowSparseNDArray`.

    Example::

        # network with sampled_softmax_loss for training
        encoder = Encoder(..)
        decoder = SampledLogits(.., prefix='decoder')
        train_net.add(encoder)
        train_net.add(decoder)
        loss = SoftmaxCrossEntropyLoss()

        # training
        for x, y, sampled_values in train_batches:
            sampled_cls, cnt_sampled, cnt_true = sampled_values
            logits, new_targets = train_net(x, sampled_cls, cnt_sampled, cnt_true, y)
            l = loss(logits, new_targets)

        # network for testing
        test_net.add(encoder)
        test_net.add(Dense(..., prefix='decoder', params=decoder.params))

        # testing
        for x, y in test_batches:
            logits = test_net(x)
            l = loss(logits, y)

    Parameters
    ----------
    num_classes: int
        Number of possible classes.
    num_sampled: int
        Number of classes randomly sampled for each batch.
    in_unit: int
        Dimensionality of the input space.
    remove_accidental_hits: bool
        Whether to remove "accidental hits" when a sampled candidate is equal to
        one of the true classes.
    dtype : str or np.dtype, default 'float32'
        Data type of output embeddings.
    weight_initializer : str or `Initializer`, optional
        Initializer for the `kernel` weights matrix.
    bias_initializer: str or `Initializer`, optional
        Initializer for the bias vector.

    Inputs:
        - **x**: A tensor of shape `(batch_size, in_unit)`. The forward activation of
          the input network.
        - **sampled_candidates**: A tensor of shape `(num_sampled,)`.
          The sampled candidate classes.
        - **expected_count_sampled**: A tensor of shape `(num_sampled,)`.
          The expected count for sampled candidates.
        - **expected_count_true**: A tensor of shape `(num_sampled)`.
          The expected count for true classes.
        - **label**: A tensor of shape `(batch_size,1)`.
          The target classes.

    Outputs:
        - **out**: A tensor of shape `(batch_size, 1+num_sampled)`.
          The output probability for the true class and sampled classes
        - **new_targets**: A tensor of shape `(batch_size,)`.
          The new target classes.

    """
    def __init__(self, num_classes, num_sampled, in_unit, remove_accidental_hits,
                 prefix=None, params=None):
        super(SampledLogits, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(num_classes, in_unit),
                                          grad_stype='row_sparse')
            self.bias = self.params.get('bias', shape=(num_classes, 1),
                                        grad_stype='row_sparse')
        self.block = _SampledLogits(num_classes, num_sampled, in_unit, remove_accidental_hits)
        self._num_classes = num_classes
        self._num_sampled = num_sampled
        self._in_unit = in_unit
        self._remove_accidental_hits = remove_accidental_hits

    def hybrid_forward(self, F, x, sampled_candidates, expected_count_sampled,
                       expected_count_true, label, weight, bias):
        """Forward computation."""
        # (batch_size,)
        label = F.reshape(label, shape=(-1,))
        # (num_sampled+batch_size,)
        ids = F.concat(sampled_candidates, label, dim=0)
        # lookup weights and biases
        # (num_sampled+batch_size, dim)
        w_all = F.Embedding(data=ids, weight=weight,
                            input_dim=self._num_classes, output_dim=self._in_unit,
                            sparse_grad=True)
        # (num_sampled+batch_size, 1)
        b_all = F.Embedding(data=ids, weight=bias,
                            input_dim=self._num_classes, output_dim=1,
                            sparse_grad=True)
        return self.block(x, sampled_candidates, expected_count_sampled,
                          expected_count_true, label, w_all, b_all)

    def __repr__(self):
        s = '{name}({mapping})'
        mapping = '{0} -> {1}, with {2} samples'.format(self._in_unit, self._num_classes,
                                                        self._num_sampled)
        return s.format(name=self.__class__.__name__,
                        mapping=mapping,
                        **self.__dict__)

class SparseSampledLogits(Block):
    """Block that computes sampled output training logits and labels suitable for
    sampled softmax loss or noise contrastive estimation loss.

    Please use `loss.SoftmaxCrossEntropyLoss` for sampled softmax loss, and
    `loss.SigmoidBinaryCrossEntropyLoss` for nce loss.

    The block is designed for distributed training with extremely large
    number of classes. Both weight and gradient w.r.t. weight are `RowSparseNDArray`.

    Different from SampledLogits block, the parameters have to be saved before they
    are used for testing.

    Example::

        # network with sampled_softmax_loss for training
        encoder = Encoder(..)
        train_net.add(encoder)
        train_net.add(SampledLogits(.., prefix='decoder')))
        loss = SoftmaxCrossEntropyLoss()

        # training
        for x, y, sampled_values in train_batches:
            sampled_cls, cnt_sampled, cnt_true = sampled_values
            logits, new_targets = train_net(x, sampled_cls, cnt_sampled, cnt_true, y)
            l = loss(logits, new_targets)

        # save params
        train_net.save_parameters('net.params')

        # network for testing
        test_net.add(encoder)
        test_net.add(Dense(..., prefix='decoder'))

        # load params
        test_net.load_parameters('net.params')

        # testing
        for x, y in test_batches:
            logits = test_net(x)
            l = loss(logits, y)

    Parameters
    ----------
    num_classes: int
        Number of possible classes.
    num_sampled: int
        Number of classes randomly sampled for each batch.
    in_unit: int
        Dimensionality of the input space.
    remove_accidental_hits: bool
        Whether to remove "accidental hits" when a sampled candidate is equal to
        one of the true classes.
    dtype : str or np.dtype, default 'float32'
        Data type of output embeddings.
    weight_initializer : str or `Initializer`, optional
        Initializer for the `kernel` weights matrix.
    bias_initializer: str or `Initializer`, optional
        Initializer for the bias vector.

    Inputs:
        - **x**: A tensor of shape `(batch_size, in_unit)`. The forward activation of
          the input network.
        - **sampled_candidates**: A tensor of shape `(num_sampled,)`.
          The sampled candidate classes.
        - **expected_count_sampled**: A tensor of shape `(num_sampled,)`.
          The expected count for sampled candidates.
        - **expected_count_true**: A tensor of shape `(num_sampled)`.
          The expected count for true classes.
        - **label**: A tensor of shape `(batch_size,1)`.
          The target classes.

    Outputs:
        - **out**: A tensor of shape `(batch_size, 1+num_sampled)`.
          The output probability for the true class and sampled classes
        - **new_targets**: A tensor of shape `(batch_size,)`.
          The new target classes.

    """
    def __init__(self, num_classes, num_sampled, in_unit, remove_accidental_hits,
                 dtype='float32', weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None):
        super(SparseSampledLogits, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(num_classes, in_unit),
                                          init=weight_initializer, dtype=dtype,
                                          grad_stype='row_sparse', stype='row_sparse')
            self.bias = self.params.get('bias', shape=(num_classes,), init=bias_initializer)
            self._logits = _SampledLogits(num_classes, num_sampled, in_unit, remove_accidental_hits)
        self._num_classes = num_classes
        self._num_sampled = num_sampled
        self._in_unit = in_unit
        self._remove_accidental_hits = remove_accidental_hits
        self._kwargs = {'input_dim': self._num_classes, 'output_dim': self._in_unit,
                        'sparse_grad': True}

    def forward(self, x, sampled_candidates, expected_count_sampled,
                expected_count_true, label):
        """Forward computation."""
        # (batch_size,)
        label = label.reshape(shape=(-1,))
        # (num_sampled+batch_size,)
        ids = ndarray.concat(sampled_candidates, label, dim=0)
        # lookup weights and biases
        weight = self.weight.row_sparse_data(ids)
        bias = self.bias.data(ids.context)
        # (num_sampled+batch_size, dim)
        w_all = ndarray.Embedding(data=ids, weight=weight, **self._kwargs)
        # (num_sampled+batch_size,)
        b_all = ndarray.take(bias, indices=ids)
        out, new_targets = self._logits(x, sampled_candidates, expected_count_sampled,
                                        expected_count_true, label, w_all, b_all)
        return out, new_targets

    def __repr__(self):
        s = '{name}({mapping})'
        mapping = '{0} -> {1}, num_sampled = {2}, remove_accidental_hits = {3}'
        mapping = mapping.format(self._in_unit, self._num_classes, self._num_sampled,
                                 str(self._remove_accidental_hits))
        return s.format(name=self.__class__.__name__,
                        mapping=mapping, **self.__dict__)
