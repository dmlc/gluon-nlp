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
"""Machine translation models and translators."""


__all__ = []

import warnings
import numpy as np
from mxnet.gluon import Block, HybridBlock
from mxnet.gluon import nn
import mxnet as mx

class BERTClassifier(Block):
    """ Model for sentence classification task with BERT.

    Parameters
    ----------
    bert: BERTModel
        Bidirectional encoder with transformer.
    num_classes : int, default is 2
        The number of target classes.
    dropout : float or None, default 0.0.
        Dropout probability for the bert output.
    prefix : str or None
        See document of `mx.gluon.Block`.
    params : ParameterDict or None
        See document of `mx.gluon.Block`.
    """
    def __init__(self, bert, num_classes=2, dropout=0.0, prefix=None, params=None):
        super(BERTClassifier, self).__init__(prefix=prefix, params=params)
        self.bert = bert
        with self.name_scope():
            self.classifier = nn.HybridSequential(prefix=prefix)
            if dropout:
                self.classifier.add(nn.Dropout(rate=dropout))
            self.classifier.add(nn.Dense(units=num_classes, flatten=False))

    def forward(self, inputs, token_types, valid_length=None):
        """Generate the unnormalized score for the given the input sequences.

        Parameters
        ----------
        inputs : NDArray
        token_types : NDArray
        valid_length : NDArray or None

        Returns
        -------
        outputs : NDArray
            Shape (batch_size, num_classes)
        """
        seq_out, pooler_out  = self.bert(inputs, token_types, valid_length)
        #print('pooled out nd', pooler_out)
        #print('pooled out', pooler_out.asnumpy().mean())
        #out_np = pooler_out.asnumpy()
        #import numpy as np
        #np.save('/tmp/mx.out', out_np)
        return self.classifier(pooler_out)
