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

# pylint: disable=wildcard-import
"""NLP training model."""

import gluonnlp as nlp

import mxnet as mx

from . import cache, embedding, language_model
from .cache import *
from .embedding import *
from .language_model import *

__all__ = language_model.__all__ + cache.__all__ + embedding.__all__ + \
    ['get_cache_model']


def get_cache_model(name, dataset_name='wikitext-2', window=2000,
                    theta=0.6, lambdas=0.2, ctx=mx.cpu(), **kwargs):
    r"""Returns a cache model using a pre-trained language model.

    We implement the neural cache language model proposed in the following work::

        @article{grave2016improving,
        title={Improving neural language models with a continuous cache},
        author={Grave, Edouard and Joulin, Armand and Usunier, Nicolas},
        journal={ICLR},
        year={2017}
        }

    Parameters
    ----------
    name : str
        Name of the cache language model.
    dataset_name : str or None, default 'wikitext-2'.
        The dataset name on which the pre-trained model is trained.
        Options are 'wikitext-2'. If specified, then the returned vocabulary is extracted from
        the training set of the dataset.
        If None, then vocab is required, for specifying embedding weight size, and is directly
        returned.
    window : int
        Size of cache window
    theta : float
        The scala controls the flatness of the cache distribution
        that predict the next word as shown below:

        .. math::

            p_{cache} \propto \sum_{i=1}^{t-1} \mathbb{1}_{w=x_{i+1}} exp(\theta {h_t}^T h_i)

        where :math:`p_{cache}` is the cache distribution, :math:`\mathbb{1}` is
        the identity function, and :math:`h_i` is the output of timestep i.
    lambdas : float
        Linear scalar between only cache and vocab distribution, the formulation is as below:

        .. math::

            p = (1 - \lambda) p_{vocab} + \lambda p_{cache}

        where :math:`p_{vocab}` is the vocabulary distribution and :math:`p_{cache}`
        is the cache distribution.
    vocab : gluonnlp.Vocab or None, default None
        Vocabulary object to be used with the language model.
        Required when dataset_name is not specified.
    pretrained : bool, default False
        Whether to load the pre-trained weights for model.
    ctx : Context, default CPU
        The context in which to load the pre-trained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the pre-trained model parameters.

    Returns
    -------
    Block
        The model.
    """
    lm_model, vocab = nlp.model.\
        get_model(name, dataset_name=dataset_name, pretrained=True, ctx=ctx, **kwargs)
    cache_cell = CacheCell(lm_model, len(vocab), window, theta, lambdas)
    return cache_cell
