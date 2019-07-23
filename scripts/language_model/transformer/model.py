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

# pylint: disable=wildcard-import, arguments-differ
"""Module for pre-defined NLP models."""

import os

import mxnet as mx

import gluonnlp as nlp
from gluonnlp.base import get_home_dir

from .transformer import TransformerXL

__all__ = ['get_model']


def get_model(name, **kwargs):
    """Returns a pre-defined model by name."""
    models = {
        # TODO better naming scheme when moving this to main API?
        'transformerxl': transformerxl,
    }
    name = name.lower()
    if name not in models:
        raise ValueError('Model %s is not supported. Available options are\n\t%s' %
                         (name, '\n\t'.join(sorted(models.keys()))))
    return models[name](**kwargs)


def transformerxl(dataset_name: str, vocab: nlp.Vocab, ctx: mx.Context = mx.cpu(),
                  root: str = os.path.join(get_home_dir(), 'models'), **kwargs):
    """Generic pre-trained Transformer-XL model.

    The hyperparameters are chosen based on the specified dataset_name from the
    published hyperparameters of Dai et al.


    References:
    Dai, Z., Yang, Z., Yang, Y., Cohen, W. W., Carbonell, J., Le, Q. V., &
    Salakhutdinov, R. (2019). Transformer-XL: Attentive language models beyond
    a fixed-length context. arXiv preprint arXiv:1901.02860, (), .

    Parameters
    ----------
    dataset_name
        Used to load hyperparameters for the dataset.
    vocab
        Vocabulary for the dataset.
    ctx
        The context in which to load the pretrained weights.
    root
        Location for keeping the model parameters.
        Defaulting to '~/.mxnet/models'.

    Returns
    -------
    TransformerXL, gluonnlp.Vocab

    """

    dataset_name_to_kwargs = dict(
        wt103={
            'embed_cutoffs': [20000, 40000, 200000],
            'embed_size': 1024,
            'embed_div_val': 4,
            'tie_input_output_embeddings': True,
            'tie_input_output_projections': [False, True, True, True],
            'num_layers': 18,
            'hidden_size': 4096,
            'units': 1024,
            'num_heads': 16,
            'dropout': 0,
            'attention_dropout': 0
        },
        # enwiki8=None,  # TODO
        # lm1b=None,
        # text8=None,
    )

    options = dataset_name_to_kwargs[dataset_name]
    options.update(**kwargs)
    model = TransformerXL(vocab_size=len(vocab), **options)
    return model, vocab
