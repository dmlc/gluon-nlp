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
import time
import zipfile
from typing import Optional

import mxnet as mx
from mxnet.gluon.model_zoo import model_store
from mxnet.gluon.utils import _get_repo_url, check_sha1, download

import gluonnlp as nlp
from gluonnlp.base import get_home_dir
from gluonnlp.data.utils import _url_format
from gluonnlp.model.utils import _load_pretrained_params, _load_vocab

from .data import XLNetTokenizer
from .transformer import TransformerXL, XLNet

__all__ = ['get_model']

model_store._model_sha1.update({
    name: checksum
    for checksum, name in [
        ('ca7a092186ec3f42ef25590a872450409faaa84f', 'xlnet_cased_l12_h768_a12_126gb'),
        ('ceae74798c1577bcf5ffb3c46b73b056a5ead786', 'xlnet_cased_l24_h1024_a16_126gb'),
    ]
})


def get_model(name, **kwargs):
    """Returns a pre-defined model by name."""
    models = {
        # TODO better naming scheme when moving this to main API?
        'transformerxl': transformerxl,
        'xlnet_cased_l12_h768_a12': xlnet_cased_l12_h768_a12,
        'xlnet_cased_l24_h1024_a16': xlnet_cased_l24_h1024_a16
    }
    name = name.lower()
    if name not in models:
        raise ValueError('Model %s is not supported. Available options are\n\t%s' %
                         (name, '\n\t'.join(sorted(models.keys()))))
    return models[name](**kwargs)


def transformerxl(dataset_name: str, vocab: nlp.Vocab, **kwargs):
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
        }, lm1b={
            'embed_cutoffs': [60000, 100000, 640000],
            'embed_size': 1280,
            'embed_div_val': 4,
            'project_same_dim': False,
            'tie_input_output_embeddings': True,
            'num_layers': 24,
            'hidden_size': 8192,
            'units': 1280,
            'num_heads': 16,
            'dropout': 0,
            'attention_dropout': 0
        }, enwik8={
            'embed_size': 1024,
            'tie_input_output_embeddings': True,
            'num_layers': 24,
            'hidden_size': 3072,
            'units': 1024,
            'num_heads': 8,
            'dropout': 0,
            'attention_dropout': 0
        }, text8={
            'embed_size': 1024,
            'tie_input_output_embeddings': True,
            'num_layers': 24,
            'hidden_size': 3072,
            'units': 1024,
            'num_heads': 8,
            'dropout': 0,
            'attention_dropout': 0
        })

    options = dataset_name_to_kwargs[dataset_name]
    options.update(**kwargs)
    model = TransformerXL(vocab_size=len(vocab), **options)
    return model, vocab


def xlnet_cased_l12_h768_a12(dataset_name: Optional[str] = None, vocab: Optional[nlp.Vocab] = None,
                             tokenizer: Optional[XLNetTokenizer] = None, pretrained: bool = True,
                             ctx: mx.Context = mx.cpu(),
                             root=os.path.join(get_home_dir(), 'models'),
                             do_lower_case=False, **kwargs):
    """XLNet model.

    References:
    Yang, Z., Dai, Z., Yang, Y., Carbonell, J., Salakhutdinov, R., & Le, Q. V.
    (2019). XLNet: Generalized Autoregressive Pretraining for Language
    Understanding. arXiv preprint arXiv:1906.08237.


    Parameters
    ----------
    dataset_name : str or None, default None
        If not None, the dataset name is used to load a vocabulary for the
        dataset. If the `pretrained` argument is set to True, the dataset name
        is further used to select the pretrained parameters to load.
        Options include 'books_enwiki_giga5_clueweb2012b_commoncrawl'.
    vocab : gluonnlp.vocab.Vocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset_name is not
        specified. Ignored if dataset_name is specified.
    tokenizer : XLNetTokenizer or None, default None
        XLNetTokenizer for the dataset. Must be provided if dataset_name is not
        specified. Ignored if dataset_name is specified.
    pretrained : bool, default True
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
        MXNET_HOME defaults to '~/.mxnet'.

    Returns
    -------
    XLNet, gluonnlp.Vocab
    """

    kwargs.update(**{
        'hidden_size': 3072,
        'units': 768,
        'activation': 'gelu',
        'num_heads': 12,
        'num_layers': 12,
    })
    if vocab is None or dataset_name is not None:
        vocab = _load_vocab('xlnet_' + dataset_name, vocab, root)
    net = XLNet(vocab_size=len(vocab), **kwargs)
    if pretrained:
        _load_pretrained_params(net=net, model_name='xlnet_cased_l12_h768_a12',
                                dataset_name=dataset_name, root=root, ctx=ctx,
                                ignore_extra=not kwargs.get('use_decoder', True))
    if tokenizer is None or dataset_name is not None:
        tokenizer = _get_xlnet_tokenizer(dataset_name, root, do_lower_case)
    return net, vocab, tokenizer


def xlnet_cased_l24_h1024_a16(dataset_name: Optional[str] = None, vocab: Optional[nlp.Vocab] = None,
                              tokenizer: Optional[XLNetTokenizer] = None, pretrained: bool = True,
                              ctx: mx.Context = mx.cpu(),
                              root=os.path.join(get_home_dir(), 'models'),
                              do_lower_case=False, **kwargs):
    """XLNet model.

    References:
    Yang, Z., Dai, Z., Yang, Y., Carbonell, J., Salakhutdinov, R., & Le, Q. V.
    (2019). XLNet: Generalized Autoregressive Pretraining for Language
    Understanding. arXiv preprint arXiv:1906.08237.


    Parameters
    ----------
    dataset_name : str or None, default None
        If not None, the dataset name is used to load a vocabulary for the
        dataset. If the `pretrained` argument is set to True, the dataset name
        is further used to select the pretrained parameters to load.
        Options include 'books_enwiki_giga5_clueweb2012b_commoncrawl'.
    vocab : gluonnlp.vocab.Vocab or None, default None
        Vocabulary for the dataset. Must be provided if dataset_name is not
        specified. Ignored if dataset_name is specified.
    tokenizer : XLNetTokenizer or None, default None
        XLNetTokenizer for the dataset. Must be provided if dataset_name is not
        specified. Ignored if dataset_name is specified.
    pretrained : bool, default True
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
        MXNET_HOME defaults to '~/.mxnet'.

    Returns
    -------
    XLNet, gluonnlp.Vocab, XLNetTokenizer

    """
    kwargs.update(**{
        'hidden_size': 4096,
        'units': 1024,
        'activation': 'approx_gelu',
        'num_heads': 16,
        'num_layers': 24,
    })
    if vocab is None or dataset_name is not None:
        vocab = _load_vocab('xlnet_' + dataset_name, vocab, root)
    net = XLNet(vocab_size=len(vocab), **kwargs)
    if pretrained:
        _load_pretrained_params(net=net, model_name='xlnet_cased_l24_h1024_a16',
                                dataset_name=dataset_name, root=root, ctx=ctx,
                                ignore_extra=not kwargs.get('use_decoder', True))
    if tokenizer is None or dataset_name is not None:
        tokenizer = _get_xlnet_tokenizer(dataset_name, root, do_lower_case)
    return net, vocab, tokenizer


def _get_xlnet_tokenizer(dataset_name, root, do_lower_case=False):
    assert dataset_name.lower() == '126gb'
    root = os.path.expanduser(root)
    file_path = os.path.join(root, 'xlnet_126gb-871f0b3c.spiece')
    sha1_hash = '871f0b3c13b92fc5aea8fba054a214c420e302fd'
    if os.path.exists(file_path):
        if not check_sha1(file_path, sha1_hash):
            print('Detected mismatch in the content of model tokenizer. Downloading again.')
    else:
        print('Tokenizer file is not found. Downloading.')

    os.makedirs(root, exist_ok=True)

    repo_url = _get_repo_url()
    prefix = str(time.time())
    zip_file_path = os.path.join(root, prefix + 'xlnet_126gb-871f0b3c.zip')
    if repo_url[-1] != '/':
        repo_url = repo_url + '/'
    download(_url_format.format(repo_url=repo_url, file_name='xlnet_126gb-871f0b3c'),
             path=zip_file_path, overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        if not os.path.exists(file_path):
            zf.extractall(root)
    try:
        os.remove(zip_file_path)
    except OSError as e:
        # file has already been removed.
        if e.errno == 2:
            pass
        else:
            raise e

    if not check_sha1(file_path, sha1_hash):
        raise ValueError('Downloaded file has different hash. Please try again.')

    tokenizer = XLNetTokenizer(file_path, lower=do_lower_case)
    return tokenizer
