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
"""
BART Model

@article{lewis2019bart,
    title = {BART: Denoising Sequence-to-Sequence Pre-training for Natural
Language Generation, Translation, and Comprehension},
    author = {Mike Lewis and Yinhan Liu and Naman Goyal and Marjan Ghazvininejad and
              Abdelrahman Mohamed and Omer Levy and Veselin Stoyanov
              and Luke Zettlemoyer },
    journal={arXiv preprint arXiv:1910.13461},
    year = {2019},
}

"""

__all__ = ['BartModel', 'list_pretrained_bart', 'get_pretrained_bart']

import os
from typing import Tuple, List

import mxnet as mx
from mxnet import use_np
from mxnet.gluon import nn

from ..base import get_model_zoo_home_dir, get_repo_model_zoo_url, \
                   get_model_zoo_checksum_dir
from ..registry import BACKBONE_REGISTRY
from ..utils.misc import download, load_checksum_stats
from .transformer import TransformerModel
from ..utils.config import CfgNode as CN
from ..utils.registry import Registry
from ..data.tokenizers import HuggingFaceByteBPETokenizer

bart_cfg_reg = Registry('bart_cfg')


@bart_cfg_reg.register()
def bart_base():
    cfg = CN()
    # Config for the bart base model
    cfg.MODEL = CN()
    cfg.MODEL.vocab_size = 51201
    cfg.MODEL.max_src_length = 1024
    cfg.MODEL.max_tgt_length = 1024
    cfg.MODEL.scale_embed = False
    cfg.MODEL.pos_embed_type = 'learned'
    cfg.MODEL.shared_embed = True
    cfg.MODEL.tie_weights = True
    cfg.MODEL.attention_dropout = 0.1
    cfg.MODEL.activation_dropout = 0.0
    cfg.MODEL.dropout = 0.1
    cfg.MODEL.layer_norm_eps = 1E-5
    cfg.MODEL.pooler_activation = 'tanh'
    cfg.MODEL.data_norm = True
    cfg.MODEL.layout = 'NT'
    cfg.MODEL.dtype = 'float32'

    # Parameters for the encoder
    cfg.MODEL.ENCODER = CN()
    cfg.MODEL.ENCODER.num_layers = 6
    cfg.MODEL.ENCODER.units = 768
    cfg.MODEL.ENCODER.num_heads = 12
    cfg.MODEL.ENCODER.hidden_size = 3072
    cfg.MODEL.ENCODER.recurrent = False
    cfg.MODEL.ENCODER.pre_norm = False
    cfg.MODEL.ENCODER.activation = 'gelu'
    cfg.MODEL.ENCODER.use_qkv_bias = True

    # Parameters for the decoder
    cfg.MODEL.DECODER = CN()
    cfg.MODEL.DECODER.num_layers = 6
    cfg.MODEL.DECODER.units = 768
    cfg.MODEL.DECODER.num_heads = 12
    cfg.MODEL.DECODER.hidden_size = 3072
    cfg.MODEL.DECODER.recurrent = False
    cfg.MODEL.DECODER.pre_norm = False
    cfg.MODEL.DECODER.activation = 'gelu'
    cfg.MODEL.DECODER.use_qkv_bias = True

    # Parameters for the initializer
    cfg.INITIALIZER = CN()
    cfg.INITIALIZER.embed = ['xavier', 'gaussian', 'in', 1.0]
    cfg.INITIALIZER.weight = ['xavier', 'uniform', 'avg', 1.0]
    cfg.INITIALIZER.bias = ['zeros']
    cfg.VERSION = 1
    cfg.freeze()
    return cfg


@bart_cfg_reg.register()
def bart_large():
    cfg = bart_base()
    cfg.defrost()
    cfg.MODEL.vocab_size = 50265
    cfg.MODEL.ENCODER.units = 1024
    cfg.MODEL.ENCODER.hidden_size = 4096
    cfg.MODEL.ENCODER.num_heads = 16
    cfg.MODEL.ENCODER.num_layers = 12
    cfg.MODEL.DECODER.units = 1024
    cfg.MODEL.DECODER.hidden_size = 4096
    cfg.MODEL.DECODER.num_heads = 16
    cfg.MODEL.DECODER.num_layers = 12
    cfg.freeze()
    return cfg


PRETRAINED_URL = {
    'fairseq_bart_base': {
        'cfg': bart_base(),
        'merges': 'fairseq_bart_base/gpt2-396d4d8e.merges',
        'vocab': 'fairseq_bart_base/gpt2-f4dedacb.vocab',
        'params': 'fairseq_bart_base/model-8f4929b5.params',
        'lowercase': False,
    },
    'fairseq_bart_large': {
        'cfg': bart_large(),
        'merges': 'fairseq_bart_large/gpt2-396d4d8e.merges',
        'vocab': 'fairseq_bart_large/gpt2-f1335494.vocab',
        'params': 'fairseq_bart_large/model-862277b1.params',
        'lowercase': False,
    }
}


FILE_STATS = load_checksum_stats(os.path.join(get_model_zoo_checksum_dir(), 'bart.txt'))


@use_np
class BartModel(TransformerModel):
    def __init__(self,
                 use_pooler: bool = False,
                 classifier_activation: bool = False,
                 extract_feature: bool = False,
                 pooler_activation='tanh',
                 **kwargs):
        """

        Parameters
        ----------
        use_pooler
            Whether to use pooler
        classifier_activation
        extract_feature
            Whether to extract the feature
        pooler_activation
        **kwargs
        """
        super().__init__(**kwargs)
        assert self._src_vocab_size == self._tgt_vocab_size, \
            'Vocab size mismatch between encoder and decoder'
        self._vocab_size = self._src_vocab_size
        self.extract_feature = extract_feature
        self.use_pooler = use_pooler
        self.classifier_activation = classifier_activation
        if not extract_feature:
            if self.tie_weights:
                self.tgt_final_layer = \
                    nn.Dense(units=self._tgt_vocab_size,
                             in_units=self.dec_units,
                             flatten=False,
                             use_bias=False,
                             dtype=self._dtype)
                self.tgt_final_layer.weight = self.tgt_embed_layer.weight
            else:
                self.tgt_final_layer = \
                    nn.Dense(units=self._tgt_vocab_size,
                             in_units=self.dec_units,
                             flatten=False,
                             weight_initializer=self.weight_initializer,
                             use_bias=False,
                             dtype=self._dtype)
        elif use_pooler and classifier_activation:
            # Construct pooler
            self.pooler = nn.Dense(units=self.units,
                                   in_units=self.units,
                                   flatten=False,
                                   activation=pooler_activation,
                                   weight_initializer=self.weight_initializer,
                                   bias_initializer=self.bias_initializer,
                                   dtype=self._dtype)

    def hybrid_forward(self, F, src_data, src_valid_length, tgt_data, tgt_valid_length):
        """

        Parameters
        ----------
        F
        src_data
            - layout = 'NT'
                Shape (batch_size, src_length)
            - layout = 'TN'
                Shape (src_length, batch_size)
        src_valid_length
            Shape (batch_size,)
        tgt_data
            - layout = 'NT'
                Shape (batch_size, tgt_length)
            - layout = 'TN'
                Shape (tgt_length, batch_size)
        tgt_valid_length
            Shape (batch_size,)

        Returns
        -------
        A tuple contains

        - If 'self.extract_feature' = True
            - contextual_embedding
                - layout = 'NT'
                    Shape (batch_size, tgt_length, units)
                - layout = 'TN'
                    Shape (tgt_length, batch_size, units)
            - pooled_output, optional, only enabled if use_pooler = True
                Shape (batch_size, units)
        - If 'self.extract_feature' = False
            - dec_out
                - layout = 'NT'
                    Shape (batch_size, tgt_length, tgt_vocab_size)
                - layout = 'TN'
                    Shape (tgt_length, batch_size, tgt_vocab_size)
        """
        enc_out = self.encode(F, src_data, src_valid_length)
        contextual_embedding = self.decode_seq(F, tgt_data, tgt_valid_length, enc_out,
                                               src_valid_length)
        if self.extract_feature:
            if self.use_pooler:
                pooled_output = self.apply_pooling(F, contextual_embedding, tgt_valid_length)
                return contextual_embedding, pooled_output
            else:
                return contextual_embedding
        else:
            dec_out = self.tgt_final_layer(contextual_embedding)
            return dec_out

    def apply_pooling(self, F, sequence, valid_length):
        """Generate the representation given the inputs.

        This is used for pre-training or fine-tuning a BART model.
        In BART, the pooled output is the embedding of the last token.

        Parameters
        ----------
        F
            ndarray or symbol
        sequence
            - layout = 'NT'
                Shape (batch_size, sequence_length, units)
            - layout = 'TN'
                Shape (sequence_length, batch_size, units)
        valid_length
            Valid length of each sequence
            shape (batch_size,)

        Returns
        -------
        outputs
            Shape (batch_size, units)
        """
        if self._layout == 'NT':
            batch_indices = F.npx.arange_like(sequence, axis=0).astype(mx.np.int32)
            outputs = sequence[batch_indices, valid_length - 1]
        elif self._layout == 'TN':
            batch_indices = F.npx.arange_like(sequence, axis=1).astype(mx.np.int32)
            outputs = sequence[valid_length - 1, batch_indices]
        else:
            raise NotImplementedError
        if self.classifier_activation:
            return self.pooler(outputs)
        else:
            return outputs

    @property
    def layout(self) -> str:
        return self._layout

    @property
    def vocab_size(self):
        return self._vocab_size

    @classmethod
    def get_cfg(cls, key=None):
        if key is None:
            return bart_base()
        else:
            return bart_cfg_reg.create(key)

    @classmethod
    def from_cfg(cls, cfg,
                 dtype=None,
                 extract_feature=False,
                 use_pooler=True,
                 classifier_activation=False):
        """

        Parameters
        ----------
        cfg
            The configuration
        dtype
            Data type of the loaded config
        extract_feature
            Whether to only extract feature.
            If so, the output of the layer will be contextual embeddings or the
            contextual embedding + pooled output
        use_pooler
            Whether to use pooler
        classifier_activation
            Whether to use the classifier activation

        Returns
        -------
        model
            The initialized BartModel
        """
        cfg = cls.get_cfg().clone_merge(cfg)
        embed_initializer = mx.init.create(*cfg.INITIALIZER.embed)
        weight_initializer = mx.init.create(*cfg.INITIALIZER.weight)
        bias_initializer = mx.init.create(*cfg.INITIALIZER.bias)
        if dtype is None:
            dtype = cfg.MODEL.dtype
        return cls(src_vocab_size=cfg.MODEL.vocab_size,
                   tgt_vocab_size=cfg.MODEL.vocab_size,
                   max_src_length=cfg.MODEL.max_src_length,
                   max_tgt_length=cfg.MODEL.max_tgt_length,
                   scale_embed=cfg.MODEL.scale_embed,
                   pos_embed_type=cfg.MODEL.pos_embed_type,
                   shared_embed=cfg.MODEL.shared_embed,
                   tie_weights=cfg.MODEL.tie_weights,
                   data_norm=cfg.MODEL.data_norm,
                   extract_feature=extract_feature,
                   use_pooler=use_pooler,
                   classifier_activation=classifier_activation,
                   attention_dropout=cfg.MODEL.attention_dropout,
                   activation_dropout=cfg.MODEL.activation_dropout,
                   dropout=cfg.MODEL.dropout,
                   pooler_activation=cfg.MODEL.pooler_activation,
                   layer_norm_eps=cfg.MODEL.layer_norm_eps,
                   enc_num_layers=cfg.MODEL.ENCODER.num_layers,
                   enc_units=cfg.MODEL.ENCODER.units,
                   enc_num_heads=cfg.MODEL.ENCODER.num_heads,
                   enc_hidden_size=cfg.MODEL.ENCODER.hidden_size,
                   enc_recurrent=cfg.MODEL.ENCODER.recurrent,
                   enc_activation=cfg.MODEL.ENCODER.activation,
                   enc_pre_norm=cfg.MODEL.ENCODER.pre_norm,
                   dec_num_layers=cfg.MODEL.DECODER.num_layers,
                   dec_units=cfg.MODEL.DECODER.units,
                   dec_num_heads=cfg.MODEL.DECODER.num_heads,
                   dec_hidden_size=cfg.MODEL.DECODER.hidden_size,
                   dec_recurrent=cfg.MODEL.DECODER.recurrent,
                   dec_activation=cfg.MODEL.DECODER.activation,
                   dec_pre_norm=cfg.MODEL.DECODER.pre_norm,
                   layout=cfg.MODEL.layout,
                   embed_initializer=embed_initializer,
                   weight_initializer=weight_initializer,
                   bias_initializer=bias_initializer,
                   dtype=dtype)


def list_pretrained_bart():
    return sorted(list(PRETRAINED_URL.keys()))


def get_pretrained_bart(model_name: str = 'fairseq_bart_base',
                        root: str = get_model_zoo_home_dir(),
                        load_backbone: bool = True) \
        -> Tuple[CN, HuggingFaceByteBPETokenizer, str, List]:
    """Get the pretrained RoBERTa weights

    Parameters
    ----------
    model_name
        The name of the RoBERTa model.
    root
        The downloading root
    load_backbone
        Whether to load the weights of the backbone network

    Returns
    -------
    cfg
        Network configuration
    tokenizer
        The HuggingFaceByteBPETokenizer
    params_path
        Path to the parameters
    additional_output
        The additional outputs

    """
    assert model_name in PRETRAINED_URL, '{} is not found. All available are {}'.format(
        model_name, list_pretrained_bart())
    cfg_path = PRETRAINED_URL[model_name]['cfg']
    if isinstance(cfg_path, CN):
        cfg = cfg_path
    else:
        cfg = None
    merges_path = PRETRAINED_URL[model_name]['merges']
    vocab_path = PRETRAINED_URL[model_name]['vocab']
    params_path = PRETRAINED_URL[model_name]['params']

    local_paths = dict()
    download_jobs = [('vocab', vocab_path), ('merges', merges_path)]
    if cfg is None:
        download_jobs.append(('cfg', cfg_path))
    for k, path in download_jobs:
        local_paths[k] = download(url=get_repo_model_zoo_url() + path,
                                  path=os.path.join(root, path),
                                  sha1_hash=FILE_STATS[path])
    if load_backbone:
        local_params_path = download(url=get_repo_model_zoo_url() + params_path,
                                     path=os.path.join(root, params_path),
                                     sha1_hash=FILE_STATS[params_path])
    else:
        local_params_path = None

    do_lower = True if 'lowercase' in PRETRAINED_URL[model_name]\
                       and PRETRAINED_URL[model_name]['lowercase'] else False
    tokenizer = HuggingFaceByteBPETokenizer(
        merges_file=local_paths['merges'],
        vocab_file=local_paths['vocab'],
        lowercase=do_lower)
    additional_out = []
    if cfg is None:
        cfg = BartModel.get_cfg().clone_merge(local_paths['cfg'])
    return cfg, tokenizer, local_params_path, additional_out


BACKBONE_REGISTRY.register('bart', [BartModel,
                                    get_pretrained_bart,
                                    list_pretrained_bart])
