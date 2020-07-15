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
RoBERTa Model

@article{liu2019roberta,
    title = {RoBERTa: A Robustly Optimized BERT Pretraining Approach},
    author = {Yinhan Liu and Myle Ott and Naman Goyal and Jingfei Du and
              Mandar Joshi and Danqi Chen and Omer Levy and Mike Lewis and
              Luke Zettlemoyer and Veselin Stoyanov},
    journal={arXiv preprint arXiv:1907.11692},
    year = {2019},
}
"""

__all__ = ['RobertaModel', 'RobertaForMLM', 'list_pretrained_roberta', 'get_pretrained_roberta']

import os
from typing import Tuple

import mxnet as mx
from mxnet import use_np
from mxnet.gluon import HybridBlock, nn

from ..op import select_vectors_by_position
from ..base import (get_model_zoo_home_dir, get_repo_model_zoo_url,
                    get_model_zoo_checksum_dir)
from ..layers import PositionalEmbedding, get_activation
from ..registry import BACKBONE_REGISTRY
from ..utils.misc import download, load_checksum_stats
from .transformer import TransformerEncoderLayer
from ..initializer import TruncNorm
from ..utils.config import CfgNode as CN
from ..attention_cell import gen_self_attn_mask
from ..utils.registry import Registry
from ..data.tokenizers import HuggingFaceByteBPETokenizer

PRETRAINED_URL = {
    'fairseq_roberta_base': {
        'cfg': 'fairseq_roberta_base/model-565d1db7.yml',
        'merges': 'fairseq_roberta_base/gpt2-396d4d8e.merges',
        'vocab': 'fairseq_roberta_base/gpt2-f1335494.vocab',
        'params': 'fairseq_roberta_base/model-09a1520a.params',
        'mlm_params': 'fairseq_roberta_base/model_mlm-29889e2b.params',
        'lowercase': False,
    },
    'fairseq_roberta_large': {
        'cfg': 'fairseq_roberta_large/model-6e66dc4a.yml',
        'merges': 'fairseq_roberta_large/gpt2-396d4d8e.merges',
        'vocab': 'fairseq_roberta_large/gpt2-f1335494.vocab',
        'params': 'fairseq_roberta_large/model-6b043b91.params',
        'mlm_params': 'fairseq_roberta_large/model_mlm-119f38e1.params',
        'lowercase': False,
    }
}

FILE_STATS = load_checksum_stats(os.path.join(get_model_zoo_checksum_dir(), 'roberta.txt'))
roberta_cfg_reg = Registry('roberta_cfg')


@roberta_cfg_reg.register()
def roberta_base():
    cfg = CN()
    # Config for the roberta base model
    cfg.MODEL = CN()
    cfg.MODEL.vocab_size = 50265
    cfg.MODEL.units = 768
    cfg.MODEL.hidden_size = 3072
    cfg.MODEL.max_length = 512
    cfg.MODEL.num_heads = 12
    cfg.MODEL.num_layers = 12
    cfg.MODEL.pos_embed_type = 'learned'
    cfg.MODEL.activation = 'gelu'
    cfg.MODEL.pooler_activation = 'tanh'
    cfg.MODEL.layer_norm_eps = 1E-5
    cfg.MODEL.hidden_dropout_prob = 0.1
    cfg.MODEL.attention_dropout_prob = 0.1
    cfg.MODEL.dtype = 'float32'
    cfg.INITIALIZER = CN()
    cfg.INITIALIZER.embed = ['truncnorm', 0, 0.02]
    cfg.INITIALIZER.weight = ['truncnorm', 0, 0.02]
    cfg.INITIALIZER.bias = ['zeros']
    cfg.VERSION = 1
    cfg.freeze()
    return cfg


@roberta_cfg_reg.register()
def roberta_large():
    cfg = roberta_base()
    cfg.defrost()
    cfg.MODEL.units = 1024
    cfg.MODEL.hidden_size = 4096
    cfg.MODEL.num_heads = 16
    cfg.MODEL.num_layers = 24
    cfg.freeze()
    return cfg


@use_np
class RobertaModel(HybridBlock):
    def __init__(self,
                 vocab_size=50265,
                 units=768,
                 hidden_size=3072,
                 num_layers=12,
                 num_heads=12,
                 max_length=512,
                 hidden_dropout_prob=0.1,
                 attention_dropout_prob=0.1,
                 pos_embed_type='learned',
                 activation='gelu',
                 pooler_activation='tanh',
                 layer_norm_eps=1E-5,
                 embed_initializer=TruncNorm(stdev=0.02),
                 weight_initializer=TruncNorm(stdev=0.02),
                 bias_initializer='zeros',
                 dtype='float32',
                 use_pooler=True,
                 classifier_activation=False,
                 encoder_normalize_before=True,
                 output_all_encodings=False,
                 prefix=None,
                 params=None):
        """

        Parameters
        ----------
        vocab_size
        units
        hidden_size
        num_layers
        num_heads
        max_length
        hidden_dropout_prob
        attention_dropout_prob
        pos_embed_type
        activation
        pooler_activation
        layer_norm_eps
        embed_initializer
        weight_initializer
        bias_initializer
        dtype
        use_pooler
            Whether to output the CLS hidden state
        classifier_activation
            Whether to use classification head
        encoder_normalize_before
        output_all_encodings
        prefix
        params
        """
        super(RobertaModel, self).__init__(prefix=prefix, params=params)
        self._dtype = dtype
        self._output_all_encodings = output_all_encodings
        self.vocab_size = vocab_size
        self.units = units
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.pos_embed_type = pos_embed_type
        self.activation = activation
        self.pooler_activation = pooler_activation
        self.layer_norm_eps = layer_norm_eps
        self.use_pooler = use_pooler
        self.classifier_activation = classifier_activation
        self.encoder_normalize_before = encoder_normalize_before
        self.weight_initializer = weight_initializer
        self.embed_initializer = embed_initializer
        self.bias_initializer = bias_initializer

        with self.name_scope():
            self.word_embed = nn.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.units,
                weight_initializer=self.embed_initializer,
                dtype=self._dtype,
                prefix='word_embed_'
            )
            if self.encoder_normalize_before:
                self.embed_ln = nn.LayerNorm(
                    epsilon=self.layer_norm_eps,
                    in_channels=self.units,
                    prefix='embed_ln_'
                )
            self.embed_dropout = nn.Dropout(self.hidden_dropout_prob)
            self.pos_embed = PositionalEmbedding(
                units=self.units,
                max_length=self.max_length,
                dtype=self._dtype,
                method=pos_embed_type,
                prefix='pos_embed_'
            )

            self.encoder = RobertaEncoder(
                units=self.units,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                num_heads=self.num_heads,
                attention_dropout_prob=self.attention_dropout_prob,
                hidden_dropout_prob=self.hidden_dropout_prob,
                layer_norm_eps=self.layer_norm_eps,
                weight_initializer=self.weight_initializer,
                bias_initializer=self.bias_initializer,
                activation=self.activation,
                dtype=self._dtype,
                output_all_encodings=self._output_all_encodings
            )
            self.encoder.hybridize()

            if self.use_pooler and self.classifier_activation:
                # Construct pooler
                self.pooler = nn.Dense(units=self.units,
                                       in_units=self.units,
                                       flatten=False,
                                       activation=self.pooler_activation,
                                       weight_initializer=self.weight_initializer,
                                       bias_initializer=self.bias_initializer,
                                       prefix='pooler_')

    def hybrid_forward(self, F, tokens, valid_length):
        outputs = []
        embedding = self.get_initial_embedding(F, tokens)

        contextual_embeddings, additional_outputs = self.encoder(embedding, valid_length)
        outputs.append(contextual_embeddings)
        if self._output_all_encodings:
            contextual_embeddings = contextual_embeddings[-1]

        if self.use_pooler:
            pooled_out = self.apply_pooling(contextual_embeddings)
            outputs.append(pooled_out)

        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    def get_initial_embedding(self, F, inputs):
        """Get the initial token embeddings that considers the token type and positional embeddings

        Parameters
        ----------
        F
        inputs
            Shape (batch_size, seq_length)

        Returns
        -------
        embedding
            The initial embedding that will be fed into the encoder
        """
        embedding = self.word_embed(inputs)
        if self.pos_embed_type:
            positional_embedding = self.pos_embed(F.npx.arange_like(inputs, axis=1))
            positional_embedding = F.np.expand_dims(positional_embedding, axis=0)
            embedding = embedding + positional_embedding
        if self.encoder_normalize_before:
            embedding = self.embed_ln(embedding)
        embedding = self.embed_dropout(embedding)

        return embedding

    def apply_pooling(self, sequence):
        """Generate the representation given the inputs.

        This is used for pre-training or fine-tuning a mobile bert model.
        Get the first token of the whole sequence which is [CLS]

        sequence:
            Shape (batch_size, sequence_length, units)
        return:
            Shape (batch_size, units)
        """
        outputs = sequence[:, 0, :]
        if self.classifier_activation:
            return self.pooler(outputs)
        else:
            return outputs

    @staticmethod
    def get_cfg(key=None):
        if key:
            return roberta_cfg_reg.create(key)
        else:
            return roberta_base()

    @classmethod
    def from_cfg(cls,
                 cfg,
                 use_pooler=True,
                 classifier_activation=False,
                 encoder_normalize_before=True,
                 output_all_encodings=False,
                 prefix=None,
                 params=None):
        cfg = RobertaModel.get_cfg().clone_merge(cfg)
        embed_initializer = mx.init.create(*cfg.INITIALIZER.embed)
        weight_initializer = mx.init.create(*cfg.INITIALIZER.weight)
        bias_initializer = mx.init.create(*cfg.INITIALIZER.bias)
        return cls(vocab_size=cfg.MODEL.vocab_size,
                   units=cfg.MODEL.units,
                   hidden_size=cfg.MODEL.hidden_size,
                   num_layers=cfg.MODEL.num_layers,
                   num_heads=cfg.MODEL.num_heads,
                   max_length=cfg.MODEL.max_length,
                   hidden_dropout_prob=cfg.MODEL.hidden_dropout_prob,
                   attention_dropout_prob=cfg.MODEL.attention_dropout_prob,
                   pos_embed_type=cfg.MODEL.pos_embed_type,
                   activation=cfg.MODEL.activation,
                   pooler_activation=cfg.MODEL.pooler_activation,
                   layer_norm_eps=cfg.MODEL.layer_norm_eps,
                   embed_initializer=embed_initializer,
                   weight_initializer=weight_initializer,
                   bias_initializer=bias_initializer,
                   dtype=cfg.MODEL.dtype,
                   use_pooler=use_pooler,
                   encoder_normalize_before=encoder_normalize_before,
                   output_all_encodings=output_all_encodings,
                   prefix=prefix,
                   params=params)


@use_np
class RobertaEncoder(HybridBlock):
    def __init__(self,
                 units=768,
                 hidden_size=3072,
                 num_layers=12,
                 num_heads=12,
                 attention_dropout_prob=0.1,
                 hidden_dropout_prob=0.1,
                 layer_norm_eps=1E-5,
                 weight_initializer=TruncNorm(stdev=0.02),
                 bias_initializer='zeros',
                 activation='gelu',
                 dtype='float32',
                 output_all_encodings=False,
                 output_attention=False,
                 prefix='encoder_',
                 params=None):
        super(RobertaEncoder, self).__init__(prefix=prefix, params=params)
        self.units = units
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attention_dropout_prob = attention_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.activation = activation
        self._dtype = dtype
        self._output_all_encodings = output_all_encodings
        self._output_attention = output_attention
        with self.name_scope():
            self.all_layers = nn.HybridSequential(prefix='layers_')
            with self.all_layers.name_scope():
                for layer_idx in range(self.num_layers):
                    self.all_layers.add(
                        TransformerEncoderLayer(
                            units=self.units,
                            hidden_size=self.hidden_size,
                            num_heads=self.num_heads,
                            attention_dropout_prob=self.attention_dropout_prob,
                            hidden_dropout_prob=self.hidden_dropout_prob,
                            layer_norm_eps=self.layer_norm_eps,
                            weight_initializer=weight_initializer,
                            bias_initializer=bias_initializer,
                            activation=self.activation,
                            dtype=self._dtype,
                            prefix='{}_'.format(layer_idx)
                        )
                    )

    def hybrid_forward(self, F, x, valid_length):
        atten_mask = gen_self_attn_mask(F, x, valid_length,
                                        dtype=self._dtype, attn_type='full')
        all_encodings_outputs = [x]
        additional_outputs = []
        for layer_idx in range(self.num_layers):
            layer = self.all_layers[layer_idx]
            x, attention_weights = layer(x, atten_mask)
            if self._output_all_encodings:
                all_encodings_outputs.append(x)
            if self._output_attention:
                additional_outputs.append(attention_weights)
        # sequence_mask is not necessary here because masking could be performed in downstream tasks
        if self._output_all_encodings:
            return all_encodings_outputs, additional_outputs
        else:
            return x, additional_outputs


@use_np
class RobertaForMLM(HybridBlock):
    def __init__(self, backbone_cfg,
                 weight_initializer=None,
                 bias_initializer=None,
                 prefix=None,
                 params=None):
        """

        Parameters
        ----------
        backbone_cfg
        weight_initializer
        bias_initializer
        prefix
        params
        """
        super().__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.backbone_model = RobertaModel.from_cfg(backbone_cfg, prefix='')
            if weight_initializer is None:
                weight_initializer = self.backbone_model.weight_initializer
            if bias_initializer is None:
                bias_initializer = self.backbone_model.bias_initializer
            self.units = self.backbone_model.units
            self.mlm_decoder = nn.HybridSequential(prefix='mlm_')
            with self.mlm_decoder.name_scope():
                # Extra non-linear layer
                self.mlm_decoder.add(nn.Dense(units=self.units,
                                              in_units=self.units,
                                              flatten=False,
                                              weight_initializer=weight_initializer,
                                              bias_initializer=bias_initializer,
                                              prefix='proj_'))
                self.mlm_decoder.add(get_activation(self.backbone_model.activation))
                self.mlm_decoder.add(nn.LayerNorm(epsilon=self.backbone_model.layer_norm_eps,
                                                  in_channels=self.units,
                                                  prefix='ln_'))
                # only load the dense weights with a re-initialized bias
                # parameters are stored in 'word_embed_bias' which is
                # not used in original embedding
                self.mlm_decoder.add(
                    nn.Dense(
                        units=self.backbone_model.vocab_size,
                        in_units=self.units,
                        flatten=False,
                        params=self.backbone_model.word_embed.collect_params('.*weight'),
                        bias_initializer=bias_initializer,
                        prefix='score_'))
            self.mlm_decoder.hybridize()

    def hybrid_forward(self, F, inputs, valid_length, masked_positions):
        """Getting the scores of the masked positions.

        Parameters
        ----------
        F
        inputs :
            Shape (batch_size, seq_length)
        valid_length :
            The valid length of each sequence
            Shape (batch_size,)
        masked_positions :
            The masked position of the sequence
            Shape (batch_size, num_masked_positions).

        Returns
        -------
        contextual_embedding
            Shape (batch_size, seq_length, units).
        pooled_out
            Shape (batch_size, units)
        mlm_scores :
            Shape (batch_size, num_masked_positions, vocab_size)
        """

        all_encodings_outputs, pooled_out = self.backbone_model(inputs, valid_length)
        if self.backbone_model._output_all_encodings:
            contextual_embeddings = all_encodings_outputs[-1]
        else:
            contextual_embeddings = all_encodings_outputs
        mlm_features = select_vectors_by_position(F, contextual_embeddings, masked_positions)
        mlm_scores = self.mlm_decoder(mlm_features)
        return all_encodings_outputs, pooled_out, mlm_scores


def list_pretrained_roberta():
    return sorted(list(PRETRAINED_URL.keys()))


def get_pretrained_roberta(model_name: str = 'fairseq_roberta_base',
                           root: str = get_model_zoo_home_dir(),
                           load_backbone: bool = True,
                           load_mlm: bool = False) \
        -> Tuple[CN, HuggingFaceByteBPETokenizer, str]:
    """Get the pretrained RoBERTa weights

    Parameters
    ----------
    model_name
        The name of the RoBERTa model.
    root
        The downloading root
    load_backbone
        Whether to load the weights of the backbone network
    load_mlm
        Whether to load the weights of MLM

    Returns
    -------
    cfg
        Network configuration
    tokenizer
        The HuggingFaceByteBPETokenizer
    params_path
        Path to the parameters
    mlm_params_path
        Path to the parameter that includes both the backbone and the MLM
    """
    assert model_name in PRETRAINED_URL, '{} is not found. All available are {}'.format(
        model_name, list_pretrained_roberta())
    cfg_path = PRETRAINED_URL[model_name
    ]['cfg']
    merges_path = PRETRAINED_URL[model_name]['merges']
    vocab_path = PRETRAINED_URL[model_name]['vocab']
    params_path = PRETRAINED_URL[model_name]['params']
    mlm_params_path = PRETRAINED_URL[model_name]['mlm_params']

    local_paths = dict()
    for k, path in [('cfg', cfg_path), ('vocab', vocab_path),
                    ('merges', merges_path)]:
        local_paths[k] = download(url=get_repo_model_zoo_url() + path,
                                  path=os.path.join(root, path),
                                  sha1_hash=FILE_STATS[path])
    if load_backbone:
        local_params_path = download(url=get_repo_model_zoo_url() + params_path,
                                     path=os.path.join(root, params_path),
                                     sha1_hash=FILE_STATS[params_path])
    else:
        local_params_path = None
    if load_mlm and mlm_params_path is not None:
        local_mlm_params_path = download(url=get_repo_model_zoo_url() + mlm_params_path,
                                         path=os.path.join(root, mlm_params_path),
                                         sha1_hash=FILE_STATS[mlm_params_path])
    else:
        local_mlm_params_path = None
    do_lower = True if 'lowercase' in PRETRAINED_URL[model_name]\
                       and PRETRAINED_URL[model_name]['lowercase'] else False
    tokenizer = HuggingFaceByteBPETokenizer(
                    merges_file=local_paths['merges'],
                    vocab_file=local_paths['vocab'],
                    lowercase=do_lower)
    cfg = RobertaModel.get_cfg().clone_merge(local_paths['cfg'])
    return cfg, tokenizer, local_params_path, local_mlm_params_path


BACKBONE_REGISTRY.register('roberta', [RobertaModel,
                                       get_pretrained_roberta,
                                       list_pretrained_roberta])
