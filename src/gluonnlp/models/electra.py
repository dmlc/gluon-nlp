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
"""Electra Model.

@inproceedings{clark2020electra,
  title = {{ELECTRA}: Pre-training Text Encoders as Discriminators Rather Than Generators},
  author = {Kevin Clark and Minh-Thang Luong and Quoc V. Le and Christopher D. Manning},
  booktitle = {ICLR},
  year = {2020},
  url = {https://openreview.net/pdf?id=r1xMH1BtvB}
}

"""
__all__ = ['ElectraModel', 'ElectraDiscriminator', 'ElectraGenerator',
           'ElectraForPretrain', 'list_pretrained_electra', 'get_pretrained_electra']

import os
from typing import Tuple, Optional, List

import mxnet as mx
import numpy as np
from mxnet import use_np
from mxnet.gluon import HybridBlock, nn
from ..registry import BACKBONE_REGISTRY
from ..op import gumbel_softmax, select_vectors_by_position, add_vectors_by_position, update_vectors_by_position
from ..base import get_model_zoo_home_dir, get_repo_model_zoo_url, get_model_zoo_checksum_dir
from ..layers import PositionalEmbedding, get_activation
from .transformer import TransformerEncoderLayer
from ..initializer import TruncNorm
from ..utils.config import CfgNode as CN
from ..utils.misc import load_checksum_stats, download
from ..utils.registry import Registry
from ..attention_cell import gen_self_attn_mask
from ..data.tokenizers import HuggingFaceWordPieceTokenizer

electra_cfg_reg = Registry('electra_cfg')


def get_generator_cfg(model_config):
    """
    Get the generator configuration from the Electra model config.
    The size of generator is usually smaller than discriminator but same in electra small,
    which exists  a conflict between source code and original paper.
    """
    generator_cfg = model_config.clone()
    generator_layers_scale = model_config.MODEL.generator_layers_scale
    generator_units_scale = model_config.MODEL.generator_units_scale
    generator_cfg.defrost()
    # the round function is used to slove int(0.3333*768)!=256 for electra base
    generator_cfg.MODEL.units = round(generator_units_scale * model_config.MODEL.units)
    generator_cfg.MODEL.hidden_size = round(generator_units_scale * model_config.MODEL.hidden_size)
    generator_cfg.MODEL.num_heads = round(generator_units_scale * model_config.MODEL.num_heads)
    generator_cfg.MODEL.num_layers = round(generator_layers_scale * model_config.MODEL.num_layers)
    generator_cfg.freeze()
    return generator_cfg


@electra_cfg_reg.register()
def google_electra_small():
    cfg = CN()
    # Model
    cfg.MODEL = CN()
    cfg.MODEL.vocab_size = 30522
    cfg.MODEL.embed_size = 128
    cfg.MODEL.units = 256
    cfg.MODEL.hidden_size = 1024
    cfg.MODEL.max_length = 512
    cfg.MODEL.num_heads = 4
    cfg.MODEL.num_layers = 12
    cfg.MODEL.pos_embed_type = 'learned'
    cfg.MODEL.activation = 'gelu'
    cfg.MODEL.layer_norm_eps = 1E-12
    cfg.MODEL.num_token_types = 2
    # Dropout regularization
    cfg.MODEL.hidden_dropout_prob = 0.1
    cfg.MODEL.attention_dropout_prob = 0.1
    cfg.MODEL.dtype = 'float32'
    # Layout flags
    cfg.MODEL.layout = 'NT'
    cfg.MODEL.compute_layout = 'auto'
    # Generator hyper-parameters
    cfg.MODEL.generator_layers_scale = 1.0
    cfg.MODEL.generator_units_scale = 1.0
    # Initializer
    cfg.INITIALIZER = CN()
    cfg.INITIALIZER.embed = ['truncnorm', 0, 0.02]
    cfg.INITIALIZER.weight = ['truncnorm', 0, 0.02]  # TruncNorm(0, 0.02)
    cfg.INITIALIZER.bias = ['zeros']
    cfg.VERSION = 1
    cfg.freeze()
    return cfg


@electra_cfg_reg.register()
def google_electra_base():
    cfg = google_electra_small()
    cfg.defrost()
    cfg.MODEL.embed_size = 768
    cfg.MODEL.units = 768
    cfg.MODEL.hidden_size = 3072
    cfg.MODEL.num_heads = 12
    cfg.MODEL.num_layers = 12
    cfg.MODEL.generator_units_scale = 0.33333
    cfg.freeze()
    return cfg


@electra_cfg_reg.register()
def google_electra_large():
    cfg = google_electra_small()
    cfg.defrost()
    cfg.MODEL.embed_size = 1024
    cfg.MODEL.units = 1024
    cfg.MODEL.hidden_size = 4096
    cfg.MODEL.num_heads = 16
    cfg.MODEL.num_layers = 24
    cfg.MODEL.generator_units_scale = 0.25
    cfg.freeze()
    return cfg


PRETRAINED_URL = {
    'google_electra_small': {
        'cfg': google_electra_small(),
        'vocab': 'google_electra_small/vocab-e6d2b21d.json',
        'params': 'google_electra_small/model-2654c8b4.params',
        'disc_model': 'google_electra_small/disc_model-137714b6.params',
        'gen_model': 'google_electra_small/gen_model-0c30d1c5.params',
        'lowercase': True,
    },
    'google_electra_base': {
        'cfg': google_electra_base(),
        'vocab': 'google_electra_base/vocab-e6d2b21d.json',
        'params': 'google_electra_base/model-31c235cc.params',
        'disc_model': 'google_electra_base/disc_model-514bd353.params',
        'gen_model': 'google_electra_base/gen_model-253a62c9.params',
        'lowercase': True,
    },
    'google_electra_large': {
        'cfg': google_electra_large(),
        'vocab': 'google_electra_large/vocab-e6d2b21d.json',
        'params': 'google_electra_large/model-9baf9ff5.params',
        'disc_model': 'google_electra_large/disc_model-5b820c02.params',
        'gen_model': 'google_electra_large/gen_model-82c1b17b.params',
        'lowercase': True,
    },
    'gluon_electra_small_owt': {
        'cfg': 'gluon_electra_small_owt/model-6e276d98.yml',
        'vocab': 'gluon_electra_small_owt/vocab-e6d2b21d.json',
        'params': 'gluon_electra_small_owt/model-e9636891.params',
        'disc_model': 'gluon_electra_small_owt/disc_model-87836017.params',
        'gen_model': 'gluon_electra_small_owt/gen_model-45a6fb67.params',
        'lowercase': True,
    }
}

FILE_STATS = load_checksum_stats(os.path.join(get_model_zoo_checksum_dir(), 'electra.txt'))


# TODO(sxjscience) Use BertTransformer
@use_np
class ElectraEncoder(HybridBlock):
    def __init__(self, units=512,
                 hidden_size=2048,
                 num_layers=6,
                 num_heads=8,
                 attention_dropout_prob=0.,
                 hidden_dropout_prob=0.,
                 output_attention=False,
                 dtype='float32',
                 output_all_encodings=False,
                 layer_norm_eps=1E-12,
                 weight_initializer=TruncNorm(stdev=0.02),
                 bias_initializer='zeros',
                 activation='gelu',
                 layout='NT'):
        """

        Parameters
        ----------
        units
            The number of units
        hidden_size
            The hidden size
        num_layers
            Number of layers
        num_heads
            Number of heads
        attention_dropout_prob
            Dropout probability of the attention layer
        hidden_dropout_prob
            Dropout probability
        output_attention
            Whether to output the attention weights
        dtype
            Data type of the weights
        output_all_encodings
        layer_norm_eps
        weight_initializer
        bias_initializer
        activation
        layout
        """
        super().__init__()
        assert units % num_heads == 0, \
            'In ElectraEncoder, The units should be divisible ' \
            'by the number of heads. Received units={}, num_heads={}' \
            .format(units, num_heads)

        self._dtype = dtype
        self._layout = layout
        self._num_layers = num_layers

        self._output_attention = output_attention
        self._output_all_encodings = output_all_encodings

        self.all_encoder_layers = nn.HybridSequential()
        for layer_idx in range(num_layers):
            self.all_encoder_layers.add(
                TransformerEncoderLayer(units=units,
                                        hidden_size=hidden_size,
                                        num_heads=num_heads,
                                        attention_dropout_prob=attention_dropout_prob,
                                        hidden_dropout_prob=hidden_dropout_prob,
                                        layer_norm_eps=layer_norm_eps,
                                        weight_initializer=weight_initializer,
                                        bias_initializer=bias_initializer,
                                        activation=activation,
                                        dtype=dtype,
                                        layout=layout))

    @property
    def layout(self):
        return self._layout

    def hybrid_forward(self, F, data, valid_length):
        """
        Generate the representation given the inputs.

        This is used in training or fine-tuning a Electra model.

        Parameters
        ----------
        F
        data
            - layout = 'NT'
                Shape (batch_size, seq_length, C)
            - layout = 'TN'
                Shape (seq_length, batch_size, C)
        valid_length
            Shape (batch_size,)

        Returns
        -------
        out
            - layout = 'NT'
                Shape (batch_size, seq_length, C_out)
            - layout = 'TN'
                Shape (seq_length, batch_size, C_out)
        """
        if self.layout == 'NT':
            time_axis, batch_axis = 1, 0
        else:
            time_axis, batch_axis = 0, 1
        # 1. Embed the data
        attn_mask = gen_self_attn_mask(F, data, valid_length,
                                       dtype=self._dtype,
                                       layout=self._layout,
                                       attn_type='full')
        out = data
        all_encodings_outputs = []
        additional_outputs = []
        for layer_idx in range(self._num_layers):
            layer = self.all_encoder_layers[layer_idx]
            out, attention_weights = layer(out, attn_mask)
            # out : [batch_size, seq_len, units]
            # attention_weights : [batch_size, num_heads, seq_len, seq_len]
            if self._output_all_encodings:
                out = F.npx.sequence_mask(out,
                                          sequence_length=valid_length,
                                          use_sequence_length=True,
                                          axis=time_axis)
                all_encodings_outputs.append(out)

            if self._output_attention:
                additional_outputs.append(attention_weights)

        if not self._output_all_encodings:
            # if self._output_all_encodings, SequenceMask is already applied above
            out = F.npx.sequence_mask(out, sequence_length=valid_length,
                                      use_sequence_length=True, axis=time_axis)
            return out, additional_outputs
        else:
            return all_encodings_outputs, additional_outputs


@use_np
class ElectraModel(HybridBlock):
    """Electra Model

    This is almost the same as bert model with embedding_size adjustable (factorized embedding).
    """

    def __init__(self,
                 vocab_size=30000,
                 units=768,
                 hidden_size=3072,
                 embed_size=128,
                 num_layers=12,
                 num_heads=12,
                 max_length=512,
                 hidden_dropout_prob=0.,
                 attention_dropout_prob=0.,
                 num_token_types=2,
                 pos_embed_type='learned',
                 activation='gelu',
                 layer_norm_eps=1E-12,
                 embed_initializer=TruncNorm(stdev=0.02),
                 weight_initializer=TruncNorm(stdev=0.02),
                 bias_initializer='zeros',
                 dtype='float32',
                 use_pooler=True,
                 layout='NT',
                 compute_layout='auto'):
        super().__init__()
        self._dtype = dtype
        self.use_pooler = use_pooler
        self.pos_embed_type = pos_embed_type
        self.num_token_types = num_token_types
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.units = units
        self.max_length = max_length
        self.activation = activation
        self.embed_initializer = embed_initializer
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.layer_norm_eps = layer_norm_eps
        self._layout = layout
        if compute_layout is None or compute_layout == 'auto':
            self._compute_layout = layout
        else:
            self._compute_layout = compute_layout
        # Construct ElectraEncoder
        self.encoder = ElectraEncoder(
            units=units,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            attention_dropout_prob=attention_dropout_prob,
            hidden_dropout_prob=hidden_dropout_prob,
            output_attention=False,
            output_all_encodings=False,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
            dtype=dtype,
            layout=self._compute_layout,
        )
        self.encoder.hybridize()

        self.word_embed = nn.Embedding(input_dim=vocab_size,
                                       output_dim=embed_size,
                                       weight_initializer=embed_initializer,
                                       dtype=dtype)
        # Construct token type embedding
        self.token_type_embed = nn.Embedding(input_dim=num_token_types,
                                             output_dim=embed_size,
                                             weight_initializer=weight_initializer)
        self.token_pos_embed = PositionalEmbedding(units=embed_size,
                                                   max_length=max_length,
                                                   dtype=self._dtype,
                                                   method=pos_embed_type)
        self.embed_layer_norm = nn.LayerNorm(epsilon=self.layer_norm_eps,
                                             in_channels=embed_size)

        self.embed_dropout = nn.Dropout(hidden_dropout_prob)
        if embed_size != units:
            self.embed_factorized_proj = nn.Dense(units=units,
                                                  in_units=embed_size,
                                                  flatten=False,
                                                  weight_initializer=weight_initializer,
                                                  bias_initializer=bias_initializer)

    @property
    def layout(self):
        return self._layout

    def hybrid_forward(self, F, inputs, token_types, valid_length=None):
        # pylint: disable=arguments-differ
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a Electra model.

        Parameters
        ----------
        F
        inputs
            - layout = 'NT'
                Shape (batch_size, seq_length)
            - layout = 'TN'
                Shape (seq_length, batch_size)
        token_types
            - layout = 'NT'
                Shape (batch_size, seq_length)
            - layout = 'TN'
                Shape (seq_length, batch_size)

            If the inputs contain two sequences, we will set different token types for the first
             sentence and the second sentence.
        valid_length
            The valid length of each sequence
            Shape (batch_size,)

        Returns
        -------
        contextual_embedding
            - layout = 'NT'
                Shape (batch_size, seq_length, units).
            - layout = 'TN'
                Shape (seq_length, batch_size, units).
        pooled_output
            This is optional. Shape (batch_size, units)
        """
        initial_embedding = self.get_initial_embedding(F, inputs, token_types)
        # Projecting the embedding into units
        prev_out = initial_embedding
        if self.embed_size != self.units:
            prev_out = self.embed_factorized_proj(prev_out)
        outputs = []
        if self._compute_layout != self._layout:
            # Swap the axes if the compute_layout and layout mismatch
            contextual_embeddings, additional_outputs = self.encoder(F.np.swapaxes(prev_out, 0, 1),
                                                                     valid_length)
            contextual_embeddings = F.np.swapaxes(contextual_embeddings, 0, 1)
        else:
            contextual_embeddings, additional_outputs = self.encoder(prev_out, valid_length)
        outputs.append(contextual_embeddings)
        if self.use_pooler:
            # Here we just get the first token ([CLS]) without any pooling strategy,
            # which is slightly different from bert model with the pooled_out
            # the attribute name is keeping the same as bert and albert model with defualt
            # use_pooler=True
            if self._layout == 'NT':
                pooled_out = contextual_embeddings[:, 0, :]
            else:
                pooled_out = contextual_embeddings[0, :, :]
            outputs.append(pooled_out)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    #TODO(sxjscience) Move to a `common.py`
    def get_initial_embedding(self, F, inputs, token_types=None):
        """Get the initial token embeddings that considers the token type and positional embeddings

        Parameters
        ----------
        F
        inputs
            - layout = 'NT'
                Shape (batch_size, seq_length)
            - layout = 'TN'
                Shape (seq_length, batch_size)
        token_types
            - layout = 'NT'
                Shape (batch_size, seq_length)
            - layout = 'TN'
                Shape (seq_length, batch_size)
            If None, it will be initialized as all zero

        Returns
        -------
        embedding
            The initial embedding that will be fed into the encoder
            - layout = 'NT'
                Shape (batch_size, seq_length, C_embed)
            - layout = 'TN'
                Shape (seq_length, batch_size, C_embed)
        """
        if self.layout == 'NT':
            time_axis, batch_axis = 1, 0
        else:
            time_axis, batch_axis = 0, 1
        embedding = self.word_embed(inputs)
        if token_types is None:
            token_types = F.np.zeros_like(inputs)
        type_embedding = self.token_type_embed(token_types)
        embedding = embedding + type_embedding
        if self.pos_embed_type is not None:
            positional_embedding = self.token_pos_embed(F.npx.arange_like(inputs, axis=time_axis))
            positional_embedding = F.np.expand_dims(positional_embedding, axis=batch_axis)
            embedding = embedding + positional_embedding
        # Extra layer normalization plus dropout
        embedding = self.embed_layer_norm(embedding)
        embedding = self.embed_dropout(embedding)
        return embedding

    def apply_layerwise_decay(self, layerwise_decay: int,
                              not_included: Optional[List[str]] = None,
                              num_additional_layers: int = 2):
        """Apply the layer-wise gradient decay

        .. math::
            lr = lr * layerwise_decay^(max_depth - layer_depth)

        Parameters:
        ----------
        layerwise_decay
            Power rate of the layer-wise decay
        not_included
            A list or parameter names that not included in the layer-wise decay
        num_additional_layers
            The number of layers after the current backbone. This helps determine the max depth
        """

        # Consider the task specific finetuning layer as the last layer, following with pooler
        # In addition, the embedding parameters have the smaller learning rate based on this
        # setting.
        max_depth = self.num_layers + num_additional_layers
        for _, value in self.collect_params('.*embed*').items():
            value.lr_mult = layerwise_decay ** max_depth

        for (layer_depth, layer) in enumerate(self.encoder.all_encoder_layers):
            layer_params = layer.collect_params()
            for key, value in layer_params.items():
                if not_included:
                    for pn in not_included:
                        if pn in key:
                            continue
                value.lr_mult = layerwise_decay**(max_depth - (layer_depth + 1))

    def frozen_params(self, untunable_depth, not_included=None):
        """Froze part of parameters according to layer depth.

        That is, make all layer that shallower than `untunable_depth` untunable
        to stop the gradient backward computation and accelerate the training.

        Parameters:
        ----------
        untunable_depth: int
            the depth of the neural network starting from 1 to number of layers
        not_included: list of str
            A list or parameter names that not included in the untunable parameters
        """
        all_layers = self.encoder.all_encoder_layers
        for _, value in self.collect_params('.*embed*').items():
            value.grad_req = 'null'

        for layer in all_layers[:untunable_depth]:
            for key, value in layer.collect_params().items():
                if not_included:
                    for pn in not_included:
                        if pn in key:
                            continue
                value.grad_req = 'null'

    @staticmethod
    def get_cfg(key=None):
        if key is not None:
            return electra_cfg_reg.create(key)
        else:
            return google_electra_base()

    @classmethod
    def from_cfg(cls, cfg, use_pooler=True, dtype=None) -> 'ElectraModel':
        cfg = ElectraModel.get_cfg().clone_merge(cfg)
        assert cfg.VERSION == 1, 'Wrong version!'
        embed_initializer = mx.init.create(*cfg.INITIALIZER.embed)
        weight_initializer = mx.init.create(*cfg.INITIALIZER.weight)
        bias_initializer = mx.init.create(*cfg.INITIALIZER.bias)
        if dtype is None:
            dtype = cfg.MODEL.dtype
        return cls(vocab_size=cfg.MODEL.vocab_size,
                   units=cfg.MODEL.units,
                   hidden_size=cfg.MODEL.hidden_size,
                   embed_size=cfg.MODEL.embed_size,
                   num_layers=cfg.MODEL.num_layers,
                   num_heads=cfg.MODEL.num_heads,
                   max_length=cfg.MODEL.max_length,
                   hidden_dropout_prob=cfg.MODEL.hidden_dropout_prob,
                   attention_dropout_prob=cfg.MODEL.attention_dropout_prob,
                   num_token_types=cfg.MODEL.num_token_types,
                   pos_embed_type=cfg.MODEL.pos_embed_type,
                   activation=cfg.MODEL.activation,
                   layer_norm_eps=cfg.MODEL.layer_norm_eps,
                   dtype=dtype,
                   embed_initializer=embed_initializer,
                   weight_initializer=weight_initializer,
                   bias_initializer=bias_initializer,
                   use_pooler=use_pooler,
                   layout=cfg.MODEL.layout,
                   compute_layout=cfg.MODEL.compute_layout)


@use_np
class ElectraDiscriminator(HybridBlock):
    """
    It is slightly different from the traditional mask language model which recover the
    masked word (find the matched word in dictionary). The Object of Discriminator in
    Electra is 'replaced token detection' that is a binary classification task to
    predicts every token whether it is an original or a replacement.
    """

    def __init__(self, backbone_cfg,
                 weight_initializer=None,
                 bias_initializer=None):
        """

        Parameters
        ----------
        backbone_cfg
        weight_initializer
        bias_initializer
        """
        super().__init__()
        self.backbone_model = ElectraModel.from_cfg(backbone_cfg)
        if weight_initializer is None:
            weight_initializer = self.backbone_model.weight_initializer
        if bias_initializer is None:
            bias_initializer = self.backbone_model.bias_initializer
        self.rtd_encoder = nn.HybridSequential()
        # Extra non-linear layer
        self.rtd_encoder.add(nn.Dense(units=self.backbone_model.units,
                                      in_units=self.backbone_model.units,
                                      flatten=False,
                                      weight_initializer=weight_initializer,
                                      bias_initializer=bias_initializer))
        self.rtd_encoder.add(get_activation(self.backbone_model.activation))
        self.rtd_encoder.add(nn.Dense(units=1,
                                      in_units=self.backbone_model.units,
                                      flatten=False,
                                      weight_initializer=weight_initializer,
                                      bias_initializer=bias_initializer))
        self.rtd_encoder.hybridize()

    def hybrid_forward(self, F, inputs, token_types, valid_length):
        """Getting the scores of the replaced token detection of the whole sentence
        based on the corrupted tokens produced from a generator.

        Parameters
        ----------
        F
        inputs
            - layout = 'NT'
                Shape (batch_size, seq_length)
            - layout = 'TN'
                Shape (seq_length, batch_size)
        token_types
            - layout = 'NT'
                Shape (batch_size, seq_length)
            - layout = 'TN'
                Shape (seq_length, batch_size)

            If the inputs contain two sequences, we will set different token types for the first
             sentence and the second sentence.
        valid_length
            The valid length of each sequence
            Shape (batch_size,)

        Returns
        -------
        contextual_embedding
            - layout = 'NT'
                Shape (batch_size, seq_length, units).
            - layout = 'TN'
                Shape (seq_length, batch_size, units).
        pooled_out
            Shape (batch_size, units)
        rtd_scores
            - layout = 'NT'
                Shape (batch_size, seq_length)
            - layout = 'TN'
                Shape (seq_length, batch_size)
        """
        contextual_embeddings, pooled_out = self.backbone_model(inputs, token_types, valid_length)
        rtd_scores = self.rtd_encoder(contextual_embeddings).squeeze(-1)
        return contextual_embeddings, pooled_out, rtd_scores


@use_np
class ElectraGenerator(HybridBlock):
    """
    This is a typical mlm model whose size is usually the 1/4 - 1/2 of the discriminator.
    """

    def __init__(self, backbone_cfg,
                 weight_initializer=None,
                 bias_initializer=None):
        """

        Parameters
        ----------
        backbone_cfg
            Configuration of the backbone model
        weight_initializer
        bias_initializer
        """
        super().__init__()
        self.backbone_model = ElectraModel.from_cfg(backbone_cfg)
        if weight_initializer is None:
            weight_initializer = self.backbone_model.weight_initializer
        if bias_initializer is None:
            bias_initializer = self.backbone_model.bias_initializer
        self.mlm_decoder = nn.HybridSequential()
        # Extra non-linear layer
        self.mlm_decoder.add(nn.Dense(units=self.backbone_model.embed_size,
                                      in_units=self.backbone_model.units,
                                      flatten=False,
                                      weight_initializer=weight_initializer,
                                      bias_initializer=bias_initializer))
        self.mlm_decoder.add(get_activation(self.backbone_model.activation))
        self.mlm_decoder.add(nn.LayerNorm(epsilon=self.backbone_model.layer_norm_eps,
                                          in_channels=self.backbone_model.embed_size))
        # only load the dense weights with a re-initialized bias
        # parameters are stored in 'word_embed_bias' which is
        # not used in original embedding
        self.mlm_decoder.add(
            nn.Dense(
                units=self.backbone_model.vocab_size,
                in_units=self.backbone_model.embed_size,
                flatten=False,
                bias_initializer=bias_initializer))
        self.mlm_decoder[-1].weight = self.backbone_model.word_embed.weight
        self.mlm_decoder.hybridize()

    # TODO(sxjscience,zheyu) Should design a better API
    def tie_embeddings(self, word_embed_params=None,
                       token_type_embed_params=None,
                       token_pos_embed_params=None,
                       embed_layer_norm_params=None):
        """Tie the embedding layers between the backbone and the MLM decoder

        Parameters
        ----------
        word_embed_params
        token_type_embed_params
        token_pos_embed_params
        embed_layer_norm_params

        """
        self.backbone_model.word_embed.share_parameters(word_embed_params)
        self.mlm_decoder[-1].share_parameters(word_embed_params)
        self.backbone_model.token_type_embed.share_parameters(token_type_embed_params)
        self.backbone_model.token_pos_embed.share_parameters(token_pos_embed_params)
        self.backbone_model.embed_layer_norm.share_parameters(embed_layer_norm_params)

    def hybrid_forward(self, F, inputs, token_types, valid_length, masked_positions):
        """Getting the scores of the masked positions.

        Parameters
        ----------
        F
        inputs
            - layout = 'NT'
                Shape (batch_size, seq_length)
            - layout = 'TN'
                Shape (seq_length, batch_size)
        token_types
            - layout = 'NT'
                Shape (batch_size, seq_length)
            - layout = 'TN'
                Shape (seq_length, batch_size)

            If the inputs contain two sequences, we will set different token types for the first
             sentence and the second sentence.
        valid_length :
            The valid length of each sequence
            Shape (batch_size,)
        masked_positions :
            The masked position of the sequence
            Shape (batch_size, num_masked_positions).

        Returns
        -------
        contextual_embedding
            - layout = 'NT'
                Shape (batch_size, seq_length, units).
            - layout = 'TN'
                Shape (seq_length, batch_size, units).
        pooled_out
            Shape (batch_size, units)
        mlm_scores :
            Shape (batch_size, num_masked_positions, vocab_size)
        """
        contextual_embeddings, pooled_out = self.backbone_model(inputs, token_types, valid_length)
        if self.backbone_model.layout == 'NT':
            mlm_features = select_vectors_by_position(F, contextual_embeddings, masked_positions)
        else:
            mlm_features = select_vectors_by_position(F, F.np.swapaxes(contextual_embeddings, 0, 1),
                                                      masked_positions)
        mlm_scores = self.mlm_decoder(mlm_features)
        return contextual_embeddings, pooled_out, mlm_scores


@use_np
class ElectraForPretrain(HybridBlock):
    """
    An integrated model combined with a generator and a discriminator.  Generator here
    produces a corrupted tokens playing as fake data to fool a discriminator whose
    objective is to distinguish whether each token in the input sentence it accepts
    is the same as the original. It is a classification task instead of prediction
    task as other pretrained models such as bert.
    """

    def __init__(self,
                 disc_cfg,
                 uniform_generator=False,
                 tied_generator=False,
                 tied_embeddings=True,
                 disallow_correct=False,
                 temperature=1.0,
                 gumbel_eps=1E-9,
                 dtype='float32',
                 weight_initializer=None,
                 bias_initializer=None):
        """

        Parameters
        ----------
        disc_cfg :
            Config for discriminator model including scaled size for generator
        uniform_generator :
            Wether to get a generator with uniform weights, the mlm_scores from
            which are totally random. In this case , a discriminator learns from
            a random 15% of the input tokens distinct from the subset.
        tied_generator :
            Whether to tie backbone model weights of generator and discriminator.
            The size of G and D are required to be same if set to True.
        tied_embeddings :
            Whether to tie the embeddings of generator and discriminator
        disallow_correct :
            Whether the correct smaples of generator are allowed,
            that is 15% of tokens are always fake.
        temperature :
            Temperature of gumbel distribution for sampling from generator
        weight_initializer
        bias_initializer
        """
        super().__init__()
        self._uniform_generator = uniform_generator
        self._tied_generator = tied_generator
        self._tied_embeddings = tied_embeddings
        self._disallow_correct = disallow_correct
        self._temperature = temperature
        self._gumbel_eps = gumbel_eps
        self._dtype = dtype

        self.disc_cfg = disc_cfg
        self.vocab_size = disc_cfg.MODEL.vocab_size
        self.gen_cfg = get_generator_cfg(disc_cfg)
        self.discriminator = ElectraDiscriminator(disc_cfg,
                                                  weight_initializer=weight_initializer,
                                                  bias_initializer=bias_initializer)
        self.disc_backbone = self.discriminator.backbone_model

        if not uniform_generator and not tied_generator:
            self.generator = ElectraGenerator(self.gen_cfg,
                                              weight_initializer=weight_initializer,
                                              bias_initializer=bias_initializer)
            if tied_embeddings:
                self.generator.tie_embeddings(self.disc_backbone.word_embed.collect_params(),
                                              self.disc_backbone.token_type_embed.collect_params(),
                                              self.disc_backbone.token_pos_embed.collect_params(),
                                              self.disc_backbone.embed_layer_norm.collect_params())
            self.generator.hybridize()

        elif tied_generator:
            # Reuse the weight of the discriminator backbone model
            self.generator = ElectraGenerator(self.gen_cfg,
                                              weight_initializer=weight_initializer,
                                              bias_initializer=bias_initializer)
            # TODO(sxjscience, zheyu) Verify
            self.generator.backbone_model = self.disc_backbone
            self.generator.hybridize()
        elif uniform_generator:
            # get the mlm_scores randomly over vocab
            self.generator = None

        self.discriminator.hybridize()

    def hybrid_forward(self, F, inputs, token_types, valid_length,
                       original_tokens, masked_positions):
        """Getting the mlm scores of each masked positions from a generator,
        then produces the corrupted tokens sampling from a gumbel distribution.
        We also get the ground-truth and scores of the replaced token detection
        which is output by a discriminator. The ground-truth is an array with same
        shape as the input using 1 stand for original token and 0 for replacement.

        Notice: There is a problem when the masked positions have duplicate indexs.
        Try to avoid that in the data preprocessing process. In addition, loss calculation
        should be done in the training scripts as well.

        Parameters
        ----------
        F
        inputs
            The masked input
            - layout = 'NT'
                Shape (batch_size, seq_length)
            - layout = 'TN'
                Shape (seq_length, batch_size)
        token_types
            The token types.
            - layout = 'NT'
                Shape (batch_size, seq_length)
            - layout = 'TN'
                Shape (seq_length, batch_size)

            If the inputs contain two sequences, we will set different token types for the first
             sentence and the second sentence.
        valid_length
            The valid length of each sequence.
            Shape (batch_size,)
        original_tokens
            The original tokens that appear in the unmasked input sequence.
            Shape (batch_size, num_masked_positions).
        masked_positions :
            The masked position of the sequence.
            Shape (batch_size, num_masked_positions).

        Returns
        -------
        mlm_scores
            The masked language model score.
            Shape (batch_size, num_masked_positions, vocab_size)
        rtd_scores
            The replaced-token-detection score. Predicts whether the tokens are replaced or not.
            - layout = 'NT'
                Shape (batch_size, seq_length)
            - layout = 'TN'
                Shape (seq_length, batch_size)
        replaced_inputs

            Shape (batch_size, num_masked_positions)
        labels
            - layout = 'NT'
                Shape (batch_size, seq_length)
            - layout = 'TN'
                Shape (seq_length, batch_size)
        """
        if self._uniform_generator:
            # generate the corrupt tokens randomly with a mlm_scores vector whose value is all 0
            zero_logits = F.np.zeros((1, 1, self.vocab_size), dtype=self._dtype)
            mlm_scores = F.np.expand_dims(F.np.zeros_like(masked_positions, dtype=self._dtype),
                                          axis=-1)
            mlm_scores = mlm_scores + zero_logits
        else:
            _, _, mlm_scores = self.generator(inputs, token_types, valid_length, masked_positions)

        corrupted_tokens, fake_data, labels = self.get_corrupted_tokens(
            F, inputs, original_tokens, masked_positions, mlm_scores)
        # The discriminator takes the same input as the generator and the token_ids are
        # replaced with fake data
        _, _, rtd_scores = self.discriminator(fake_data, token_types, valid_length)
        return mlm_scores, rtd_scores, corrupted_tokens, labels

    def get_corrupted_tokens(self, F, inputs, original_tokens, masked_positions, logits):
        """
        Sample from the generator to create corrupted input.

        Parameters
        ----------
        F
        inputs
            The masked input
            - layout = 'NT'
                Shape (batch_size, seq_length)
            - layout = 'TN'
                Shape (seq_length, batch_size)
        original_tokens
            The original tokens that appear in the unmasked input sequence
            Shape (batch_size, num_masked_positions).
        masked_positions
            The masked position of the sequence
            Shape (batch_size, num_masked_positions).
        logits
            The logits of each tokens
            Shape (batch_size, num_masked_positions, vocab_size)

        Returns
        -------
        corrupted_tokens
            Shape (batch_size, )
        fake_data
            - layout = 'NT'
                Shape (batch_size, seq_length)
            - layout = 'TN'
                Shape (seq_length, batch_size)
        labels
            - layout = 'NT'
                Shape (batch_size, seq_length)
            - layout = 'TN'
                Shape (seq_length, batch_size)
        """

        if self._disallow_correct:
            # TODO(sxjscience), Revise the implementation
            disallow = F.npx.one_hot(masked_positions, depth=self.vocab_size, dtype=self._dtype)
            logits = logits - 1000.0 * disallow
        # gumbel_softmax() samples from the logits with a noise of Gumbel distribution
        prob = gumbel_softmax(
            F,
            logits,
            temperature=self._temperature,
            eps=self._gumbel_eps,
            use_np_gumbel=False)
        corrupted_tokens = F.np.argmax(prob, axis=-1).astype(np.int32)

        if self.disc_backbone.layout == 'TN':
            inputs = inputs.T
        original_data = update_vectors_by_position(F,
            inputs, original_tokens, masked_positions)
        fake_data = update_vectors_by_position(F,
            inputs, corrupted_tokens, masked_positions)
        updates_mask = add_vectors_by_position(F, F.np.zeros_like(inputs),
                F.np.ones_like(masked_positions), masked_positions)
        # Dealing with multiple zeros in masked_positions which
        # results in a non-zero value in the first index [CLS]
        updates_mask = F.np.minimum(updates_mask, 1)
        labels = updates_mask * F.np.not_equal(fake_data, original_data)
        if self.disc_backbone.layout == 'TN':
            return corrupted_tokens, fake_data.T, labels.T
        else:
            return corrupted_tokens, fake_data, labels


def list_pretrained_electra():
    return sorted(list(PRETRAINED_URL.keys()))


def get_pretrained_electra(model_name: str = 'google_electra_small',
                           root: str = get_model_zoo_home_dir(),
                           load_backbone: bool = True,
                           load_disc: bool = False,
                           load_gen: bool = False) \
        -> Tuple[CN, HuggingFaceWordPieceTokenizer,
                 Optional[str],
                 Tuple[Optional[str], Optional[str]]]:
    """Get the pretrained Electra weights

    Parameters
    ----------
    model_name
        The name of the Electra model.
    root
        The downloading root
    load_backbone
        Whether to load the weights of the backbone network
    load_disc
        Whether to load the weights of the discriminator
    load_gen
        Whether to load the weights of the generator

    Returns
    -------
    cfg
        Network configuration
    tokenizer
        The HuggingFaceWordPieceTokenizer
    backbone_params_path
        Path to the parameter of the backbone network
    other_net_params_paths
        Path to the parameter of the discriminator and the generator.
        They will be returned inside a tuple.
    """
    assert model_name in PRETRAINED_URL, '{} is not found. All available are {}'.format(
        model_name, list_pretrained_electra())
    cfg_path = PRETRAINED_URL[model_name]['cfg']
    if isinstance(cfg_path, CN):
        cfg = cfg_path
    else:
        cfg = None
    vocab_path = PRETRAINED_URL[model_name]['vocab']
    params_path = PRETRAINED_URL[model_name]['params']
    disc_params_path = PRETRAINED_URL[model_name]['disc_model']
    gen_params_path = PRETRAINED_URL[model_name]['gen_model']

    local_paths = dict()
    download_jobs = [('vocab', vocab_path)]
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
    if load_disc:
        local_disc_params_path = download(url=get_repo_model_zoo_url() + disc_params_path,
                                          path=os.path.join(root, disc_params_path),
                                          sha1_hash=FILE_STATS[disc_params_path])
    else:
        local_disc_params_path = None

    if load_gen:
        local_gen_params_path = download(url=get_repo_model_zoo_url() + gen_params_path,
                                         path=os.path.join(root, gen_params_path),
                                         sha1_hash=FILE_STATS[gen_params_path])
    else:
        local_gen_params_path = None

    do_lower = True if 'lowercase' in PRETRAINED_URL[model_name]\
                       and PRETRAINED_URL[model_name]['lowercase'] else False
    tokenizer = HuggingFaceWordPieceTokenizer(
        vocab_file=local_paths['vocab'],
        unk_token='[UNK]',
        pad_token='[PAD]',
        cls_token='[CLS]',
        sep_token='[SEP]',
        mask_token='[MASK]',
        lowercase=do_lower)
    if cfg is None:
        cfg = ElectraModel.get_cfg().clone_merge(local_paths['cfg'])
    return cfg, tokenizer, local_params_path, (local_disc_params_path, local_gen_params_path)


BACKBONE_REGISTRY.register('electra', [ElectraModel,
                                       get_pretrained_electra,
                                       list_pretrained_electra])
