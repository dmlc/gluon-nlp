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
Bert Model

@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
"""

__all__ = ['BertModel', 'BertForMLM', 'BertForPretrain',
           'list_pretrained_bert', 'get_pretrained_bert']

import os
from typing import Tuple

import mxnet as mx
from mxnet import use_np
from mxnet.gluon import HybridBlock, nn
from ..registry import BACKBONE_REGISTRY
from .transformer import TransformerEncoderLayer
from ..base import get_model_zoo_home_dir, get_repo_model_zoo_url, get_model_zoo_checksum_dir
from ..utils.config import CfgNode as CN
from ..utils.misc import load_checksum_stats, download
from ..utils.registry import Registry
from ..initializer import TruncNorm
from ..attention_cell import MultiHeadAttentionCell, gen_self_attn_mask
from ..layers import get_activation, PositionalEmbedding, PositionwiseFFN, InitializerType
from ..op import select_vectors_by_position
from ..data.tokenizers import HuggingFaceWordPieceTokenizer

bert_cfg_reg = Registry('bert_cfg')


@bert_cfg_reg.register()
def google_en_uncased_bert_base():
    cfg = CN()
    # Parameters for thr small model
    cfg.MODEL = CN()
    cfg.MODEL.vocab_size = 30522
    cfg.MODEL.units = 768
    cfg.MODEL.hidden_size = 3072
    cfg.MODEL.max_length = 512
    cfg.MODEL.num_heads = 12
    cfg.MODEL.num_layers = 12
    cfg.MODEL.pos_embed_type = 'learned'
    cfg.MODEL.activation = 'gelu'
    cfg.MODEL.layer_norm_eps = 1E-12
    cfg.MODEL.num_token_types = 2
    cfg.MODEL.hidden_dropout_prob = 0.1
    cfg.MODEL.attention_dropout_prob = 0.1
    cfg.MODEL.dtype = 'float32'
    cfg.MODEL.layout = 'NT'
    cfg.MODEL.compute_layout = 'auto'
    # Hyper-parameters of the Initializers
    cfg.INITIALIZER = CN()
    cfg.INITIALIZER.embed = ['truncnorm', 0, 0.02]
    cfg.INITIALIZER.weight = ['truncnorm', 0, 0.02]  # TruncNorm(0, 0.02)
    cfg.INITIALIZER.bias = ['zeros']
    # Version of the model. This helps ensure backward compatibility.
    # Also, we can not use string here due to https://github.com/rbgirshick/yacs/issues/26
    cfg.VERSION = 1
    cfg.freeze()
    return cfg


@bert_cfg_reg.register()
def google_en_uncased_bert_large():
    cfg = google_en_uncased_bert_base()
    cfg.defrost()
    cfg.MODEL.hidden_size = 4096
    cfg.MODEL.num_heads = 16
    cfg.MODEL.num_layers = 24
    cfg.MODEL.units = 1024
    cfg.freeze()
    return cfg


@bert_cfg_reg.register()
def google_en_cased_bert_base():
    cfg = google_en_uncased_bert_base()
    cfg.defrost()
    cfg.MODEL.vocab_size = 28996
    cfg.freeze()
    return cfg


@bert_cfg_reg.register()
def google_en_cased_bert_large():
    cfg = google_en_uncased_bert_large()
    cfg.defrost()
    cfg.MODEL.vocab_size = 28996
    cfg.freeze()
    return cfg


@bert_cfg_reg.register()
def google_zh_bert_base():
    cfg = google_en_uncased_bert_base()
    cfg.defrost()
    cfg.MODEL.vocab_size = 21128
    cfg.freeze()
    return cfg


@bert_cfg_reg.register()
def google_multi_cased_bert_base():
    cfg = google_en_uncased_bert_base()
    cfg.defrost()
    cfg.MODEL.vocab_size = 119547
    cfg.freeze()
    return cfg


@bert_cfg_reg.register()
def google_multi_cased_bert_large():
    cfg = google_en_uncased_bert_large()
    cfg.defrost()
    cfg.MODEL.vocab_size = 119547
    cfg.freeze()
    return cfg


PRETRAINED_URL = {
    'google_en_cased_bert_base': {
        'cfg': google_en_cased_bert_base(),
        'vocab': 'google_en_cased_bert_base/vocab-c1defaaa.json',
        'params': 'google_en_cased_bert_base/model-c566c289.params',
        'mlm_params': 'google_en_cased_bert_base/model_mlm-bde14bee.params',
        'lowercase': False,
    },

    'google_en_uncased_bert_base': {
        'cfg': google_en_uncased_bert_base(),
        'vocab': 'google_en_uncased_bert_base/vocab-e6d2b21d.json',
        'params': 'google_en_uncased_bert_base/model-3712e50a.params',
        'mlm_params': 'google_en_uncased_bert_base/model_mlm-04e88b58.params',
        'lowercase': True,
    },
    'google_en_cased_bert_large': {
        'cfg': google_en_cased_bert_large(),
        'vocab': 'google_en_cased_bert_large/vocab-c1defaaa.json',
        'params': 'google_en_cased_bert_large/model-7aa93704.params',
        'mlm_params': 'google_en_cased_bert_large/model_mlm-59ff3f6a.params',
        'lowercase': False,
    },
    'google_en_uncased_bert_large': {
        'cfg': google_en_uncased_bert_large(),
        'vocab': 'google_en_uncased_bert_large/vocab-e6d2b21d.json',
        'params': 'google_en_uncased_bert_large/model-e53bbc57.params',
        'mlm_params': 'google_en_uncased_bert_large/model_mlm-44bc70c0.params',
        'lowercase': True,
    },
    'google_zh_bert_base': {
        'cfg': google_zh_bert_base(),
        'vocab': 'google_zh_bert_base/vocab-711c13e4.json',
        'params': 'google_zh_bert_base/model-2efbff63.params',
        'mlm_params': 'google_zh_bert_base/model_mlm-75339658.params',
        'lowercase': False,
    },
    'google_multi_cased_bert_base': {
        'cfg': google_multi_cased_bert_base(),
        'vocab': 'google_multi_cased_bert_base/vocab-016e1169.json',
        'params': 'google_multi_cased_bert_base/model-c2110078.params',
        'mlm_params': 'google_multi_cased_bert_base/model_mlm-4611e7a3.params',
        'lowercase': False,
    },
    'google_en_cased_bert_wwm_large': {
        'cfg': google_en_cased_bert_large(),
        'vocab': 'google_en_cased_bert_wwm_large/vocab-c1defaaa.json',
        'params': 'google_en_cased_bert_wwm_large/model-0fe841cf.params',
        'mlm_params': None,
        'lowercase': False,
    },
    'google_en_uncased_bert_wwm_large': {
        'cfg': google_en_uncased_bert_large(),
        'vocab': 'google_en_uncased_bert_wwm_large/vocab-e6d2b21d.json',
        'params': 'google_en_uncased_bert_wwm_large/model-cb3ad3c2.params',
        'mlm_params': None,
        'lowercase': True,
    }
}


FILE_STATS = load_checksum_stats(os.path.join(get_model_zoo_checksum_dir(), 'bert.txt'))


@use_np
class BertTransformer(HybridBlock):
    def __init__(self, units: int = 512,
                 hidden_size: int = 2048,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 attention_dropout_prob: float = 0.,
                 hidden_dropout_prob: float = 0.,
                 output_attention: bool = False,
                 dtype='float32',
                 output_all_encodings: bool = False,
                 layer_norm_eps: float = 1E-12,
                 weight_initializer: InitializerType = TruncNorm(stdev=0.02),
                 bias_initializer: InitializerType = 'zeros',
                 activation='gelu',
                 layout='NT'):
        super().__init__()
        assert units % num_heads == 0,\
            'In BertTransformer, The units should be divided exactly ' \
            'by the number of heads. Received units={}, num_heads={}' \
            .format(units, num_heads)

        self._dtype = dtype
        self._num_layers = num_layers
        self._output_attention = output_attention
        self._output_all_encodings = output_all_encodings
        self._layout = layout

        self.all_layers = nn.HybridSequential()
        for layer_idx in range(num_layers):
            self.all_layers.add(
              TransformerEncoderLayer(units=units,
                                      hidden_size=hidden_size,
                                      num_heads=num_heads,
                                      attention_dropout_prob=attention_dropout_prob,
                                      hidden_dropout_prob=hidden_dropout_prob,
                                      layer_norm_eps=layer_norm_eps,
                                      weight_initializer=weight_initializer,
                                      bias_initializer=bias_initializer,
                                      activation=activation,
                                      layout=layout,
                                      dtype=dtype))

    @property
    def layout(self):
        return self._layout

    def hybrid_forward(self, F, data, valid_length):
        """
        Generate the representation given the inputs.

        This is used in training or fine-tuning a bert model.

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
        attn_mask = gen_self_attn_mask(F, data, valid_length, dtype=self._dtype,
                                       attn_type='full', layout=self.layout)
        out = data
        all_encodings_outputs = []
        additional_outputs = []
        for layer_idx in range(self._num_layers):
            layer = self.all_layers[layer_idx]
            out, attention_weights = layer(out, attn_mask)
            # out : [batch_size, seq_len, units] or [seq_len, batch_size, units]
            # attention_weights : [batch_size, num_heads, seq_len, seq_len]
            if self._output_all_encodings:
                out = F.npx.sequence_mask(out,
                                          sequence_length=valid_length,
                                          use_sequence_length=True, axis=time_axis)
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
class BertModel(HybridBlock):
    def __init__(self,
                 vocab_size=30000,
                 units=768,
                 hidden_size=3072,
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
        # Construct BertTransformer
        self.encoder = BertTransformer(
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
            layout=self._compute_layout
        )
        self.encoder.hybridize()
        # Construct word embedding
        self.word_embed = nn.Embedding(input_dim=vocab_size,
                                       output_dim=units,
                                       weight_initializer=embed_initializer,
                                       dtype=dtype)
        self.embed_layer_norm = nn.LayerNorm(epsilon=self.layer_norm_eps)
        self.embed_dropout = nn.Dropout(hidden_dropout_prob)
        # Construct token type embedding
        self.token_type_embed = nn.Embedding(input_dim=num_token_types,
                                             output_dim=units,
                                             weight_initializer=weight_initializer)
        self.token_pos_embed = PositionalEmbedding(units=units,
                                                   max_length=max_length,
                                                   dtype=self._dtype,
                                                   method=pos_embed_type)
        if self.use_pooler:
            # Construct pooler
            self.pooler = nn.Dense(units=units,
                                   in_units=units,
                                   flatten=False,
                                   activation='tanh',
                                   weight_initializer=weight_initializer,
                                   bias_initializer=bias_initializer)

    @property
    def layout(self):
        return self._layout

    def hybrid_forward(self, F, inputs, token_types, valid_length):
        # pylint: disable=arguments-differ
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a bert model.

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
                Shape (batch_size, seq_length)

            If the inputs contain two sequences, we will set different token types for the first
             sentence and the second sentence.
        valid_length :
            The valid length of each sequence
            Shape (batch_size,)

        Returns
        -------
        contextual_embedding
            - layout = 'NT'
                Shape (batch_size, seq_length, units).
            - layout = 'TN'
                Shape (seq_length, batch_size, units).
        pooled_output :
            This is optional. Shape (batch_size, units)
        """
        initial_embedding = self.get_initial_embedding(F, inputs, token_types)
        prev_out = initial_embedding
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
            pooled_out = self.apply_pooling(contextual_embeddings)
            outputs.append(pooled_out)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

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
                Shape (batch_size, seq_length, C_emb)
            - layout = 'TN'
                Shape (seq_length, batch_size, C_emb)
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

    def apply_pooling(self, sequence):
        """Generate the representation given the inputs.

        This is used for pre-training or fine-tuning a bert model.
        Get the first token of the whole sequence which is [CLS]

        sequence
            - layout = 'NT'
                Shape (batch_size, sequence_length, units)
            - layout = 'TN'
                Shape (sequence_length, batch_size, units)
        return:
            Shape (batch_size, units)
        """
        if self.layout == 'NT':
            outputs = sequence[:, 0, :]
        else:
            outputs = sequence[0, :, :]
        return self.pooler(outputs)

    @staticmethod
    def get_cfg(key=None):
        if key is not None:
            return bert_cfg_reg.create(key)
        else:
            return google_en_uncased_bert_base()

    @classmethod
    def from_cfg(cls, cfg, use_pooler=True, dtype=None) -> 'BertModel':
        """

        Parameters
        ----------
        cfg
            Configuration
        use_pooler
            Whether to output the pooled feature
        dtype
            data type of the model

        Returns
        -------
        ret
            The constructed BertModel
        """
        cfg = BertModel.get_cfg().clone_merge(cfg)
        assert cfg.VERSION == 1, 'Wrong version!'
        embed_initializer = mx.init.create(*cfg.INITIALIZER.embed)
        weight_initializer = mx.init.create(*cfg.INITIALIZER.weight)
        bias_initializer = mx.init.create(*cfg.INITIALIZER.bias)
        if dtype is None:
            dtype = cfg.MODEL.dtype
        return cls(vocab_size=cfg.MODEL.vocab_size,
                   units=cfg.MODEL.units,
                   hidden_size=cfg.MODEL.hidden_size,
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
class BertForMLM(HybridBlock):
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
        self.backbone_model = BertModel.from_cfg(backbone_cfg)
        if weight_initializer is None:
            weight_initializer = self.backbone_model.weight_initializer
        if bias_initializer is None:
            bias_initializer = self.backbone_model.bias_initializer
        self.mlm_decoder = nn.HybridSequential()
        # Extra non-linear layer
        self.mlm_decoder.add(nn.Dense(units=self.backbone_model.units,
                                      flatten=False,
                                      weight_initializer=weight_initializer,
                                      bias_initializer=bias_initializer))
        self.mlm_decoder.add(get_activation(self.backbone_model.activation))
        self.mlm_decoder.add(nn.LayerNorm(epsilon=self.backbone_model.layer_norm_eps))
        # only load the dense weights with a re-initialized bias
        # parameters are stored in 'word_embed_bias' which is
        # not used in original embedding
        self.mlm_decoder.add(nn.Dense(units=self.backbone_model.vocab_size,
                                      flatten=False,
                                      bias_initializer=bias_initializer))
        self.mlm_decoder[-1].weight = self.backbone_model.word_embed.weight
        self.mlm_decoder.hybridize()

    @property
    def layout(self):
        return self.backbone_model.layout

    def hybrid_forward(self, F, inputs, token_types, valid_length,
                       masked_positions):
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
                Shape (seq_length, batch_size, units)
        pooled_out
            Shape (batch_size, units)
        mlm_scores :
            Shape (batch_size, num_masked_positions, vocab_size)
        """
        contextual_embeddings, pooled_out = self.backbone_model(inputs, token_types, valid_length)
        if self.layout == 'NT':
            mlm_features = select_vectors_by_position(F, contextual_embeddings, masked_positions)
        else:
            mlm_features = select_vectors_by_position(F, F.np.swapaxes(contextual_embeddings, 0, 1),
                                                      masked_positions)
        mlm_scores = self.mlm_decoder(mlm_features)
        return contextual_embeddings, pooled_out, mlm_scores


@use_np
class BertForPretrain(HybridBlock):
    def __init__(self, backbone_cfg,
                 weight_initializer=None,
                 bias_initializer=None):
        """

        Parameters
        ----------
        backbone_cfg
            The cfg of the backbone model
        weight_initializer
        bias_initializer
        """
        super().__init__()
        self.backbone_model = BertModel.from_cfg(backbone_cfg)
        if weight_initializer is None:
            weight_initializer = self.backbone_model.weight_initializer
        if bias_initializer is None:
            bias_initializer = self.backbone_model.bias_initializer
        # Construct nsp_classifier for next sentence prediction
        self.nsp_classifier = nn.Dense(units=2,
                                       weight_initializer=weight_initializer)
        self.mlm_decoder = nn.HybridSequential()
        # Extra non-linear layer
        self.mlm_decoder.add(nn.Dense(units=self.backbone_model.units,
                                      flatten=False,
                                      weight_initializer=weight_initializer,
                                      bias_initializer=bias_initializer))
        self.mlm_decoder.add(get_activation(self.backbone_model.activation))
        self.mlm_decoder.add(nn.LayerNorm(epsilon=self.backbone_model.layer_norm_eps))
        # only load the dense weights with a re-initialized bias
        # parameters are stored in 'word_embed_bias' which is
        # not used in original embedding
        self.mlm_decoder.add(nn.Dense(units=self.backbone_model.vocab_size,
                                      flatten=False,
                                      bias_initializer=bias_initializer))
        self.mlm_decoder[-1].weight = self.backbone_model.word_embed.weight
        self.mlm_decoder.hybridize()

    @property
    def layout(self):
        return self.backbone_model.layout

    def hybrid_forward(self, F, inputs, token_types, valid_length,
                       masked_positions):
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a bert model.

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
        masked_positions
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
        nsp_score :
            Shape (batch_size, 2)
        mlm_scores :
            Shape (batch_size, num_masked_positions, vocab_size)
        """
        contextual_embeddings, pooled_out = self.backbone_model(inputs, token_types, valid_length)
        nsp_score = self.nsp_classifier(pooled_out)
        if self.layout == 'NT':
            mlm_features = select_vectors_by_position(F, contextual_embeddings, masked_positions)
        else:
            mlm_features = select_vectors_by_position(F, F.np.swapaxes(contextual_embeddings, 0, 1),
                                                      masked_positions)
        mlm_scores = self.mlm_decoder(mlm_features)
        return contextual_embeddings, pooled_out, nsp_score, mlm_scores


def list_pretrained_bert():
    return sorted(list(PRETRAINED_URL.keys()))


def get_pretrained_bert(model_name: str = 'google_en_cased_bert_base',
                        root: str = get_model_zoo_home_dir(),
                        load_backbone: str = True,
                        load_mlm: str = False)\
        -> Tuple[CN, HuggingFaceWordPieceTokenizer, str, str]:
    """Get the pretrained bert weights

    Parameters
    ----------
    model_name
        The name of the bert model.
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
        The HuggingFaceWordPieceTokenizer
    backbone_params_path
        Path to the parameter of the backbone network
    mlm_params_path
        Path to the parameter that includes both the backbone and the MLM
    """
    assert model_name in PRETRAINED_URL, '{} is not found. All available are {}'.format(
        model_name, list_pretrained_bert())
    cfg_path = PRETRAINED_URL[model_name]['cfg']
    if isinstance(cfg_path, CN):
        cfg = cfg_path
    else:
        cfg = None
    vocab_path = PRETRAINED_URL[model_name]['vocab']
    params_path = PRETRAINED_URL[model_name]['params']
    mlm_params_path = PRETRAINED_URL[model_name]['mlm_params']
    local_paths = dict()
    download_jobs = [('vocab', vocab_path)]
    if cfg is None:
        download_jobs.append(('cfg', cfg_path))
    for key, path in download_jobs:
        local_paths[key] = download(url=get_repo_model_zoo_url() + path,
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
    tokenizer = HuggingFaceWordPieceTokenizer(
                    vocab_file=local_paths['vocab'],
                    unk_token='[UNK]',
                    pad_token='[PAD]',
                    cls_token='[CLS]',
                    sep_token='[SEP]',
                    mask_token='[MASK]',
                    lowercase=do_lower)
    if cfg is None:
        cfg = BertModel.get_cfg().clone_merge(local_paths['cfg'])
    return cfg, tokenizer, local_params_path, local_mlm_params_path


BACKBONE_REGISTRY.register('bert', [BertModel,
                                    get_pretrained_bert,
                                    list_pretrained_bert])
