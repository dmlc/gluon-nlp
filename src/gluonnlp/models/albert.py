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
"""Albert Model.

@inproceedings{lan2020albert,
  title={Albert: A lite bert for self-supervised learning of language representations},
  author={Lan, Zhenzhong and Chen, Mingda and Goodman, Sebastian and Gimpel, Kevin and Sharma, Piyush and Soricut, Radu},
  booktitle={ICLR},
  year={2020}
}

"""
__all__ = ['AlbertModel', 'AlbertForMLM', 'AlbertForPretrain',
           'list_pretrained_albert', 'get_pretrained_albert',
           'albert_cfg_reg']

import os
from typing import Tuple

import mxnet as mx
from mxnet import use_np
from mxnet.gluon import HybridBlock, nn
from .transformer import TransformerEncoderLayer
from ..registry import BACKBONE_REGISTRY
from ..base import get_model_zoo_home_dir, get_repo_model_zoo_url, get_model_zoo_checksum_dir
from ..utils.config import CfgNode as CN
from ..utils.misc import load_checksum_stats, download
from ..utils.registry import Registry
from ..initializer import TruncNorm
from ..attention_cell import gen_self_attn_mask
from ..layers import get_activation, PositionalEmbedding
from ..op import select_vectors_by_position
from ..data.tokenizers import SentencepieceTokenizer

albert_cfg_reg = Registry('albert_cfg')


@albert_cfg_reg.register()
def google_albert_base():
    cfg = CN()
    # Model Parameters
    cfg.MODEL = CN()
    cfg.MODEL.vocab_size = 30000
    cfg.MODEL.embed_size = 128
    cfg.MODEL.units = 768
    cfg.MODEL.hidden_size = 3072
    cfg.MODEL.max_length = 512
    cfg.MODEL.num_heads = 12
    cfg.MODEL.num_layers = 12
    cfg.MODEL.pos_embed_type = 'learned'
    cfg.MODEL.activation = 'gelu(tanh)'
    cfg.MODEL.layer_norm_eps = 1E-12
    cfg.MODEL.num_groups = 1
    cfg.MODEL.num_token_types = 2
    cfg.MODEL.hidden_dropout_prob = 0.0
    cfg.MODEL.attention_dropout_prob = 0.0
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


@albert_cfg_reg.register()
def google_albert_large():
    cfg = google_albert_base()
    cfg.defrost()
    cfg.MODEL.hidden_size = 4096
    cfg.MODEL.num_heads = 16
    cfg.MODEL.num_layers = 24
    cfg.MODEL.units = 1024
    cfg.freeze()
    return cfg


@albert_cfg_reg.register()
def google_albert_xlarge():
    cfg = google_albert_base()
    cfg.defrost()
    cfg.MODEL.hidden_size = 8192
    cfg.MODEL.num_heads = 32
    cfg.MODEL.num_layers = 24
    cfg.MODEL.units = 2048
    cfg.freeze()
    return cfg


@albert_cfg_reg.register()
def google_albert_xxlarge():
    cfg = google_albert_base()
    cfg.defrost()
    cfg.MODEL.hidden_size = 16384
    cfg.MODEL.num_heads = 64
    cfg.MODEL.num_layers = 12
    cfg.MODEL.units = 4096
    cfg.freeze()
    return cfg


PRETRAINED_URL = {
    'google_albert_base_v2': {
        'cfg': google_albert_base(),
        'spm_model': 'google_albert_base_v2/spm-65999e5d.model',
        'vocab': 'google_albert_base_v2/vocab-2ee53ae7.json',
        'params': 'google_albert_base_v2/model-125be477.params',
        'mlm_params': 'google_albert_base_v2/model_mlm-fe20650e.params',
        'lowercase': True,
    },
    'google_albert_large_v2': {
        'cfg': google_albert_large(),
        'spm_model': 'google_albert_large_v2/spm-65999e5d.model',
        'vocab': 'google_albert_large_v2/vocab-2ee53ae7.json',
        'params': 'google_albert_large_v2/model-ad60bcd5.params',
        'mlm_params': 'google_albert_large_v2/model_mlm-6a5015ee.params',
        'lowercase': True,
    },
    'google_albert_xlarge_v2': {
        'cfg': google_albert_xlarge(),
        'spm_model': 'google_albert_xlarge_v2/spm-65999e5d.model',
        'vocab': 'google_albert_xlarge_v2/vocab-2ee53ae7.json',
        'params': 'google_albert_xlarge_v2/model-4149c9e2.params',
        'mlm_params': 'google_albert_xlarge_v2/model_mlm-ee184d38.params',
        'lowercase': True,
    },
    'google_albert_xxlarge_v2': {
        'cfg': google_albert_xxlarge(),
        'spm_model': 'google_albert_xxlarge_v2/spm-65999e5d.model',
        'vocab': 'google_albert_xxlarge_v2/vocab-2ee53ae7.json',
        'params': 'google_albert_xxlarge_v2/model-5601a0ed.params',
        'mlm_params': 'google_albert_xxlarge_v2/model_mlm-d2e2b06f.params',
        'lowercase': True,
    },
}

FILE_STATS = load_checksum_stats(os.path.join(get_model_zoo_checksum_dir(), 'albert.txt'))


@use_np
class AlbertEncoder(HybridBlock):
    def __init__(self, units=512, hidden_size=2048,
                 num_layers=6,
                 num_groups=1,
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
        super().__init__()
        assert units % num_heads == 0,\
            'In AlbertEncoder, The units should be divided exactly ' \
            'by the number of heads. Received units={}, num_heads={}' \
            .format(units, num_heads)

        self._dtype = dtype
        self._num_layers = num_layers
        self._num_groups = num_groups
        assert num_layers % num_groups == 0
        self._num_layers_each_group = num_layers // num_groups

        self._output_attention = output_attention
        self._output_all_encodings = output_all_encodings
        self._layout = layout


        self.all_encoder_groups = nn.HybridSequential()
        for group_idx in range(num_groups):
            self.all_encoder_groups.add(
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

        This is used in training or fine-tuning a Bert model.

        Parameters
        ----------
        F
        data
            - layout = 'NT'
                Shape (batch_size, seq_length, C)
            - layout = 'TN'
                Shape (seq_length, batch_size, C)
        valid_length :
            Shape (batch_size,)

        Returns
        -------
        out
            - layout = 'NT'
                Shape (batch_size, seq_length, C_out)
            - layout = 'TN'
                Shape (seq_length, batch_size, C)
        """
        # 1. Embed the data
        time_axis = 1 if self.layout == 'NT' else 0
        attn_mask = gen_self_attn_mask(F, data, valid_length, dtype=self._dtype,
                                       attn_type='full', layout=self.layout)
        out = data
        all_encodings_outputs = []
        additional_outputs = []
        for layer_idx in range(self._num_layers):
            groups_id = layer_idx // self._num_layers_each_group
            layer = self.all_encoder_groups[groups_id]
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
                                      use_sequence_length=True,
                                      axis=time_axis)
            return out, additional_outputs
        else:
            return all_encodings_outputs, additional_outputs


@use_np
class AlbertModel(HybridBlock):
    def __init__(self,
                 vocab_size=30000,
                 units=768,
                 hidden_size=3072,
                 embed_size=128,
                 num_layers=12,
                 num_heads=12,
                 num_groups=1,
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
        # Construct AlbertEncoder
        self.encoder = AlbertEncoder(
            units=units,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_groups=num_groups,
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
                                       output_dim=embed_size,
                                       weight_initializer=embed_initializer,
                                       dtype=dtype)
        if embed_size != units:
            self.embed_factorized_proj = nn.Dense(units=units,
                                                  flatten=False,
                                                  weight_initializer=weight_initializer,
                                                  bias_initializer=bias_initializer)
        self.embed_layer_norm = nn.LayerNorm(epsilon=self.layer_norm_eps)
        self.embed_dropout = nn.Dropout(hidden_dropout_prob)
        # Construct token type embedding
        self.token_type_embed = nn.Embedding(input_dim=num_token_types,
                                             output_dim=embed_size,
                                             weight_initializer=weight_initializer)
        self.token_pos_embed = PositionalEmbedding(units=embed_size,
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

    def hybrid_forward(self, F, inputs, token_types, valid_length=None):
        # pylint: disable=arguments-differ
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a Albert model.

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

        Returns
        -------
        contextual_embedding
            - layout = 'NT'
                Shape (batch_size, seq_length, units)
            - layout = 'TN'
                Shape (seq_length, batch_size, units)
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
            # Swap input to reflect the compute_layout
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
            batch_axis, time_axis = 0, 1
        else:
            batch_axis, time_axis = 1, 0
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

        This is used for pre-training or fine-tuning a Bert model.
        Get the first token of the whole sequence which is [CLS]

        Parameters
        ----------
        sequence
            - layout = 'NT'
                Shape (batch_size, sequence_length, units)
            - layout = 'TN'
                Shape (sequence_length, batch_size, units)

        Returns
        -------
        pooled_out
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
            return albert_cfg_reg.create(key)
        else:
            return google_albert_base()

    @classmethod
    def from_cfg(cls, cfg, use_pooler=True, dtype=None) -> 'AlbertModel':
        """

        Parameters
        ----------
        cfg
        use_pooler
            Whether to use pooler
        dtype
            The dtype of the backbone model

        Returns
        -------
        model
            The created AlbertModel
        """
        cfg = cls.get_cfg().clone_merge(cfg)
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
                   num_groups=cfg.MODEL.num_groups,
                   max_length=cfg.MODEL.max_length,
                   hidden_dropout_prob=cfg.MODEL.hidden_dropout_prob,
                   attention_dropout_prob=cfg.MODEL.attention_dropout_prob,
                   num_token_types=cfg.MODEL.num_token_types,
                   pos_embed_type=cfg.MODEL.pos_embed_type,
                   activation=cfg.MODEL.activation,
                   layer_norm_eps=cfg.MODEL.layer_norm_eps,
                   dtype=dtype,
                   layout=cfg.MODEL.layout,
                   embed_initializer=embed_initializer,
                   weight_initializer=weight_initializer,
                   bias_initializer=bias_initializer,
                   use_pooler=use_pooler)


@use_np
class AlbertForMLM(HybridBlock):
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
        self.backbone_model = AlbertModel.from_cfg(backbone_cfg)
        if weight_initializer is None:
            weight_initializer = self.backbone_model.weight_initializer
        if bias_initializer is None:
            bias_initializer = self.backbone_model.bias_initializer
        self.mlm_decoder = nn.HybridSequential()
        # Extra non-linear layer
        self.mlm_decoder.add(nn.Dense(units=self.backbone_model.embed_size,
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
            The type of the token. For example, if the inputs contain two sequences,
            we will set different token types for the first sentence and the second sentence.
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
                Shape (batch_size, seq_length, units)
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
class AlbertForPretrain(HybridBlock):
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
        self.backbone_model = AlbertModel.from_cfg(backbone_cfg)
        if weight_initializer is None:
            weight_initializer = self.backbone_model.weight_initializer
        if bias_initializer is None:
            bias_initializer = self.backbone_model.bias_initializer
        # Construct sop_classifier for sentence order prediction
        self.sop_classifier = nn.Dense(units=2,
                                       weight_initializer=weight_initializer)
        self.mlm_decoder = nn.HybridSequential()
        # Extra non-linear layer
        self.mlm_decoder.add(nn.Dense(units=self.backbone_model.embed_size,
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

        This is used in training or fine-tuning a Albert model.

        Parameters
        ----------
        F
        inputs
            - layout = 'NT'
                Shape (batch_size, seq_length)
            - layout = 'TN'
                Shape (seq_length, batch_size)
        token_types :
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
        sop_score :
            Shape (batch_size, 2)
        mlm_scores :
            Shape (batch_size, num_masked_positions, vocab_size)
        """
        contextual_embeddings, pooled_out = self.backbone_model(inputs, token_types, valid_length)
        sop_score = self.sop_classifier(pooled_out)
        if self.layout == 'NT':
            mlm_features = select_vectors_by_position(F, contextual_embeddings, masked_positions)
        else:
            mlm_features = select_vectors_by_position(F, F.np.swapaxes(contextual_embeddings, 0, 1),
                                                      masked_positions)
        mlm_scores = self.mlm_decoder(mlm_features)
        return contextual_embeddings, pooled_out, sop_score, mlm_scores


def list_pretrained_albert():
    return sorted(list(PRETRAINED_URL.keys()))


def get_pretrained_albert(model_name: str = 'google_albert_base_v2',
                          root: str = get_model_zoo_home_dir(),
                          load_backbone: str = True,
                          load_mlm: str = False)\
        -> Tuple[CN, SentencepieceTokenizer, str, str]:
    """Get the pretrained Albert weights

    Parameters
    ----------
    model_name
        The name of the Albert model.
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
        The SentencepieceTokenizer
    backbone_params_path
        Path to the parameter of the backbone network
    mlm_params_path
        Path to the parameter that includes both the backbone and the MLM
    """
    assert model_name in PRETRAINED_URL, '{} is not found. All available are {}'.format(
        model_name, list_pretrained_albert())
    cfg_path = PRETRAINED_URL[model_name]['cfg']
    if isinstance(cfg_path, CN):
        cfg = cfg_path
    else:
        cfg = None
    spm_model_path = PRETRAINED_URL[model_name]['spm_model']
    vocab_path = PRETRAINED_URL[model_name]['vocab']
    params_path = PRETRAINED_URL[model_name]['params']
    mlm_params_path = PRETRAINED_URL[model_name]['mlm_params']
    local_paths = dict()
    download_jobs = [('spm_model', spm_model_path), ('vocab', vocab_path)]
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
    if load_mlm:
        local_mlm_params_path = download(url=get_repo_model_zoo_url() + mlm_params_path,
                                         path=os.path.join(root, mlm_params_path),
                                         sha1_hash=FILE_STATS[mlm_params_path])
    else:
        local_mlm_params_path = None
    do_lower = True if 'lowercase' in PRETRAINED_URL[model_name]\
                       and PRETRAINED_URL[model_name]['lowercase'] else False
    tokenizer = SentencepieceTokenizer(local_paths['spm_model'],
                                       vocab=local_paths['vocab'],
                                       lowercase=do_lower)
    if cfg is None:
        cfg = AlbertModel.get_cfg().clone_merge(local_paths['cfg'])
    return cfg, tokenizer, local_params_path, local_mlm_params_path


BACKBONE_REGISTRY.register('albert', [AlbertModel,
                                      get_pretrained_albert,
                                      list_pretrained_albert])
