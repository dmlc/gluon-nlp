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

__all__ = ['BertModel', 'BertForMLM', 'QTBertForPretrain', 'init_weights']

import logging

import torch as th

from ..attention_cell import gen_self_attn_mask
from .transformer import TransformerEncoderLayer
from ..layers import get_activation, sequence_mask
from ...utils.registry import Registry
from ...models.bert import bert_cfg_reg  # TODO CFG.INITIALIZER not supported and handled via def init_weights


class BertTransformer(th.nn.Module):
    def __init__(self, units: int = 512, hidden_size: int = 2048, num_layers: int = 6,
                 num_heads: int = 8, attention_dropout_prob: float = 0.,
                 hidden_dropout_prob: float = 0., output_attention: bool = False,
                 output_all_encodings: bool = False, layer_norm_eps: float = 1E-12,
                 activation='gelu', layout='NT'):
        super().__init__()
        assert units % num_heads == 0,\
            'In BertTransformer, The units should be divided exactly ' \
            'by the number of heads. Received units={}, num_heads={}' \
            .format(units, num_heads)

        self._num_layers = num_layers
        self._output_attention = output_attention
        self._output_all_encodings = output_all_encodings
        self._layout = layout

        self.all_layers = th.nn.ModuleList([
            TransformerEncoderLayer(units=units, hidden_size=hidden_size, num_heads=num_heads,
                                    attention_dropout_prob=attention_dropout_prob,
                                    hidden_dropout_prob=hidden_dropout_prob,
                                    layer_norm_eps=layer_norm_eps, activation=activation,
                                    layout=layout) for _ in range(num_layers)
        ])

    @property
    def layout(self):
        return self._layout

    def forward(self, data, valid_length):
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
        attn_mask = gen_self_attn_mask(data, valid_length, attn_type='full', layout=self.layout)
        out = data
        all_encodings_outputs = []
        additional_outputs = []
        for layer_idx in range(self._num_layers):
            layer = self.all_layers[layer_idx]
            out, attention_weights = layer(out, attn_mask)
            # out : [batch_size, seq_len, units] or [seq_len, batch_size, units]
            # attention_weights : [batch_size, num_heads, seq_len, seq_len]
            if self._output_all_encodings:
                out = sequence_mask(out, valid_len=valid_length, axis=time_axis)
                all_encodings_outputs.append(out)

            if self._output_attention:
                additional_outputs.append(attention_weights)

        if not self._output_all_encodings:
            # if self._output_all_encodings, SequenceMask is already applied above
            out = sequence_mask(out, valid_len=valid_length, axis=time_axis)
            return out, additional_outputs
        else:
            return all_encodings_outputs, additional_outputs


class BertModel(th.nn.Module):
    def __init__(self, vocab_size=30000, units=768, hidden_size=3072, num_layers=12, num_heads=12,
                 max_length=512, hidden_dropout_prob=0., attention_dropout_prob=0.,
                 num_token_types=2, pos_embed_type='learned', activation='gelu',
                 layer_norm_eps=1E-12, use_pooler=True, layout='NT', compute_layout='auto'):
        super().__init__()
        self.use_pooler = use_pooler
        self.pos_embed_type = pos_embed_type
        self.num_token_types = num_token_types
        self.vocab_size = vocab_size
        self.units = units
        self.max_length = max_length
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self._layout = layout
        if compute_layout is None or compute_layout == 'auto':
            self._compute_layout = layout
        else:
            self._compute_layout = compute_layout
        # Construct BertTransformer
        self.encoder = BertTransformer(
            units=units, hidden_size=hidden_size, num_layers=num_layers, num_heads=num_heads,
            attention_dropout_prob=attention_dropout_prob, hidden_dropout_prob=hidden_dropout_prob,
            output_attention=False, output_all_encodings=False, activation=activation,
            layer_norm_eps=layer_norm_eps, layout=self._compute_layout)
        # Construct word embedding
        self.word_embed = th.nn.Embedding(num_embeddings=vocab_size, embedding_dim=units)
        self.embed_layer_norm = th.nn.LayerNorm(units, eps=self.layer_norm_eps)
        self.embed_dropout = th.nn.Dropout(hidden_dropout_prob)
        # Construct token type embedding
        self.token_type_embed = th.nn.Embedding(num_embeddings=num_token_types, embedding_dim=units)
        assert pos_embed_type == 'learned'
        self.token_pos_embed = th.nn.Embedding(num_embeddings=max_length, embedding_dim=units)
        if self.use_pooler:
            # Construct pooler
            self.pooler = th.nn.Linear(out_features=units, in_features=units)

    @property
    def layout(self):
        return self._layout

    def forward(self, inputs, token_types, valid_length):
        # pylint: disable=arguments-differ
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a bert model.

        Parameters
        ----------
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
        if token_types is None:
            token_types = th.zeros_like(inputs)
        initial_embedding = self.get_initial_embedding(inputs, token_types)
        prev_out = initial_embedding
        outputs = []
        if self._compute_layout != self._layout:
            # Swap the axes if the compute_layout and layout mismatch
            contextual_embeddings, additional_outputs = self.encoder(th.transpose(prev_out, 0, 1),
                                                                     valid_length)
            contextual_embeddings = th.transpose(contextual_embeddings, 0, 1)
        else:
            contextual_embeddings, additional_outputs = self.encoder(prev_out, valid_length)
        outputs.append(contextual_embeddings)
        if self.use_pooler:
            pooled_out = self.apply_pooling(contextual_embeddings)
            outputs.append(pooled_out)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    def get_initial_embedding(self, inputs, token_types=None):
        """Get the initial token embeddings that considers the token type and positional embeddings

        Parameters
        ----------
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
            token_types = th.zeros_like(inputs)
        type_embedding = self.token_type_embed(token_types)
        embedding = embedding + type_embedding
        if self.pos_embed_type is not None:
            positional_embedding = self.token_pos_embed(
                th.arange(end=inputs.shape[time_axis], device=inputs.device))
            positional_embedding = th.unsqueeze(positional_embedding, dim=batch_axis)
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
        return th.tanh(self.pooler(outputs))

    @staticmethod
    def get_cfg(key=None):
        if key is not None:
            return bert_cfg_reg.create(key)
        else:
            return bert_cfg_reg.create('google_en_uncased_bert_base')

    @classmethod
    def from_cfg(cls, cfg, use_pooler=True) -> 'BertModel':
        """

        Parameters
        ----------
        cfg
            Configuration
        use_pooler
            Whether to output the pooled feature

        Returns
        -------
        ret
            The constructed BertModel
        """
        cfg = BertModel.get_cfg().clone_merge(cfg)
        assert cfg.VERSION == 1, 'Wrong version!'
        return cls(vocab_size=cfg.MODEL.vocab_size, units=cfg.MODEL.units,
                   hidden_size=cfg.MODEL.hidden_size, num_layers=cfg.MODEL.num_layers,
                   num_heads=cfg.MODEL.num_heads, max_length=cfg.MODEL.max_length,
                   hidden_dropout_prob=cfg.MODEL.hidden_dropout_prob,
                   attention_dropout_prob=cfg.MODEL.attention_dropout_prob,
                   num_token_types=cfg.MODEL.num_token_types,
                   pos_embed_type=cfg.MODEL.pos_embed_type, activation=cfg.MODEL.activation,
                   layer_norm_eps=cfg.MODEL.layer_norm_eps, use_pooler=use_pooler,
                   layout=cfg.MODEL.layout, compute_layout=cfg.MODEL.compute_layout)


class BertForMLM(th.nn.Module):
    def __init__(self, backbone_cfg):
        """

        Parameters
        ----------
        backbone_cfg
        """
        super().__init__()
        self.backbone_model = BertModel.from_cfg(backbone_cfg)
        self.mlm_decoder = th.nn.Sequential(
            th.nn.Linear(out_features=self.backbone_model.units,
                         in_features=self.backbone_model.units),
            get_activation(self.backbone_model.activation),
            th.nn.LayerNorm(self.backbone_model.units, eps=self.backbone_model.layer_norm_eps),
            th.nn.Linear(out_features=self.backbone_model.vocab_size,
                         in_features=self.backbone_model.units))
        # TODO such weight sharing not supported in torchscript
        self.mlm_decoder[-1].weight = self.backbone_model.word_embed.weight

    @property
    def layout(self):
        return self.backbone_model.layout

    def forward(self, inputs, token_types, valid_length, masked_positions):
        """Getting the scores of the masked positions.

        Parameters
        ----------
        inputs
            - layout = 'NT'
                Shape (batch_size, seq_length)
            - layout = 'TN'
                Shape (seq_length, batch_size)

        token_types
            If the inputs contain two sequences, we will set different token types for the first
             sentence and the second sentence.

            - layout = 'NT'
                Shape (batch_size, seq_length)
            - layout = 'TN'
                Shape (seq_length, batch_size)

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
            mlm_features = contextual_embeddings[
                th.arange(contextual_embeddings.shape[0]).unsqueeze(1), masked_positions]
        else:
            contextual_embeddings_t = th.transpose(contextual_embeddings, 0, 1)
            mlm_features = contextual_embeddings_t[
                th.arange(contextual_embeddings_t.shape[0]).unsqueeze(1), masked_positions]
        mlm_scores = self.mlm_decoder(mlm_features)
        return contextual_embeddings, pooled_out, mlm_scores


class BertForPretrain(th.nn.Module):
    def __init__(self, backbone_cfg):
        """

        Parameters
        ----------
        backbone_cfg
            The cfg of the backbone model
        """
        super().__init__()
        self.backbone_model = BertModel.from_cfg(backbone_cfg)
        # Construct nsp_classifier for next sentence prediction
        self.nsp_classifier = th.nn.Linear(out_features=2, in_features=self.backbone_model.units)
        self.mlm_decoder = th.nn.Sequential(
            th.nn.Linear(out_features=self.backbone_model.units,
                         in_features=self.backbone_model.units),
            get_activation(self.backbone_model.activation),
            th.nn.LayerNorm(self.backbone_model.units, eps=self.backbone_model.layer_norm_eps),
            th.nn.Linear(out_features=self.backbone_model.vocab_size,
                         in_features=self.backbone_model.units))
        # TODO such weight sharing not supported in torchscript
        self.mlm_decoder[-1].weight = self.backbone_model.word_embed.weight

    @property
    def layout(self):
        return self.backbone_model.layout

    def forward(self, inputs, token_types, valid_length, masked_positions):
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a bert model.

        Parameters
        ----------
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
            mlm_features = contextual_embeddings[
                th.arange(contextual_embeddings.shape[0]).unsqueeze(1), masked_positions]
        else:
            mlm_features = th.transpose(contextual_embeddings, 0,
                                        1)[th.arange(contextual_embeddings.shape[1]).unsqueeze(1),
                                           masked_positions]
        mlm_scores = self.mlm_decoder(mlm_features)
        return contextual_embeddings, pooled_out, nsp_score, mlm_scores


class QTBertForPretrain(th.nn.Module):
    def __init__(self, backbone_cfg):
        """

        Parameters
        ----------
        backbone_cfg
            The cfg of the backbone model
        """
        super().__init__()

        self.backbone_model = BertModel.from_cfg(backbone_cfg)
        self.quickthought = th.nn.Sequential(
            th.nn.Linear(out_features=self.backbone_model.units,
                         in_features=self.backbone_model.units),
            get_activation(self.backbone_model.activation),
            th.nn.LayerNorm(self.backbone_model.units, eps=self.backbone_model.layer_norm_eps))
        self.mlm_decoder = th.nn.Sequential(
            th.nn.Linear(out_features=self.backbone_model.units,
                         in_features=self.backbone_model.units),
            get_activation(self.backbone_model.activation),
            th.nn.LayerNorm(self.backbone_model.units, eps=self.backbone_model.layer_norm_eps),
            th.nn.Linear(out_features=self.backbone_model.vocab_size,
                         in_features=self.backbone_model.units))
        # TODO such weight sharing not supported in torchscript
        self.mlm_decoder[-1].weight = self.backbone_model.word_embed.weight

    @property
    def layout(self):
        return self.backbone_model.layout

    def forward(self, inputs, token_types, valid_length, masked_positions):
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a bert model.

        Parameters
        ----------
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
            The masked position of the sequence with respect to flattened batch
            Shape (N, ) for N masked positions across whole batch.

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
            Shape (N, vocab_size)
        """
        assert len(inputs) % 2 == 0, 'Model expects QuickThought paired inputs'
        contextual_embeddings, pooled_out = self.backbone_model(inputs, token_types, valid_length)
        if self.layout == 'NT':
            mlm_features = contextual_embeddings.flatten(0, 1)[masked_positions]
        else:
            mlm_features = th.transpose(contextual_embeddings, 0, 1).flatten(0, 1)[masked_positions]
        mlm_scores = self.mlm_decoder(mlm_features)
        qt_embeddings = self.quickthought(pooled_out)
        qt_similarity = self._cosine_similarity(qt_embeddings[:len(inputs) // 2],
                                                qt_embeddings[len(inputs) // 2:])
        return contextual_embeddings, pooled_out, mlm_scores, qt_similarity

    def _cosine_similarity(self, a, b):
        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        return th.mm(a_norm, b_norm.transpose(0, 1))


def init_weights(module):
    if type(module) in (th.nn.Linear, th.nn.Embedding):
        th.nn.init.trunc_normal_(module.weight, mean=0, std=0.02, a=-0.04, b=0.04)
        if type(module) == th.nn.Linear and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, th.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    else:
        logging.debug(f'Not performing custom initialization for {type(module)}')
