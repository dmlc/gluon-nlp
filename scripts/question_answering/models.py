import mxnet as mx
from mxnet import np, npx
from mxnet.gluon import nn, HybridBlock
from mxnet.util import use_np
from gluonnlp.layers import get_activation
from gluonnlp.op import select_vectors_by_position
from gluonnlp.attention_cell import masked_logsoftmax, masked_softmax


@use_np
class ModelForQABasic(HybridBlock):
    """The basic pretrained model for QA. It is used in the original BERT paper for SQuAD 1.1.

    Here, we directly use the backbone network to extract the contextual embeddings and use
    another dense layer to map the contextual embeddings to the start scores and end scores.

    use_segmentation is used to mark whether we segment the input sentence. In RoBERTa and XLMR,
    this flag is set to True, then the QA model no longer accept `token_types` as valid input.

    - use_segmentation=True:
        tokens :      <CLS> Question <SEP> Context <SEP>
        token_typess:  0       0       0      1      1

    - use_segmentation=False:
        tokens :      <CLS> Question <SEP> Context <SEP>
        token_typess:  None
    """
    def __init__(self, backbone, weight_initializer=None, bias_initializer=None,
                 use_segmentation=True):
        super().__init__()

        self.backbone = backbone
        self.use_segmentation = use_segmentation
        self.qa_outputs = nn.Dense(units=2, flatten=False,
                                   weight_initializer=weight_initializer,
                                   bias_initializer=bias_initializer)

    def forward(self, tokens, token_types, valid_length, p_mask):
        """

        Parameters
        ----------
        tokens
            Shape (batch_size, seq_length)
            The merged input tokens
        token_types
            Shape (batch_size, seq_length)
            Token types for the sequences, used to indicate whether the word belongs to the
            first sentence or the second one.
        valid_length
            Shape (batch_size,)
            Valid length of the sequence. This is used to mask the padded tokens.
        p_mask
            The mask that is associated with the tokens.

        Returns
        -------
        start_logits
            Shape (batch_size, sequence_length)
            The log-softmax scores that the position is the start position.
        end_logits
            Shape (batch_size, sequence_length)
            The log-softmax scores that the position is the end position.
        """
        # Get contextual embedding with the shape (batch_size, sequence_length, C)
        if self.use_segmentation:
            contextual_embeddings = self.backbone(tokens, token_types, valid_length)
        else:
            contextual_embeddings = self.backbone(tokens, valid_length)
        scores = self.qa_outputs(contextual_embeddings)
        start_scores = scores[:, :, 0]
        end_scores = scores[:, :, 1]
        start_logits = masked_logsoftmax(start_scores, mask=p_mask, axis=-1)
        end_logits = masked_logsoftmax(end_scores, mask=p_mask, axis=-1)
        return start_logits, end_logits

    def inference(self, tokens, token_types, valid_length, p_mask,
                  start_top_n: int = 5, end_top_n: int = 5):
        """Get the inference result with beam search

        Parameters
        ----------
        tokens
            The input tokens. Shape (batch_size, sequence_length)
        token_types
            The input token types. Shape (batch_size, sequence_length)
        valid_length
            The valid length of the tokens. Shape (batch_size,)
        p_mask
            The mask which indicates that some tokens won't be used in the calculation.
            Shape (batch_size, sequence_length)
        start_top_n
            The number of candidates to select for the start position.
        end_top_n
            The number of candidates to select for the end position.

        Returns
        -------
        start_top_logits
            The top start logits
            Shape (batch_size, start_top_n)
        start_top_index
            Index of the top start logits
            Shape (batch_size, start_top_n)
        end_top_logits
            The top end logits.
            Shape (batch_size, end_top_n)
        end_top_index
            Index of the top end logits
            Shape (batch_size, end_top_n)
        """
        # Shape (batch_size, sequence_length, C)
        if self.use_segmentation:
            contextual_embeddings = self.backbone(tokens, token_types, valid_length)
        else:
            contextual_embeddings = self.backbone(tokens, valid_length)
        scores = self.qa_outputs(contextual_embeddings)
        start_scores = scores[:, :, 0]
        end_scores = scores[:, :, 1]
        start_logits = masked_logsoftmax(start_scores, mask=p_mask, axis=-1)
        end_logits = masked_logsoftmax(end_scores, mask=p_mask, axis=-1)
        # The shape of start_top_index will be (..., start_top_n)
        start_top_logits, start_top_index = mx.npx.topk(start_logits, k=start_top_n, axis=-1,
                                                        ret_typ='both')
        # Note that end_top_index and end_top_log_probs have shape (bsz, start_n_top, end_n_top)
        # So that for each start position, there are end_n_top end positions on the third dim.
        end_top_logits, end_top_index = mx.npx.topk(end_logits, k=end_top_n, axis=-1,
                                                    ret_typ='both')
        return start_top_logits, start_top_index, end_top_logits, end_top_index


@use_np
class ModelForQAConditionalV1(HybridBlock):
    """Here, we use three networks to predict the start scores, end scores and answerable scores.

    We formulate p(start, end, answerable | contextual_embedding) as the product of the
    following three terms:

    - p(start | contextual_embedding)
    - p(end | start, contextual_embedding)
    - p(answerable | contextual_embedding)

    In the inference phase, we are able to use beam search to do the inference.

    use_segmentation is used to mark whether we segment the input sentence. In RoBERTa and XLMR,
    this flag is set to True, then the QA model no longer accept `token_types` as valid input.

    - use_segmentation=True:
        tokens :      <CLS> Question <SEP> Context <SEP>
        token_typess:  0       0       0      1      1

    - use_segmentation=False:
        tokens :      <CLS> Question <SEP> Context <SEP>
        token_typess:  None
    """
    def __init__(self, backbone, units=768, layer_norm_eps=1E-12, dropout_prob=0.1,
                 activation='tanh', weight_initializer=None, bias_initializer=None,
                 use_segmentation=True):
        super().__init__()
        self.backbone = backbone
        self.use_segmentation = use_segmentation
        self.start_scores = nn.Dense(1, flatten=False,
                                     weight_initializer=weight_initializer,
                                     bias_initializer=bias_initializer)
        self.end_scores = nn.HybridSequential()
        self.end_scores.add(nn.Dense(units, flatten=False,
                                     weight_initializer=weight_initializer,
                                     bias_initializer=bias_initializer))
        self.end_scores.add(get_activation(activation))
        self.end_scores.add(nn.LayerNorm(epsilon=layer_norm_eps))
        self.end_scores.add(nn.Dense(1, flatten=False,
                                     weight_initializer=weight_initializer,
                                     bias_initializer=bias_initializer))
        self.answerable_scores = nn.HybridSequential()
        self.answerable_scores.add(nn.Dense(units, flatten=False,
                                            weight_initializer=weight_initializer,
                                            bias_initializer=bias_initializer))
        self.answerable_scores.add(get_activation(activation))
        self.answerable_scores.add(nn.Dropout(dropout_prob))
        self.answerable_scores.add(nn.Dense(2, flatten=False,
                                            weight_initializer=weight_initializer,
                                            bias_initializer=bias_initializer))
        self.quantized_backbone = None

    def get_start_logits(self, contextual_embedding, p_mask):
        """

        Parameters
        ----------
        contextual_embedding
            Shape (batch_size, sequence_length, C)

        Returns
        -------
        start_logits
            Shape (batch_size, sequence_length)
        """
        start_scores = np.squeeze(self.start_scores(contextual_embedding), -1)
        start_logits = masked_logsoftmax(start_scores, mask=p_mask, axis=-1)
        return start_logits

    def get_end_logits(self, contextual_embedding, start_positions, p_mask):
        """

        Parameters
        ----------
        contextual_embedding
            Shape (batch_size, sequence_length, C)
        start_positions
            Shape (batch_size, N)
            We process multiple candidates simultaneously
        p_mask
            Shape (batch_size, sequence_length)

        Returns
        -------
        end_logits
            Shape (batch_size, N, sequence_length)
        """
        # Select the features at the start_positions
        # start_feature will have shape (batch_size, N, C)
        start_features = select_vectors_by_position(contextual_embedding, start_positions)
        # Concatenate the start_feature and the contextual_embedding
        contextual_embedding = np.expand_dims(contextual_embedding, axis=1)  # (B, 1, T, C)
        start_features = np.expand_dims(start_features, axis=2)  # (B, N, 1, C)
        concat_features = np.concatenate([npx.broadcast_like(start_features,
                                                                 contextual_embedding, 2, 2),
                                            npx.broadcast_like(contextual_embedding,
                                                                 start_features, 1, 1)],
                                           axis=-1)  # (B, N, T, 2C)
        end_scores = self.end_scores(concat_features)
        end_scores = np.squeeze(end_scores, -1)
        end_logits = masked_logsoftmax(end_scores, mask=np.expand_dims(p_mask, axis=1),
                                       axis=-1)
        return end_logits

    def get_answerable_logits(self, contextual_embedding, p_mask):
        """Get the answerable logits.

        Parameters
        ----------
        contextual_embedding
            Shape (batch_size, sequence_length, C)
        p_mask
            Shape (batch_size, sequence_length)
            Mask the sequence.
            0 --> Denote that the element is masked,
            1 --> Denote that the element is not masked

        Returns
        -------
        answerable_logits
            Shape (batch_size, 2)
        """
        # Shape (batch_size, sequence_length)
        start_scores = np.squeeze(self.start_scores(contextual_embedding), -1)
        start_score_weights = masked_softmax(start_scores, p_mask, axis=-1)
        start_agg_feature = npx.batch_dot(np.expand_dims(start_score_weights, axis=1),
                                          contextual_embedding)
        start_agg_feature = np.squeeze(start_agg_feature, 1)
        cls_feature = contextual_embedding[:, 0, :]
        answerable_scores = self.answerable_scores(np.concatenate([start_agg_feature,
                                                                  cls_feature], axis=-1))
        answerable_logits = npx.log_softmax(answerable_scores, axis=-1)
        return answerable_logits

    def forward(self, tokens, token_types, valid_length, p_mask, start_position):
        """

        Parameters
        ----------
        tokens
            Shape (batch_size, sequence_length)
        token_types
            Shape (batch_size, sequence_length)
        valid_length
            Shape (batch_size,)
        p_mask
            Shape (batch_size, sequence_length)
        start_position
            Shape (batch_size,)

        Returns
        -------
        start_logits
            Shape (batch_size, sequence_length)
        end_logits
            Shape (batch_size, sequence_length)
        answerable_logits
        """
        backbone_net = self.backbone
        if self.quantized_backbone != None:
           backbone_net = self.quantized_backbone

        if self.use_segmentation:
            contextual_embeddings = backbone_net(tokens, token_types, valid_length)
        else:
            contextual_embeddings = backbone_net(tokens, valid_length)
        start_logits = self.get_start_logits(contextual_embeddings, p_mask)
        end_logits = self.get_end_logits(contextual_embeddings,
                                         np.expand_dims(start_position, axis=1),
                                         p_mask)
        end_logits = np.squeeze(end_logits, axis=1)
        answerable_logits = self.get_answerable_logits(contextual_embeddings, p_mask)
        return start_logits, end_logits, answerable_logits

    def inference(self, tokens, token_types, valid_length, p_mask,
                  start_top_n: int = 5, end_top_n: int = 5):
        """Get the inference result with beam search

        Parameters
        ----------
        tokens
            The input tokens. Shape (batch_size, sequence_length)
        token_types
            The input token types. Shape (batch_size, sequence_length)
        valid_length
            The valid length of the tokens. Shape (batch_size,)
        p_mask
            The mask which indicates that some tokens won't be used in the calculation.
            Shape (batch_size, sequence_length)
        start_top_n
            The number of candidates to select for the start position.
        end_top_n
            The number of candidates to select for the end position.

        Returns
        -------
        start_top_logits
            The top start logits
            Shape (batch_size, start_top_n)
        start_top_index
            Index of the top start logits
            Shape (batch_size, start_top_n)
        end_top_logits
            The top end logits.
            Shape (batch_size, start_top_n, end_top_n)
        end_top_index
            Index of the top end logits
            Shape (batch_size, start_top_n, end_top_n)
        answerable_logits
            The answerable logits. Here 0 --> answerable and 1 --> not answerable.
            Shape (batch_size, sequence_length, 2)
        """
        backbone_net = self.backbone
        if self.quantized_backbone != None:
           backbone_net = self.quantized_backbone

        # Shape (batch_size, sequence_length, C)
        if self.use_segmentation:
            contextual_embeddings = backbone_net(tokens, token_types, valid_length)
        else:
            contextual_embeddings = backbone_net(tokens, valid_length)

        start_logits = self.get_start_logits(contextual_embeddings, p_mask)
        # The shape of start_top_index will be (..., start_top_n)
        start_top_logits, start_top_index = mx.npx.topk(start_logits, k=start_top_n, axis=-1,
                                                        ret_typ='both')
        end_logits = self.get_end_logits(contextual_embeddings, start_top_index, p_mask)
        # Note that end_top_index and end_top_log_probs have shape (bsz, start_n_top, end_n_top)
        # So that for each start position, there are end_n_top end positions on the third dim.
        end_top_logits, end_top_index = mx.npx.topk(end_logits, k=end_top_n, axis=-1,
                                                    ret_typ='both')
        answerable_logits = self.get_answerable_logits(contextual_embeddings, p_mask)
        return start_top_logits, start_top_index, end_top_logits, end_top_index, \
                    answerable_logits
