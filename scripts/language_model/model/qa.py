"""XLNetForQA models."""

import mxnet as mx
from mxnet.gluon import HybridBlock, Block, loss, nn


class PoolerStartLogits(HybridBlock):
    """ Compute SQuAD start_logits from sequence hidden states. """
    def __init__(self, prefix=None, params=None):
        super(PoolerStartLogits, self).__init__(prefix=prefix, params=params)
        self.dense = nn.Dense(1, flatten=False)

    def __call__(self, hidden_states, p_masks=None):
        # pylint: disable=arguments-differ
        return super(PoolerStartLogits, self).__call__(hidden_states, p_masks)

    def hybrid_forward(self, F, hidden_states, p_mask):
        # pylint: disable=arguments-differ
        """ Args:
            **p_mask**: (`optional`) ``torch.FloatTensor`` of shape `(batch_size, seq_len)`
                invalid position mask such as query and special symbols (PAD, SEP, CLS)
                1.0 means token should be masked.
        """
        x = self.dense(hidden_states).squeeze(-1)
        if p_mask is not None:
            x = x * (1 - p_mask) - 1e30 * p_mask
        return x


class PoolerEndLogits(Block):
    """ Compute SQuAD end_logits from sequence hidden states and start token hidden state."""
    def __init__(self, units=768, is_eval=False, prefix=None, params=None):
        super(PoolerEndLogits, self).__init__(prefix=prefix, params=params)
        self._eval = is_eval
        self._hsz = units
        with self.name_scope():
            self.dense_0 = nn.Dense(units, activation='tanh', flatten=False)
            self.dense_1 = nn.Dense(1, flatten=False)
            self.layernorm = nn.LayerNorm(epsilon=1e-12, in_channels=768)

    def __call__(self,
                 hidden_states,
                 start_states=None,
                 start_positions=None,
                 p_masks=None):
        # pylint: disable=arguments-differ
        return super(PoolerEndLogits,
                     self).__call__(hidden_states, start_states,
                                    start_positions, p_masks)

    def forward(self, hidden_states, start_states, start_positions, p_mask):
        # pylint: disable=arguments-differ
        F = mx.ndarray
        if not self._eval:
            start_states = F.gather_nd(
                hidden_states,
                F.concat(
                    F.contrib.arange_like(hidden_states,
                                          axis=0).expand_dims(1),
                    start_positions.expand_dims(
                        1)).transpose())  #shape(bsz, hsz)
            start_states = start_states.expand_dims(1)
            start_states = F.broadcast_like(
                start_states, hidden_states)  # shape (bsz, slen, hsz)
        x = self.dense_0(F.concat(hidden_states, start_states, dim=-1))
        x = self.layernorm(x)
        x = self.dense_1(x).squeeze(-1)
        if p_mask is not None and self._eval:
            p_mask = p_mask.expand_dims(-1)
            p_mask = F.broadcast_like(p_mask, x)
        if p_mask is not None:
            x = x * (1 - p_mask) - 1e30 * p_mask
        return x


class XLNetPoolerAnswerClass(Block):
    """ Compute SQuAD 2.0 answer class from classification and start tokens hidden states. """
    def __init__(self, units=768, dropout=0.1, prefix=None, params=None):
        super(XLNetPoolerAnswerClass, self).__init__(prefix=prefix,
                                                     params=params)
        with self.name_scope():
            self._units = units
            self.dense_0 = nn.Dense(units,
                                    in_units=2 * units,
                                    activation='tanh',
                                    use_bias=True,
                                    flatten=False)
            self.dense_1 = nn.Dense(1,
                                    in_units=units,
                                    use_bias=False,
                                    flatten=False)
            self._dropout = nn.Dropout(dropout)

    def __call__(self, hidden_states, start_states=None, cls_index=None):
        # pylint: disable=arguments-differ
        return super(XLNetPoolerAnswerClass,
                     self).__call__(hidden_states, start_states, cls_index)
        # pylint: disable=unused-argument

    def forward(self, sequence, start_states, cls_index):
        # pylint: disable=arguments-differ
        # get the cls_token's state, currently the last state
        F = mx.ndarray
        index = F.contrib.arange_like(sequence, axis=0,
                                      ctx=sequence.context).expand_dims(1)
        valid_length_rs = cls_index.reshape((-1, 1)) - 1
        gather_index = F.concat(index, valid_length_rs).T
        cls_token_state = F.gather_nd(sequence, gather_index)

        x = self.dense_0(F.concat(start_states, cls_token_state, dim=-1))
        x = self._dropout(x)
        x = self.dense_1(x).squeeze(-1)
        return x


class XLNetForQA(Block):
    """Model for SQuAD task with XLNet.

    Parameters
    ----------
    bert: XLNet base
    prefix : str or None
        See document of `mx.gluon.Block`.
    params : ParameterDict or None
        See document of `mx.gluon.Block`.
    """
    def __init__(self,
                 xlnet_base,
                 start_top_n=None,
                 end_top_n=None,
                 version_2=False,
                 is_eval=False,
                 units=768,
                 prefix=None,
                 params=None):
        super(XLNetForQA, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.xlnet = xlnet_base
            self.start_top_n = start_top_n
            self.end_top_n = end_top_n
            self.loss = loss.SoftmaxCELoss()
            self.start_logits = PoolerStartLogits()
            self.end_logits = PoolerEndLogits(units=units, is_eval=is_eval)
            self.version2 = version_2
            self.eval = is_eval
            if version_2:
                self.answer_class = XLNetPoolerAnswerClass(units=units)
                self.cls_loss = loss.SigmoidBinaryCrossEntropyLoss()

    def __call__(self,
                 inputs,
                 token_types,
                 valid_length=None,
                 label=None,
                 p_mask=None,
                 is_impossible=None,
                 mems=None):
        #pylint: disable=arguments-differ, dangerous-default-value
        """Generate the unnormalized score for the given the input sequences."""
        valid_length = [] if valid_length is None else valid_length
        return super(XLNetForQA,
                     self).__call__(inputs, token_types, valid_length, p_mask,
                                    label, is_impossible, mems)

    def _padding_mask(self, inputs, valid_length, left_pad=False):
        F = mx.ndarray
        if left_pad:
            # left padding
            valid_length_start = valid_length.astype('int64')
            steps = F.contrib.arange_like(inputs, axis=1) + 1
            ones = F.ones_like(steps)
            mask = F.broadcast_greater(
                F.reshape(steps, shape=(1, -1)),
                F.reshape(valid_length_start, shape=(-1, 1)))
            mask = F.broadcast_mul(
                F.expand_dims(mask, axis=1),
                F.broadcast_mul(ones, F.reshape(ones, shape=(-1, 1))))
        else:
            # right padding
            valid_length = valid_length.astype(inputs.dtype)
            steps = F.contrib.arange_like(inputs, axis=1)
            ones = F.ones_like(steps)
            mask = F.broadcast_lesser(F.reshape(steps, shape=(1, -1)),
                                      F.reshape(valid_length, shape=(-1, 1)))
            mask = F.broadcast_mul(
                F.expand_dims(mask, axis=1),
                F.broadcast_mul(ones, F.reshape(ones, shape=(-1, 1))))
        return mask

    def forward(self, inputs, token_types, valid_length, p_mask, label,
                is_impossible, mems):
        # pylint: disable=arguments-differ
        """Generate the unnormalized score for the given the input sequences.

        Parameters
        ----------
        inputs : NDArray, shape (batch_size, seq_length)
            Input words for the sequences.
        token_types : NDArray, shape (batch_size, seq_length)
            Token types for the sequences, used to indicate whether the word belongs to the
            first sentence or the second one.
        valid_length : NDArray or None, shape (batch_size,)
            Valid length of the sequence. This is used to mask the padded tokens.

        Returns
        -------
        outputs : NDArray
            Shape (batch_size, seq_length, 2)
        """
        if isinstance(valid_length, list) and len(valid_length) == 0:
            valid_length = None
        attention_mask = self._padding_mask(inputs,
                                            valid_length).astype('float32')
        output, _ = self.xlnet(inputs, token_types, mems, attention_mask)
        start_logits = self.start_logits(output,
                                         p_masks=p_mask)  # shape (bsz, slen)
        bsz, slen, hsz = output.shape
        if not self.eval:
            #training
            start_positions, end_positions = label
            end_logit = self.end_logits(output,
                                        start_positions=start_positions,
                                        p_masks=p_mask)
            span_loss = (self.loss(start_logits, start_positions) +
                         self.loss(end_logit, end_positions)) / 2

            cls_loss = None
            total_loss = [span_loss]
            if self.version2:
                start_log_probs = mx.nd.softmax(start_logits, axis=-1)
                start_states = mx.nd.batch_dot(output,
                                               start_log_probs.expand_dims(-1),
                                               transpose_a=True).squeeze(-1)

                cls_logits = self.answer_class(output, start_states,
                                               valid_length)
                cls_loss = self.cls_loss(cls_logits, is_impossible)
                total_loss.append(0.5 * cls_loss)
            total_loss_sum = span_loss + 0.5 * cls_loss if cls_loss is not None else span_loss
            return total_loss, total_loss_sum
        else:
            #inference
            start_log_probs = mx.nd.log_softmax(start_logits,
                                                axis=-1)  # shape (bsz, slen)
            start_top_log_probs, start_top_index = mx.ndarray.topk(
                start_log_probs, k=self.start_top_n, axis=-1,
                ret_typ='both')  # shape (bsz, start_n_top)
            index = mx.nd.concat(*[
                mx.nd.arange(bsz, ctx=start_log_probs.context).expand_dims(1)
            ] * self.start_top_n).reshape(bsz * self.start_top_n, 1)
            start_top_index_rs = start_top_index.reshape((-1, 1))
            gather_index = mx.nd.concat(
                index, start_top_index_rs).T  #shape(2, bsz * start_n_top)
            start_states = mx.nd.gather_nd(output, gather_index).reshape(
                (bsz, self.start_top_n, hsz))  #shape (bsz, start_n_top, hsz)

            start_states = start_states.expand_dims(1)
            start_states = mx.nd.broadcast_to(
                start_states, (bsz, slen, self.start_top_n,
                               hsz))  # shape (bsz, slen, start_n_top, hsz)
            hidden_states_expanded = output.expand_dims(2)
            hidden_states_expanded = mx.ndarray.broadcast_to(
                hidden_states_expanded, shape=start_states.shape
            )  # shape (bsz, slen, start_n_top, hsz)
            end_logits = self.end_logits(
                hidden_states_expanded,
                start_states=start_states,
                p_masks=p_mask)  # shape (bsz, slen, start_n_top)
            end_log_probs = mx.nd.log_softmax(
                end_logits, axis=1)  # shape (bsz, slen, start_n_top)
            # Note that end_top_index and end_top_log_probs have shape (bsz, END_N_TOP, start_n_top)
            # So that for each start position, there are end_n_top end positions on the second dim.
            end_top_log_probs, end_top_index = mx.ndarray.topk(
                end_log_probs, k=self.end_top_n, axis=1,
                ret_typ='both')  # shape (bsz, end_n_top, start_n_top)
            end_top_log_probs = end_top_log_probs.reshape(
                (-1, self.start_top_n * self.end_top_n))
            end_top_index = end_top_index.reshape(
                (-1, self.start_top_n * self.end_top_n))

            start_probs = mx.nd.softmax(start_logits, axis=-1)
            start_states = mx.nd.batch_dot(output,
                                           start_probs.expand_dims(-1),
                                           transpose_a=True).squeeze(-1)

            cls_logits = None
            if self.version2:
                cls_logits = self.answer_class(output, start_states,
                                               valid_length)

            outputs = (start_top_log_probs, start_top_index, end_top_log_probs,
                       end_top_index, cls_logits)
            return outputs
