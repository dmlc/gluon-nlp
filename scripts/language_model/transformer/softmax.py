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
"""Attention cells."""

__all__ = ['AdaptiveLogSoftmaxWithLoss', 'ProjectedLogSoftmaxWithLoss']

from typing import List, Optional

import mxnet as mx


class ProjectedLogSoftmaxWithLoss(mx.gluon.HybridBlock):
    """ProjectedLogSoftmaxWithLoss"""

    def __init__(self, vocab_size: int, embed_size: int, units: int, use_bias: bool = True,
                 project_same_dim: bool = True, projection_initializer=None,
                 embedding_initializer=None, tie_embeddings: bool = False,
                 tie_projections: bool = False, prefix: Optional[str] = None,
                 params: Optional[mx.gluon.ParameterDict] = None):
        super().__init__(prefix=prefix, params=params)
        self._vocab_size = vocab_size
        self._embed_size = embed_size
        self._use_bias = use_bias
        self._units = units
        self._project_same_dim = project_same_dim
        self._embedding_initializer = embedding_initializer
        self._projection_initializer = projection_initializer
        self._tie_embeddings = tie_embeddings
        self._tie_projections = tie_projections

        self._projections_name = '{}projection_weight'
        self._embeddings_name = '{}embedding_weight'
        with self.name_scope():
            if units != embed_size or project_same_dim:
                name = self._get_param_name('projection')
                param = self.params.get(name, shape=(units, embed_size),
                                        init=self._projection_initializer)
                setattr(self, name, param)

            name = self._get_param_name('embedding')
            param = self.params.get(name, shape=(vocab_size, embed_size),
                                    init=self._embedding_initializer)
            setattr(self, name, param)
            if use_bias:
                name = 'outembedding_bias'
                param = self.params.get(name, shape=(self._vocab_size, ))
                setattr(self, name, param)

    def _get_param_name(self, name):
        if name == 'projection':
            return self._projections_name.format('' if self._tie_projections else 'out')
        elif name == 'embedding':
            return self._embeddings_name.format('' if self._tie_embeddings else 'out')
        else:
            raise ValueError('Invalid name')

    def hybrid_forward(self, F, hidden, target, **params):  # pylint: disable=arguments-differ
        """Compute adaptive softmax.

        Parameters
        ----------
        hidden : Symbol or NDArray
            Inputs of shape [batch_size, sequence_length, units]
        target : Symbol or NDArray
            Targets of shape [batch_size, sequence_length]

        Returns
        -------
        out : Symbol or NDArray
            Negative log likelihood of targets with shape [batch_size,
            sequence_length]
        """
        if target is None:  # TODO support None or add separate log_prob method
            raise NotImplementedError()

        # Work with flat data for simplicity
        target_flat = target.reshape((-1, ))
        hidden = F.reshape(hidden, shape=(-1, 0), reverse=True)

        # Helper arrays
        if F is mx.nd:
            range_bs_len = mx.nd.arange(target_flat.shape[0], dtype=target_flat.dtype,
                                        ctx=target_flat.context)
        else:
            # Shape inference fails when relying on F.stack(range_bs_len, ...)
            # below. Thus add zeros of intended shape here to simplify the
            # shape inference problem.
            range_bs_len = F.zeros_like(target_flat) + F.arange(start=0, stop=None,
                                                                infer_range=True)

        if self._units != self._embed_size or self._project_same_dim:
            name = self._get_param_name('projection')
            hidden = F.FullyConnected(data=hidden, weight=F.transpose(params[name]), no_bias=True,
                                      flatten=False, num_hidden=self._embed_size)

        name = self._get_param_name('embedding')
        logits = F.FullyConnected(data=hidden, weight=params[name],
                                  bias=params['outembedding_bias'] if self._use_bias else None,
                                  no_bias=not self._use_bias, flatten=False,
                                  num_hidden=self._vocab_size)
        logprob = F.log_softmax(logits)
        target_ = F.stack(range_bs_len, target_flat)
        out = F.gather_nd(logprob, indices=target_)

        out = F.reshape_like(out, target)

        return -out


class AdaptiveLogSoftmaxWithLoss(mx.gluon.HybridBlock):
    """Efficient softmax approximation

    Grave, E., Joulin, A., Cissé, M., Jégou, H., & others, (2017). Efficient
    softmax approximation for GPUs. In , Proceedings of the 34th International
    Conference on Machine Learning-Volume 70 (pp. 1302–1310).

    Parameters
    ----------
    vocab_size
    embed_size
    units
        Feature dimension of inputs. Must be specified, as shape inference
        would fail if the first batch does not contain target indices of every
        cluster.
    cutoffs
        Ordered list of cutoff values for the clusters.
    div_val
        Division value to obtain embed_size per cluster. For cluster i:
        embed_size / div_val**i.
    use_bias
        Use a bias for the output layer.
    projection_initializer
        Initializer for the projection layers.
    embedding_initializer
        Initializer for the output layers and cluster weights. Called
        embedding_initializer, as the parameters may be tied to the embedding
        parameters of AdaptiveEmbedding.
    tie_embeddings
        Share embedding parameters with an AdaptiveEmbedding Block? If True, the
        params argument must be provided and set to the ParameterDict of the
        AdaptiveEmbedding Block.
    tie_projections
        Share projection parameters with an AdaptiveEmbedding Block? If True, the
        params argument must be provided and set to the ParameterDict of the
        AdaptiveEmbedding Block. tie_projections should be a list of boolean
        values, specifying if the projection weights for the respective
        parameter are to be shared or not.

    """

    def __init__(self, vocab_size: int, embed_size: int, units: int, cutoffs: List[int],
                 div_val: int = 1, use_bias: bool = True, project_same_dim: bool = True,
                 projection_initializer=None, embedding_initializer=None,
                 tie_embeddings: bool = False, tie_projections: Optional[List[bool]] = None,
                 prefix: Optional[str] = None, params: Optional[mx.gluon.ParameterDict] = None):
        super().__init__(prefix=prefix, params=params)
        self._vocab_size = vocab_size
        self._embed_size = embed_size
        self._cutoffs = [0] + cutoffs + [vocab_size]
        self._div_val = div_val
        self._use_bias = use_bias
        self._units = units
        self._project_same_dim = project_same_dim
        self._embedding_initializer = embedding_initializer
        self._projection_initializer = projection_initializer
        self._tie_embeddings = tie_embeddings
        self._tie_projections = tie_projections

        # Sanity checks
        if cutoffs != sorted(cutoffs):
            raise ValueError('cutoffs must be a sorted list of cutoff values. '
                             'Got {}, but expected {}'.format(cutoffs, sorted(cutoffs)))
        if not cutoffs:
            raise ValueError('cutoffs must not be empty. Got {}'.format(cutoffs))
        if cutoffs[0] <= 0:
            raise ValueError('The first cutoff value ({}) must be greater 0.'.format(cutoffs[0]))
        if cutoffs[-1] >= vocab_size:
            raise ValueError(
                'The last cutoff value ({}) must be smaller than vocab_size ({}).'.format(
                    cutoffs[-1], vocab_size))

        if tie_embeddings:
            assert params is not None
        if tie_projections is not None:
            assert params is not None
            if div_val == 1:
                if self._units == self._embed_size:
                    assert len(tie_projections) == 0
                elif len(tie_projections) != 1:
                    raise ValueError(
                        'tie_projections should be None or a boolean for every cluster. '
                        'As div_val == 1 there is only a single cluster. But got ({}).'.format(
                            tie_projections))
            if len(tie_projections) != len(cutoffs) + 1:
                raise ValueError(
                    'tie_projections should be None or a boolean for every cluster. '
                    'It must thus have len(cutoffs) + 1. But got ({}) for cutoffs ({}).'.format(
                        tie_projections, cutoffs))

        self._projections_name = '{}projection{}_weight'
        self._embeddings_name = '{}embedding{}_weight'
        with self.name_scope():
            if self._div_val == 1:
                if self._units != self._embed_size or project_same_dim:
                    name = self._get_param_name('projection', 0)
                    param = self.params.get(name, shape=(self._units, self._embed_size),
                                            init=self._projection_initializer)
                    setattr(self, name, param)

                name = self._get_param_name('embedding', 0)
                param = self.params.get(name, shape=(self._vocab_size, self._embed_size),
                                        init=self._embedding_initializer)
                setattr(self, name, param)
                if use_bias:
                    name = 'outembedding0_bias'
                    param = self.params.get(name, shape=(self._vocab_size, ))
                    setattr(self, name, param)
            else:
                for i, (l_idx, r_idx) in enumerate(zip(self._cutoffs, self._cutoffs[1:])):
                    if self._units != self._embed_size // self._div_val**i or project_same_dim:
                        name = self._get_param_name('projection', i)
                        param = self.params.get(
                            name, shape=(self._units, self._embed_size // self._div_val**i),
                            init=self._projection_initializer)
                        setattr(self, name, param)

                    name = self._get_param_name('embedding', i)
                    param = self.params.get(
                        name, shape=(r_idx - l_idx, self._embed_size // self._div_val**i),
                        init=self._embedding_initializer)
                    setattr(self, name, param)
                    if use_bias:
                        name = 'outembedding{}_bias'.format(i)
                        param = self.params.get(name, shape=(r_idx - l_idx, ))
                        setattr(self, name, param)

                if self._div_val != 1:
                    self.cluster = mx.gluon.nn.Dense(len(cutoffs), flatten=False,
                                                     in_units=embed_size,
                                                     weight_initializer=embedding_initializer)

    def _get_param_name(self, name, i):
        if name == 'projection':
            tied = self._tie_projections is not None and self._tie_projections[i]
            return self._projections_name.format('' if tied else 'out', i)
        elif name == 'embedding':
            return self._embeddings_name.format('' if self._tie_embeddings else 'out', i)
        else:
            raise ValueError('Invalid name')

    def hybrid_forward(self, F, hidden, target, **params):  # pylint: disable=arguments-differ
        """Compute adaptive softmax.

        Parameters
        ----------
        hidden : Symbol or NDArray
            Inputs of shape [batch_size, sequence_length, units]
        target : Symbol or NDArray
            Targets of shape [batch_size, sequence_length]

        Returns
        -------
        out : Symbol or NDArray
            Negative log likelihood of targets with shape [batch_size,
            sequence_length]
        """
        if target is None:  # TODO support None or add separate log_prob method
            raise NotImplementedError()

        # Work with flat data for simplicity
        target_flat = target.reshape((-1, ))
        hidden = F.reshape(hidden, shape=(-1, 0), reverse=True)

        # Helper arrays
        if F is mx.nd:
            range_bs_len = mx.nd.arange(target_flat.shape[0], dtype=target_flat.dtype,
                                        ctx=target_flat.context)
        else:
            # Shape inference fails when relying on F.stack(range_bs_len, ...)
            # below. Thus add zeros of intended shape here to simplify the
            # shape inference problem.
            range_bs_len = F.zeros_like(target_flat) + F.arange(start=0, stop=None,
                                                                infer_range=True)

        if self._div_val == 1:
            if self._units != self._embed_size or self._project_same_dim:
                name = self._get_param_name('projection', 0)
                hidden = F.FullyConnected(data=hidden, weight=F.transpose(params[name]),
                                          no_bias=True, flatten=False, num_hidden=self._embed_size)

            name = self._get_param_name('embedding', 0)
            logits = F.FullyConnected(data=hidden, weight=params[name],
                                      bias=params['outembedding0_bias'] if self._use_bias else None,
                                      no_bias=not self._use_bias, flatten=False,
                                      num_hidden=self._vocab_size)
            logprob = F.log_softmax(logits)
            target_ = F.stack(range_bs_len, target_flat)
            out = F.gather_nd(logprob, indices=target_)
        else:
            # Prepare output
            if F is mx.nd:
                assert target.dtype == hidden.dtype
            out = F.zeros_like(target_flat)

            for i, (l_idx, r_idx) in enumerate(zip(self._cutoffs, self._cutoffs[1:])):
                if self._units != self._embed_size // self._div_val**i or self._project_same_dim:
                    name = self._get_param_name('projection', i)
                    proj_i = F.FullyConnected(data=hidden, weight=F.transpose(params[name]),
                                              no_bias=True, flatten=False,
                                              num_hidden=self._embed_size // self._div_val**i)
                else:
                    proj_i = hidden
                # Shape [batch_size * sequence_length, r_idx - l_idx]
                name = self._get_param_name('embedding', i)
                logits_i = F.FullyConnected(
                    data=proj_i, weight=params[name],
                    bias=params['outembedding{}_bias'.format(i)] if self._use_bias else None,
                    no_bias=not self._use_bias, flatten=False, num_hidden=r_idx - l_idx)
                if i == 0:  # Shortlist
                    logits_cluster = self.cluster(proj_i)
                    logits_shortlist_cluster = F.concat(logits_i, logits_cluster, dim=1)
                    logprob_shortlist_cluster = F.log_softmax(logits_shortlist_cluster)

                    logprob_i = F.slice_axis(logprob_shortlist_cluster, axis=1, begin=0,
                                             end=-(len(self._cutoffs) - 2))
                    logprob_cluster = F.slice_axis(logprob_shortlist_cluster, axis=1,
                                                   begin=-(len(self._cutoffs) - 2), end=None)
                else:  # Tail cluster
                    logprob_i = F.broadcast_add(
                        F.log_softmax(logits_i),
                        F.gather_nd(logprob_cluster,
                                    F.stack(range_bs_len,
                                            F.ones_like(range_bs_len) * (i - 1))).expand_dims(1))

                # Targets limited to current cluster
                cond_i = F.broadcast_logical_and(target_flat >= l_idx, target_flat < r_idx)
                target_i = F.where(cond_i, target_flat - l_idx, F.zeros_like(target_flat))
                target_i = F.stack(range_bs_len, target_i)

                # Copy for targets that fall into the current cluster to out
                out_i = F.gather_nd(logprob_i, indices=target_i)
                out = F.where(cond_i, out_i, out)

        out = F.reshape_like(out, target)

        return -out
