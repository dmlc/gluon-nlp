Beam Search for Sentence Generation
-----------------------------------

In this tutorial, we show how to load a pretrained language model on wikitext-2 in Gluon NLP Toolkit
model zoo and use beam search sampler to generate sentences.

Load Pretrained Language Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    >>> import mxnet as mx
    >>> import gluonnlp as nlp
    >>> ctx = mx.cpu()

To load the pretrained model, gluon nlp API provides a convenient way as shown in the following codes:

.. code:: python

    >>> lm_model, vocab = nlp.model.get_model(name='awd_lstm_lm_1150',
    >>>                                       dataset_name='wikitext-2',
    >>>                                       pretrained=True,
    >>>                                       ctx=ctx)


Build the Beam Search Sampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we need to define a scorer function, which is used to evaluate the scores of all the candidates. This can be achieved
by using ``BeamSearchScorer``.

.. code:: python

   >>> scorer = nlp.model.BeamSearchScorer(alpha=0, K=5)

``alpha`` and ``K`` correspond to the :math:\alpha parameter and K parameter of the length penalty term in GNMT paper, respectively.
The next step is to define a beam search sampler. Before setting up a sampler, we need to construct a decoder function.
In this example, we use ``log_softmax`` to map the output scores to log-likelihoods. Note that BeamSearchSampler takes as inputs and states with format NTC while
language model takes inputs and states with format TNC. We thus need to transform the layout of the data and states.

.. code:: python

    >>> # Transform the layout to NTC
    >>> def _transform_layout(data):
    >>>     if isinstance(data, list):
    >>>          return [_transform_layout(ele) for ele in data]
    >>>     elif isinstance(data, mx.nd.NDArray):
    >>>          return mx.nd.transpose(data, axes=(1, 0, 2))
    >>>     else:
    >>>          raise NotImplementedError
    >>>
    >>> def decoder(inputs, states):
    >>>     states = _transform_layout(states)
    >>>     outputs, states = lm_model(mx.nd.expand_dims(inputs, axis=0), states)
    >>>     states = _transform_layout(states)
    >>>     return outputs[0], states

Given a scorer and decoder, we are ready to create a sampler. We use symbol '.' to indicate the end of sentence (EOS).
We can use ``vocab`` to get the index of the EOS, and then feed the index to the sampler. The following codes shows how
to construct a beam search sampler. We will create a sampler with 4 beams and a maximum sample length of 20.

.. code:: python

    >>> eos_id = vocab['.']
    >>> beam_size = 4
    >>> max_length = 20
    >>> sampler = nlp.model.BeamSearchSampler(beam_size=beam_size,
    >>>                                       decoder=decoder,
    >>>                                       eos_id=eos_id,
    >>>                                       scorer=scorer,
    >>>                                       max_length=max_length)

Generate Samples
~~~~~~~~~~~~~~~~

Next, we are going to generate sentences starting with "I love it". We first feed ['I', 'Love'] to the
language model to get the initial states and set the initial input to be the word 'it'. We will then print the top-3 generations.

.. code:: python

    >>> bos = 'I love it'.split()
    >>> bos_ids = [vocab[ele] for ele in bos]
    >>> begin_states = lm_model.begin_state(batch_size=1, ctx=ctx)
    >>> if len(bos_ids) > 1:
    >>>     _, begin_states = lm_model(mx.nd.expand_dims(mx.nd.array(bos_ids[:-1]), axis=1),
    >>>                                begin_states)
    >>> inputs = mx.nd.full(shape=(1,), ctx=ctx, val=bos_ids[-1])
    >>>
    >>> # samples have shape (1, beam_size, length), scores have shape (1, beam_size)
    >>> samples, scores, valid_lengths = sampler(inputs, begin_states)
    >>>
    >>> samples = samples[0].asnumpy()
    >>> scores = scores[0].asnumpy()
    >>> valid_lengths = valid_lengths[0].asnumpy()
    >>> print('Generation Result:')
    >>> for i in range(3):
    >>>     sentence = bos[:-1] + [vocab.idx_to_token[ele] for ele in samples[i][:valid_lengths[i]]]
    >>>     print([' '.join(sentence), scores[i]])

    Generation Result:
    ['I love it , but it is not clear that it is not the same as it is , but it is not .', 239.38823]
    ['I love it , but it is not clear that it is not the same as it is , and it is not .', 238.67413]
    ['I love it , but it is not clear that it is not the same as it is , but it is a .', 237.9876]

Investigate the effect of beam size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The previous generation results do not look so good. Now, let's increase the beam size to see if the generation result looks better.

.. code:: python

    >>> for beam_size in range(4, 17, 4):
    >>>     sampler = nlp.model.BeamSearchSampler(beam_size=beam_size,
    >>>                                           decoder=decoder,
    >>>                                           eos_id=eos_id,
    >>>                                           scorer=scorer,
    >>>                                           max_length=20)
    >>>     samples, scores, valid_lengths = sampler(inputs, begin_states)
    >>>     samples = samples[0].asnumpy()
    >>>     scores = scores[0].asnumpy()
    >>>     valid_lengths = valid_lengths[0].asnumpy()
    >>>     sentence = bos[:-1] + [vocab.idx_to_token[ele] for ele in samples[0][:valid_lengths[0]]]
    >>>     print([beam_size, ' '.join(sentence), scores[0]])

    [4, 'I love it , but it is not clear that it is not the same as it is , but it is not .', 239.38823]
    [8, 'I love it , but it is not clear that it would be difficult to do it , but it is not a .', 243.27402]
    [12, "I love it , but it is impossible to do anything to do it , but I don ’ t think it 's .", 260.26495]
    [16, "I love it , and I don 't know how to do it , but I don ’ t think it would be .", 274.9892]

The scores and generation results are improving.
