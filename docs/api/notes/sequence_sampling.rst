Sentence Generation from Language Model
---------------------------------------

In this tutorial, we show how to load a pre-trained language model on wikitext-2 in GluonNLP Toolkit
model zoo and use beam search sampler and sequence sampler to generate sentences.

Load Pre-trained Language Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    >>> import mxnet as mx
    >>> import gluonnlp as nlp
    >>> ctx = mx.cpu()

To load the pre-trained model, GluonNLP API provides a convenient way as shown in the following codes:

.. code:: python

    >>> lm_model, vocab = nlp.model.get_model(name='awd_lstm_lm_1150',
    >>>                                       dataset_name='wikitext-2',
    >>>                                       pretrained=True,
    >>>                                       ctx=ctx)


Sampling a Sequence
~~~~~~~~~~~~~~~~~~~
Generating sequences from the language model is about generating sequences that are likely to occur according to the language model. Language model predicts the likelihood of a word occurring at a particular time step, given the context from prior time steps. Given that at each time step, the possible output is any word from the vocabulary whose size is V, the number of all possible outcomes for a sequence of length T is thus V^T. While finding the absolute optimal outcome quickly becomes intractable as time step increases, there are still many ways to sample reasonably good sequences. GluonNLP provides two such samples: ``SequenceSampler`` and ``BeamSearchSampler``.

Beam Search Sampler
+++++++++++++++++++

Now we need to define a scorer function, which is used to evaluate the scores of all the candidates. This can be achieved
by using ``BeamSearchScorer``.

.. code:: python

   >>> scorer = nlp.model.BeamSearchScorer(alpha=0, K=5, from_logits=False)

The ``BeamSearchScorer`` is a simple ``HybridBlock`` that implements the scoring function with
length penalty in Google NMT paper.
``alpha`` and ``K`` correspond to the :math:\alpha parameter and K parameter of the length penalty term, respectively.
See the `source code <http://gluon-nlp.mxnet.io/_modules/gluonnlp/model/sequence_sampler.html#BeamSearchScorer>`__ to
get a sense of how to implement your own.

The next step is to define a beam search sampler. Before setting up a sampler, we need to construct a decoder function.

.. code:: python

    >>> class LMDecoder(object):
    >>>     def __init__(self, model):
    >>>         self._model = model
    >>>     def __call__(self, inputs, states):
    >>>         outputs, states = self._model(mx.nd.expand_dims(inputs, axis=0), states)
    >>>         return outputs[0], states
    >>>     def state_info(self, *arg, **kwargs):
    >>>         return self._model.state_info(*arg, **kwargs)
    >>> decoder = LMDecoder(lm_model)

Given a scorer and decoder, we are ready to create a sampler. We use the symbol '.' to indicate the end of a sentence (EOS).
We can use ``vocab`` to get the index of the EOS, and then feed the index to the sampler. The following code shows how
to construct a beam search sampler. We create a sampler with 4 beams and a maximum sample length of 20.

.. code:: python

    >>> eos_id = vocab['.']
    >>> sampler = nlp.model.BeamSearchSampler(beam_size=4,
    >>>                                       decoder=decoder,
    >>>                                       eos_id=eos_id,
    >>>                                       scorer=scorer,
    >>>                                       max_length=20)

Generate Sequences w/ Beam Search
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, we are going to generate sentences starting with "I love it" using beam search first. We feed ['I', 'Love'] to the
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
    ['I love it .', -1.1241297]
    ['I love it , but it is not a <unk> .', -15.624882]
    ['I love it , but it is not a <unk> , but it is not a <unk> .', -28.37084]

Sequence Sampler
++++++++++++++++

The previous generation results may look a bit boring. Now, let's use sequence sampler to get some more exciting results.

``SequenceSampler`` simply samples from the contextual multinomial distribution produced by the language model at each time step. Since we may want to control how "sharp" the distribution is to tradeoff diversity with correctness, we can use the ``temperature`` option in ``SequenceSampler``, which controls the temperature of the softmax function.

.. code:: python

     >>> sampler = nlp.model.SequenceSampler(beam_size=4,
     >>>                                     decoder=decoder,
     >>>                                     eos_id=eos_id,
     >>>                                     max_length=100,
     >>>                                     temperature=0.97)


Generate Sequences w/ Sequence Sampler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now, use the sequence sampler created to sample sequences based on the same inputs used previously.

.. code:: python

    >>> bos = 'I love it'.split()
    >>> bos_ids = [vocab[ele] for ele in bos]
    >>> begin_states = lm_model.begin_state(batch_size=1, ctx=ctx)
    >>> if len(bos_ids) > 1:
    >>>     _, begin_states = lm_model(mx.nd.expand_dims(mx.nd.array(bos_ids[:-1]), axis=1),
    >>>                                begin_states)
    >>> inputs = mx.nd.full(shape=(1,), ctx=ctx, val=bos_ids[-1])
    >>> samples, scores, valid_lengths = sampler(inputs, begin_states)
    >>> samples = samples[0].asnumpy()
    >>> scores = scores[0].asnumpy()
    >>> valid_lengths = valid_lengths[0].asnumpy()
    >>> sentence = bos[:-1] + [vocab.idx_to_token[ele] for ele in samples[0][:valid_lengths[0]]]
    >>> print('Generation Result:')
    >>> for i in range(5):
    >>>     sentence = bos[:-1] + [vocab.idx_to_token[ele] for ele in samples[i][:valid_lengths[i]]]
    >>>     print([' '.join(sentence), scores[i]])

    Generation Result:
    ['I love it on the outskirts of the country .', -16.738558]
    ['I love it during two months .', -16.041046]
    ['I love it <unk> .', -6.295361]
    ['I love it , which can be taken for be contrary to current for well , importantly the relaunched anniversary resistant .', -112.43505]
    ['I love it as .', -9.422777]
