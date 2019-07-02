# Sequence Generation with Sampling and Beam Search

This tutorial demonstrates how to sample sequences using a
pre-trained language model in the following two ways: with a beam search sampler
and with a sequence sampler.

Let's use `V` to denote the vocabulary size, and `T` to denote the sequence
length. Given a language model, we can sample sequences according to the
probability that they would occur according to our model. At each time step, a
language model predicts the likelihood of each word occurring, given the context
from prior time steps. The outputs at any time step can be any word from the
vocabulary whose size is `V` and thus the number of all possible outcomes for a
sequence of length `T` is thus $$V^T$$.

While sometimes we might want to sample
sentences according to their probability of occurring, at other times we want to
find the sentences that *are most likely to occur*. This is especially true in
the case of language translation where we don't just want to see *a*
translation. We want the *best* translation. While finding the optimal outcome
quickly becomes intractable as time increases, there are still many ways to
sample reasonably good sequences. GluonNLP provides two samplers for generating
from a language model: `BeamSearchSampler` and `SequenceSampler`.

## Loading a pre-trained language model (LM)

Firstly, let's load a pre-trained language model,
from which we will sample sequences. GluonNLP makes this a painless process.

```{.python .input}
import mxnet as mx
import gluonnlp as nlp
ctx = mx.cpu()
lm_model, vocab = nlp.model.get_model(name='awd_lstm_lm_1150',
                                      dataset_name='wikitext-2',
                                      pretrained=True,
                                      ctx=ctx)
```

## Sampling a Sequence with `BeamSearchSampler`

To overcome the exponential complexity in sequence decoding, beam search decodes
greedily, keeping those sequences that are most likely based on the probability
up to the current time step. The size of this subset is called the *beam size*.
Suppose the beam size is `K` and the output vocabulary size is `V`. When
selecting the beams to keep, the beam search algorithm first predicts all
possible successor words from the previous `K` beams, each of which has `V`
possible outputs. This becomes a total of `K*V` paths. Out of these `K*V` paths,
beam search ranks them by their score keeping only the top `K` paths.

Let's take a look how to construct a `BeamSearchSampler`. The
`nlp.model.BeamSearchSampler` class takes the following arguments for
customization and extension:

- beam_size : the beam size
- decoder : callable function of the one-step-ahead decoder
- eos_id : the id of the EOS token
- scorer: the score function used in beam search
- max_length: the maximum search length

For beam search to work, we need a scorer function.

#### The scorer function

In this tutorial, we will use the `BeamSearchScorer`
as the scorer function, which implements the scoring function with length penalty in the
[Google NMT](https://arxiv.org/pdf/1609.08144.pdf) paper:

```{.python .input}
scorer = nlp.model.BeamSearchScorer(alpha=0, K=5, from_logits=False)
```

Defining the scorer is as simple as this one line.

#### The decoder function

Next, we define the decoder based on the pre-trained
language model.

```{.python .input}
class LMDecoder(object):
    def __init__(self, model):
        self._model = model
    def __call__(self, inputs, states):
        outputs, states = self._model(mx.nd.expand_dims(inputs, axis=0), states)
        return outputs[0], states
    def state_info(self, *arg, **kwargs):
        return self._model.state_info(*arg, **kwargs)
decoder = LMDecoder(lm_model)
```

#### Beam Search Sampler

Given a scorer and a decoder, we are ready to create a sampler. We use the symbol `.`
to indicate the end of sentence (EOS). We can use vocab to get the index of the
EOS to then feed the index to the sampler. The following code shows how to
construct a beam search sampler. We will create a sampler with 4 beams and a
maximum sample length of 20.

```{.python .input}
eos_id = vocab['.']
beam_sampler = nlp.model.BeamSearchSampler(beam_size=5,
                                           decoder=decoder,
                                           eos_id=eos_id,
                                           scorer=scorer,
                                           max_length=20)
```

It's really that simple!

#### Generate Sequences with Beam Search

Next, we are going to generate sentences starting with "I love it" using beam
search first. We feed ['I', 'Love'] to the language model to get the initial
states and set the initial input to be the word 'it'. We will then print the
top-3 generations.

```{.python .input}
bos = 'I love it'.split()
bos_ids = [vocab[ele] for ele in bos]
begin_states = lm_model.begin_state(batch_size=1, ctx=ctx)
if len(bos_ids) > 1:
    _, begin_states = lm_model(mx.nd.expand_dims(mx.nd.array(bos_ids[:-1]), axis=1),
                               begin_states)
inputs = mx.nd.full(shape=(1,), ctx=ctx, val=bos_ids[-1])
```

Here we define the helper function to generate the sequences so we can simply use one line
to generate new sequences for any given input.

```{.python .input}
def generate_sequences(sampler, inputs, begin_states, num_print_outcomes):

    samples, scores, valid_lengths = sampler(inputs, begin_states)
    samples = samples[0].asnumpy()
    scores = scores[0].asnumpy()
    valid_lengths = valid_lengths[0].asnumpy()
    print('Generation Result:')

    for i in range(num_print_outcomes):
        sentence = bos[:-1]

        for ele in samples[i][:valid_lengths[i]]:
            sentence.append(vocab.idx_to_token[ele])

        print([' '.join(sentence), scores[i]])
```

And then below, we have the one-liner to generate the sequences.

```{.python .input}
generate_sequences(beam_sampler, inputs, begin_states, 5)
```

## Sampling a Sequence with `SequenceSampler`

The previous generation results
may look a bit boring. Instead, let's now use the sequence sampler to get relatively more
interesting results.

A `SequenceSampler` samples from the contextual multinomial distribution
produced by the language model at each time step. Since we may want to control
how "sharp" the distribution is to tradeoff diversity with correctness, we can
use the temperature option in `SequenceSampler`, which controls the temperature
of the softmax activation function.

For each input, sequence sampler can sample
multiple **independent** sequences at once. The number of independent sequences
to sample can be specified through the argument `beam_size`.

Defining the `SequenceSampler` is as simple as this:

```{.python .input}
seq_sampler = nlp.model.SequenceSampler(beam_size=5,
                                        decoder=decoder,
                                        eos_id=eos_id,
                                        max_length=100,
                                        temperature=0.97)
```


#### Generate Sequences with Sequence Sampler

Now, instead of using the beam sampler for our `generate_sequences` function, we can use the `SequenceSampler` instead to sample sequences based on the same inputs used previously.

```{.python .input}
generate_sequences(seq_sampler, inputs, begin_states, 5)
```

Et voila! We've generated the most likely sentences based on our given input.

#### Exercises for the keen reader

- Tweak alpha and K in BeamSearchScorer, how are the results
changed? Does it do relatively better or worse than the sequence SequenceSampler?
- Try different samples to decode and figure out which results the BeamSearchSampler does better than the SequenceSampler
