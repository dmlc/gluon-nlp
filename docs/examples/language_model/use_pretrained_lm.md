# Using Pre-trained Language Model

A statistical language model is simply a probability distribution over sequences of words or characters [1].
In this tutorial, we'll restrict our attention to word-based language models.
Given a reliable language model, we can answer questions like *which among the following strings are we more likely to encounter?*

1. 'On Monday, Mr. Lamar’s “DAMN.” took home an even more elusive honor,
one that may never have even seemed within reach: the Pulitzer Prize"
1. "Frog zealot flagged xylophone the bean wallaby anaphylaxis extraneous
porpoise into deleterious carrot banana apricot."

Even if we've never seen either of these sentences in our entire lives, and even though no rapper has previously been
awarded a Pulitzer Prize, we wouldn't be shocked to see the first sentence in the New York Times.
By comparison, we can all agree that the second sentence, consisting of incoherent babble, is comparatively unlikely.
A statistical language model can assign precise probabilities to each of these and other strings of words.

Given a large corpus of text, we can estimate (or, in this case, train) a language model $\hat{p}(x_1, ..., x_n)$.
And given such a model, we can sample strings $\mathbf{x} \sim \hat{p}(x_1, ..., x_n)$, generating new strings according to their estimated probability.
Among other useful applications, we can use language models to score candidate transcriptions from speech recognition models, given a preference to sentences that seem more probable (at the expense of those deemed anomalous).

These days recurrent neural networks (RNNs) are the preferred method for language models. In this notebook, we will go through an example of using GluonNLP to

(i) implement a typical LSTM language model architecture
(ii) train the language model on a corpus of real data
(iii) bring in your own dataset for training
(iv) grab off-the-shelf pre-trained state-of-the-art language models (i.e., AWD language model) using GluonNLP.

## What is a language model (LM)?

The standard approach to language modeling consists of training a model that given a trailing window of text, predicts the next word in the sequence.
When we train the model we feed in the inputs $x_1, x_2, ...$ and try at each time step to predict the corresponding next word $x_2, ..., x_{n+1}$.
To generate text from a language model, we can iteratively predict the next word, and then feed this word as an input to the model at the subsequent time step. The image included below demonstrates this idea.

<img src="https://gluon.mxnet.io/_images/recurrent-lm.png" style="width: 500px;"/>

## Using a pre-trained AWD LSTM language model

AWD LSTM language model is the state-of-the-art RNN language model [1]. The main technique leveraged is to add weight-dropout on the recurrent hidden to hidden matrices to prevent overfitting on the recurrent connections.

### Load the vocabulary and the pre-trained model

```{.python .input}
import warnings
import math
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp

warnings.filterwarnings('ignore')
nlp.utils.check_version('0.7.0')

num_gpus = 1
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus else [mx.cpu()]
log_interval = 200

batch_size = 20 * len(context)
lr = 20
epochs = 3
bptt = 35
grad_clip = 0.25

dataset_name = 'wikitext-2'

# Load the dataset
train_dataset, val_dataset, test_dataset = [
    nlp.data.WikiText2(
        segment=segment, bos=None, eos='<eos>', skip_empty=False)
    for segment in ['train', 'val', 'test']
]

vocab = nlp.Vocab(
    nlp.data.Counter(train_dataset), padding_token=None, bos_token=None)


# Batchify for BPTT
bptt_batchify = nlp.data.batchify.CorpusBPTTBatchify(
    vocab, bptt, batch_size, last_batch='discard')
train_data, val_data, test_data = [
    bptt_batchify(x) for x in [train_dataset, val_dataset, test_dataset]
]

awd_model_name = 'awd_lstm_lm_1150'
awd_model, vocab = nlp.model.get_model(
    awd_model_name,
    vocab=vocab,
    dataset_name=dataset_name,
    pretrained=True,
    ctx=context[0])

print(awd_model)
print(vocab)
```

### Evaluate the pre-trained model on the validation and test datasets

```{.python .input}
# Specify the loss function, in this case, cross-entropy with softmax.
loss = gluon.loss.SoftmaxCrossEntropyLoss()


def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [detach(i) for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden


# Note that ctx is short for context
def evaluate(model, data_source, batch_size, ctx):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(
        batch_size=batch_size, func=mx.nd.zeros, ctx=ctx)
    for i, (data, target) in enumerate(data_source):
        data = data.as_in_context(ctx)
        target = target.as_in_context(ctx)
        output, hidden = model(data, hidden)
        hidden = detach(hidden)
        L = loss(output.reshape(-3, -1), target.reshape(-1))
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal


val_L = evaluate(awd_model, val_data, batch_size, context[0])
test_L = evaluate(awd_model, test_data, batch_size, context[0])

print('Best validation loss %.2f, val ppl %.2f' % (val_L, math.exp(val_L)))
print('Best test loss %.2f, test ppl %.2f' % (test_L, math.exp(test_L)))
```

## Using a cache LSTM LM

Cache LSTM language model [2] adds a cache-like memory to neural network language models. It can be used in conjunction with the aforementioned AWD LSTM language model or other LSTM models.
It exploits the hidden outputs to define a probability distribution over the words in the cache.
It generates  state-of-the-art results at inference time.

<img src=cache_model.png width="500">

### Load the pre-trained model and define the hyperparameters

```{.python .input}
window = 2
theta = 0.662
lambdas = 0.1279
bptt = 2000
cache_model = nlp.model.train.get_cache_model(name=awd_model_name,
                                             dataset_name=dataset_name,
                                             window=window,
                                             theta=theta,
                                             lambdas=lambdas,
                                             ctx=context[0])

print(cache_model)
```

### Define specific get_batch and evaluation helper functions for the cache model

Note that these helper functions are very similar to the ones we defined above, but are slightly different.

```{.python .input}
val_test_batch_size = 1
val_test_batchify = nlp.data.batchify.CorpusBatchify(vocab, val_test_batch_size)
val_data = val_test_batchify(val_dataset)
test_data = val_test_batchify(test_dataset)
```

```{.python .input}
def get_batch(data_source, i, seq_len=None):
    seq_len = min(seq_len if seq_len else bptt, len(data_source) - 1 - i)
    data = data_source[i:i + seq_len]
    target = data_source[i + 1:i + 1 + seq_len]
    return data, target
```

```{.python .input}
def evaluate_cache(model, data_source, batch_size, ctx):
    total_L = 0.0
    hidden = model.begin_state(
        batch_size=batch_size, func=mx.nd.zeros, ctx=ctx)
    next_word_history = None
    cache_history = None
    for i in range(0, len(data_source) - 1, bptt):
        if i > 0:
            print('Batch %d, ppl %f' % (i, math.exp(total_L / i)))
        if i == bptt:
            return total_L / i
        data, target = get_batch(data_source, i)
        data = data.as_in_context(ctx)
        target = target.as_in_context(ctx)
        L = 0
        outs, next_word_history, cache_history, hidden = model(
            data, target, next_word_history, cache_history, hidden)
        for out in outs:
            L += (-mx.nd.log(out)).asscalar()
        total_L += L / data.shape[1]
        hidden = detach(hidden)
    return total_L / len(data_source)
```

### Evaluate the pre-trained model on the validation and test datasets

```{.python .input}
val_L = evaluate_cache(cache_model, val_data, val_test_batch_size, context[0])
test_L = evaluate_cache(cache_model, test_data, val_test_batch_size, context[0])

print('Best validation loss %.2f, val ppl %.2f'%(val_L, math.exp(val_L)))
print('Best test loss %.2f, test ppl %.2f'%(test_L, math.exp(test_L)))
```


## References

[1] Merity, S., et al. “Regularizing and optimizing LSTM language models”. ICLR 2018

[2] Grave, E., et al. “Improving neural language models with a continuous cache”. ICLR 2017
