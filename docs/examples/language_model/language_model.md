# LSTM-based Language Models

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
A statistical language model can assign precise probabilities to each string of words.

Given a large corpus of text, we can estimate (i.e. train) a language model $\hat{p}(x_1, ..., x_n)$.
And given such a model, we can sample strings $\mathbf{x} \sim \hat{p}(x_1, ..., x_n)$, generating new strings according to their estimated probability.
Among other useful applications, we can use language models to score candidate transcriptions from speech recognition models, given a preference to sentences that seem more probable (at the expense of those deemed anomalous).

These days recurrent neural networks (RNNs) are the preferred method for LM. In this notebook, we will go through an example of using GluonNLP to

(i) implement a typical LSTM language model architecture
(ii) train the language model on a corpus of real data
(iii) bring in your own dataset for training
(iv) grab off-the-shelf pre-trained state-of-the-art language models (i.e., AWD language model) using GluonNLP.

## Language model definition - one sentence

The standard approach to language modeling consists of training a model that given a trailing window of text, predicts the next word in the sequence.
When we train the model we feed in the inputs $x_1, x_2, ...$ and try at each time step to predict the corresponding next word $x_2, ..., x_{n+1}$.
To generate text from a language model, we can iteratively predict the next word, and then feed this word as an input to the model at the subsequent time step.

<img src="https://gluon.mxnet.io/_images/recurrent-lm.png" style="width: 500px;"/>

## Train your own language model

Now let's step through how to train your own
language model using GluonNLP.


### Preparation

We'll start by taking care of
our basic dependencies and setting up our environment

#### Load gluonnlp

```{.python .input}
import warnings
warnings.filterwarnings('ignore')

import glob
import time
import math

import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon.utils import download

import gluonnlp as nlp
```

#### Set environment

```{.python .input}
num_gpus = 1
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus else [mx.cpu()]
log_interval = 200
```

#### Set hyperparameters

```{.python .input}
batch_size = 20 * len(context)
lr = 20
epochs = 3
bptt = 35
grad_clip = 0.25
```

#### Load dataset, extract vocabulary, numericalize, and batchify for truncated Back Propagation Through Time (BPTT)

```{.python .input}
dataset_name = 'wikitext-2'
train_dataset, val_dataset, test_dataset = [
    nlp.data.WikiText2(
        segment=segment, bos=None, eos='<eos>', skip_empty=False)
    for segment in ['train', 'val', 'test']
]

vocab = nlp.Vocab(
    nlp.data.Counter(train_dataset), padding_token=None, bos_token=None)

bptt_batchify = nlp.data.batchify.CorpusBPTTBatchify(
    vocab, bptt, batch_size, last_batch='discard')
train_data, val_data, test_data = [
    bptt_batchify(x) for x in [train_dataset, val_dataset, test_dataset]
]
```

#### Load pre-defined language model architecture

```{.python .input}
model_name = 'standard_lstm_lm_200'
model, vocab = nlp.model.get_model(model_name, vocab=vocab, dataset_name=None)
print(model)
print(vocab)

model.initialize(mx.init.Xavier(), ctx=context)

trainer = gluon.Trainer(model.collect_params(), 'sgd', {
    'learning_rate': lr,
    'momentum': 0,
    'wd': 0
})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

### Training

Now that everything is ready, we can start training the model.

#### Detach gradients on states for truncated BPTT

```{.python .input}
def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [detach(i) for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden
```

#### Evaluation

```{.python .input}
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
```

#### Training loop

Our loss function will be the standard cross-entropy loss function used for multiclass classification, applied at each time step to compare our predictions to the true next word in the sequence.
We can calculate gradients with respect to our parameters using truncated [back-propagation-through-time (BPTT)](https://en.wikipedia.org/wiki/Backpropagation_through_time).
In this case, we'll backpropagate for $35$ time steps, updating our weights with stochastic gradient descent with the learning rate of $20$, hyperparameters that we chose earlier in the notebook.

<img src="https://upload.wikimedia.org/wikipedia/commons/e/ee/Unfold_through_time.png" width="500">

```{.python .input}
def train(model, train_data, val_data, test_data, epochs, lr):
    best_val = float("Inf")
    start_train_time = time.time()
    parameters = model.collect_params().values()
    for epoch in range(epochs):
        total_L = 0.0
        start_epoch_time = time.time()
        start_log_interval_time = time.time()
        hiddens = [model.begin_state(batch_size//len(context), func=mx.nd.zeros, ctx=ctx)
                   for ctx in context]
        for i, (data, target) in enumerate(train_data):
            data_list = gluon.utils.split_and_load(data, context,
                                                   batch_axis=1, even_split=True)
            target_list = gluon.utils.split_and_load(target, context,
                                                     batch_axis=1, even_split=True)
            hiddens = detach(hiddens)
            L = 0
            Ls = []
            with autograd.record():
                for j, (X, y, h) in enumerate(zip(data_list, target_list, hiddens)):
                    output, h = model(X, h)
                    batch_L = loss(output.reshape(-3, -1), y.reshape(-1,))
                    L = L + batch_L.as_in_context(context[0]) / (len(context) * X.size)
                    Ls.append(batch_L / (len(context) * X.size))
                    hiddens[j] = h
            L.backward()
            grads = [p.grad(x.context) for p in parameters for x in data_list]
            gluon.utils.clip_global_norm(grads, grad_clip)

            trainer.step(1)

            total_L += sum([mx.nd.sum(l).asscalar() for l in Ls])

            if i % log_interval == 0 and i > 0:
                cur_L = total_L / log_interval
                print('[Epoch %d Batch %d/%d] loss %.2f, ppl %.2f, '
                      'throughput %.2f samples/s'%(
                    epoch, i, len(train_data), cur_L, math.exp(cur_L),
                    batch_size * log_interval / (time.time() - start_log_interval_time)))
                total_L = 0.0
                start_log_interval_time = time.time()

        mx.nd.waitall()

        print('[Epoch %d] throughput %.2f samples/s'%(
                    epoch, len(train_data)*batch_size / (time.time() - start_epoch_time)))
        val_L = evaluate(model, val_data, batch_size, context[0])
        print('[Epoch %d] time cost %.2fs, valid loss %.2f, valid ppl %.2f'%(
            epoch, time.time()-start_epoch_time, val_L, math.exp(val_L)))

        if val_L < best_val:
            best_val = val_L
            test_L = evaluate(model, test_data, batch_size, context[0])
            model.save_parameters('{}_{}-{}.params'.format(model_name, dataset_name, epoch))
            print('test loss %.2f, test ppl %.2f'%(test_L, math.exp(test_L)))
        else:
            lr = lr*0.25
            print('Learning rate now %f'%(lr))
            trainer.set_learning_rate(lr)

    print('Total training throughput %.2f samples/s'%(
                            (batch_size * len(train_data) * epochs) /
                            (time.time() - start_train_time)))
```

#### Train and evaluate

```{.python .input}
train(model, train_data, val_data, test_data, epochs, lr)
```

#### Use your own dataset

When we train a language model, we fit to the statistics of a given dataset.
While many papers focus on a few standard datasets, such as WikiText or the Penn Tree Bank, that's just to provide a standard benchmark for the purpose of comparing models against each other.
In general, for any given use case, you'll want to train your own language model using a dataset of your own choice.
Here, for demonstration, we'll grab some `.txt` files corresponding to Sherlock Holmes novels.

```{.python .input}
TRAIN_PATH = "./sherlockholmes.train.txt"
VALID_PATH = "./sherlockholmes.valid.txt"
TEST_PATH = "./sherlockholmes.test.txt"
PREDICT_PATH = "./tinyshakespeare/input.txt"
download(
    "https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/sherlockholmes/sherlockholmes.train.txt",
    TRAIN_PATH,
    sha1_hash="d65a52baaf32df613d4942e0254c81cff37da5e8")
download(
    "https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/sherlockholmes/sherlockholmes.valid.txt",
    VALID_PATH,
    sha1_hash="71133db736a0ff6d5f024bb64b4a0672b31fc6b3")
download(
    "https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/sherlockholmes/sherlockholmes.test.txt",
    TEST_PATH,
    sha1_hash="b7ccc4778fd3296c515a3c21ed79e9c2ee249f70")
download(
    "https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/tinyshakespeare/input.txt",
    PREDICT_PATH,
    sha1_hash="04486597058d11dcc2c556b1d0433891eb639d2e")
sherlockholmes_dataset = glob.glob("sherlockholmes.*.txt")
print(sherlockholmes_dataset)
```

```{.python .input}
import nltk
moses_tokenizer = nlp.data.SacreMosesTokenizer()

sherlockholmes_val = nlp.data.CorpusDataset(
    'sherlockholmes.valid.txt',
    sample_splitter=nltk.tokenize.sent_tokenize,
    tokenizer=moses_tokenizer,
    flatten=True,
    eos='<eos>')

sherlockholmes_val_data = bptt_batchify(sherlockholmes_val)
```

```{.python .input}
sherlockholmes_L = evaluate(model, sherlockholmes_val_data, batch_size,
                            context[0])
print('Best validation loss %.2f, test ppl %.2f' %
      (sherlockholmes_L, math.exp(sherlockholmes_L)))
```

```{.python .input}
train(
    model,
    sherlockholmes_val_data,
    sherlockholmes_val_data,
    sherlockholmes_val_data,
    epochs=3,
    lr=20)
```

## Use pre-trained AWD LSTM language model

AWD LSTM language model is the state-of-the-art RNN language model [1]. The main technique is to add weight-dropout on the recurrent hidden to hidden matrices to prevent overfitting on the recurrent connections.

### Load vocabulary and pre-trained model

```{.python .input}
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

### Evaluate the pre-trained model on val and test datasets

```{.python .input}
val_L = evaluate(awd_model, val_data, batch_size, context[0])
test_L = evaluate(awd_model, test_data, batch_size, context[0])
print('Best validation loss %.2f, val ppl %.2f' % (val_L, math.exp(val_L)))
print('Best test loss %.2f, test ppl %.2f' % (test_L, math.exp(test_L)))
```

## Use Cache LSTM language model

Cache LSTM language model [2] adds a cache-like memory to neural network language models. E.g. AWD LSTM language model.
It exploits the hidden outputs to define a probability distribution over the words in the cache.
It generates the state-of-the-art results in inference time.

<img src=cache_model.png width="500">

### Load pre-trained model and define hyperparameters

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

### Define specific get_batch and evaluation for cache model

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

### Evaluate the pre-trained model on val and test datasets

```{.python .input}
val_L = evaluate_cache(cache_model, val_data, val_test_batch_size, context[0])
test_L = evaluate_cache(cache_model, test_data, val_test_batch_size, context[0])
print('Best validation loss %.2f, val ppl %.2f'%(val_L, math.exp(val_L)))
print('Best test loss %.2f, test ppl %.2f'%(test_L, math.exp(test_L)))
```

```{.python .input}
val_L = evaluate_cache(cache_model, val_data, val_test_batch_size, context[0])
test_L = evaluate_cache(cache_model, test_data, val_test_batch_size, context[0])
print('Best validation loss %.2f, val ppl %.2f' % (val_L, math.exp(val_L)))
print('Best test loss %.2f, test ppl %.2f' % (test_L, math.exp(test_L)))
```

## Reference

[1] Merity, S., et al. “Regularizing and optimizing LSTM language models”. ICLR 2018

[2] Grave, E., et al. “Improving neural language models with a continuous cache”. ICLR 2017
