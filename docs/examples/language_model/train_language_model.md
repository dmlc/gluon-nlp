# Train your own LSTM based Language Model

Now let's go through the step-by-step process on how to train your own
language model using GluonNLP.

## Preparation

We'll start by taking care of
our basic dependencies and setting up our environment.

Firstly, we import the required modules for GluonNLP and the LM.

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
nlp.utils.check_version('0.7.0')
```

Then we setup the environment for GluonNLP.

Please note that we should change num_gpus according to how many NVIDIA GPUs are available on the target machine in the following code.

```{.python .input}
num_gpus = 1
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus else [mx.cpu()]
log_interval = 200
```

Next we setup the hyperparameters for the LM we are using.

Note that BPTT stands for "back propagation through time," and LR stands for learning rate. A link to more information on truncated BPTT can be found [here.](https://en.wikipedia.org/wiki/Backpropagation_through_time)

```{.python .input}
batch_size = 20 * len(context)
lr = 20
epochs = 3
bptt = 35
grad_clip = 0.25
```

## Loading the dataset

Now, we load the dataset, extract the vocabulary, numericalize, and batchify in order to perform truncated BPTT.

```{.python .input}
dataset_name = 'wikitext-2'

# Load the dataset
train_dataset, val_dataset, test_dataset = [
    nlp.data.WikiText2(
        segment=segment, bos=None, eos='<eos>', skip_empty=False)
    for segment in ['train', 'val', 'test']
]

# Extract the vocabulary and numericalize with "Counter"
vocab = nlp.Vocab(
    nlp.data.Counter(train_dataset), padding_token=None, bos_token=None)

# Batchify for BPTT
bptt_batchify = nlp.data.batchify.CorpusBPTTBatchify(
    vocab, bptt, batch_size, last_batch='discard')
train_data, val_data, test_data = [
    bptt_batchify(x) for x in [train_dataset, val_dataset, test_dataset]
]
```

And then we load the pre-defined language model architecture as so:

```{.python .input}
model_name = 'standard_lstm_lm_200'
model, vocab = nlp.model.get_model(model_name, vocab=vocab, dataset_name=None)
print(model)
print(vocab)

# Initialize the model
model.initialize(mx.init.Xavier(), ctx=context)

# Initialize the trainer and optimizer and specify some hyperparameters
trainer = gluon.Trainer(model.collect_params(), 'sgd', {
    'learning_rate': lr,
    'momentum': 0,
    'wd': 0
})

# Specify the loss function, in this case, cross-entropy with softmax.
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

## Training the LM

Now that everything is ready, we can start training the model.

We first define a helper function for detaching the gradients on specific states for easier truncated BPTT.

```{.python .input}
def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [detach(i) for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden
```

And then a helper evaluation function.

```{.python .input}
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
```

### The main training loop

Our loss function will be the standard cross-entropy loss function used for multi-class classification, applied at each time step to compare the model's predictions to the true next word in the sequence.
We can calculate gradients with respect to our parameters using truncated BPTT.
In this case, we'll back propagate for $35$ time steps, updating our weights with stochastic gradient descent and a learning rate of $20$; these correspond to the hyperparameters that we specified earlier in the notebook.

<img src="https://upload.wikimedia.org/wikipedia/commons/e/ee/Unfold_through_time.png" width="500">

```{.python .input}
# Function for actually training the model
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

We can now actually perform the training

```{.python .input}
train(model, train_data, val_data, test_data, epochs, lr)
```

## Using your own dataset

When we train a language model, we fit to the statistics of a given dataset.
While many papers focus on a few standard datasets, such as WikiText or the Penn Tree Bank, that's just to provide a standard benchmark for the purpose of comparing models against one another.
In general, for any given use case, you'll want to train your own language model using a dataset of your own choice.
Here, for demonstration, we'll grab some `.txt` files corresponding to Sherlock Holmes novels.

We first download the new dataset.

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

print(glob.glob("sherlockholmes.*.txt"))
```

Then we specify the tokenizer as well as batchify the dataset.

```{.python .input}
import nltk
moses_tokenizer = nlp.data.SacreMosesTokenizer()

sherlockholmes_datasets = [
    nlp.data.CorpusDataset(
        'sherlockholmes.{}.txt'.format(name),
        sample_splitter=nltk.tokenize.sent_tokenize,
        tokenizer=moses_tokenizer,
        flatten=True,
        eos='<eos>') for name in ['train', 'valid', 'test']
]

sherlockholmes_train_data, sherlockholmes_val_data, sherlockholmes_test_data = [
    bptt_batchify(dataset) for dataset in sherlockholmes_datasets
]
```

We setup the evaluation to see whether our previous model trained on the other dataset does well on the new dataset.

```{.python .input}
sherlockholmes_L = evaluate(model, sherlockholmes_val_data, batch_size,
                            context[0])
print('Best validation loss %.2f, test ppl %.2f' %
      (sherlockholmes_L, math.exp(sherlockholmes_L)))
```

Or we have the option of training the model on the new dataset with just one line of code.

```{.python .input}
train(
    model,
    sherlockholmes_train_data, # This is your input training data, we leave batchifying and tokenizing as an exercise for the reader
    sherlockholmes_val_data,
    sherlockholmes_test_data, # This would be your test data, again left as an exercise for the reader
    epochs=3,
    lr=20)
```
