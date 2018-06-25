# Sentiment Analysis (SA) with pretrained Language Model (LM)

In this notebook, we are going to build a sentiment analysis model based on the pretrained language model. We are focusing on the best usability to support traditional nlp tasks in a simple fashion. The building process is simple three steps. Let us get started now.

We use movie reviews from the Large Movie Review Dataset, as known as the IMDB dataset. In this task, given a moview, the model attemps to predict its sentiment, which can be positive or negative.

## Preparation and settings

### Load mxnet and gluonnlp

```{.python .input  n=1}
import random
import time
import multiprocessing as mp
import numpy as np

import mxnet as mx
from mxnet import nd, gluon, autograd

import gluonnlp as nlp

random.seed(123)
np.random.seed(123)
mx.random.seed(123)
```

### Hyperparameters

Our model is based on a standard LSTM model. We use a hidden size of 200. We use bucketing for speeding up the processing of variable-length sequences. To enable multi-gpu training, we can simply change num_gpus to some value larger than 1.

```{.python .input  n=2}
dropout = 0
language_model_name = 'standard_lstm_lm_200'
pretrained = True
num_gpus = 1
learning_rate = 0.005 * num_gpus
batch_size = 16 * num_gpus
bucket_num = 10
bucket_ratio = 0.2
epochs = 1
grad_clip = None
log_interval = 100
```

```{.python .input  n=3}
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus else [mx.cpu()]
```

## Sentiment analysis model with pre-trained language model encoder

The model architecture is based on pretrained LM:

![sa-model](samodel-v3.png)

Our model is composed of a two-layer LSTM followed by an average pooling and a sigmoid output layer as illustrated in the Figure above. From the embedding layer, the new representations will be passed to LSTM cells. These will include information about the sequence of words in the data. Thus, given an input sequence, the memory cells in the LSTM layer will produce a representation sequence. This representation sequence is then averaged over all timesteps resulting in representation h. Finally, this representation is fed to a sigmoid output layer. We’re using the sigmoid  because we’re trying to predict if this text has positive or negative sentiment, and a sigmoid activation function allows the model to compute the posterior probability.

```{.python .input  n=4}
class SentimentNet(gluon.Block):
    def __init__(self, embedding_block, encoder_block, dropout,
                 prefix=None, params=None):
        super(SentimentNet, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.embedding = embedding_block
            self.encoder = encoder_block
            self.out_layer = gluon.nn.HybridSequential()
            with self.out_layer.name_scope():
                self.out_layer.add(gluon.nn.Dropout(dropout))
                self.out_layer.add(gluon.nn.Dense(1, flatten=False))

    def forward(self, data, valid_length):
        encoded = self.encoder(nd.Dropout(self.embedding(data),
                                          0.2, axes=(0,)))  # Shape(T, N, C)
        # Zero out the values with position exceeding the valid length.
        masked_encoded = nd.SequenceMask(encoded,
                                         sequence_length=valid_length,
                                         use_sequence_length=True)
        agg_state = nd.broadcast_div(nd.sum(masked_encoded, axis=0),
                                     nd.expand_dims(valid_length, axis=1))
        out = self.out_layer(agg_state)
        return out


lm_model, vocab = nlp.model.get_model(name=language_model_name,
                                      dataset_name='wikitext-2',
                                      pretrained=pretrained,
                                      ctx=context,
                                      dropout=dropout)
net = SentimentNet(embedding_block=lm_model.embedding,
                   encoder_block=lm_model.encoder,
                   dropout=dropout)
net.out_layer.initialize(mx.init.Xavier(), ctx=context)
net.hybridize()
print(net)
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "SentimentNet(\n  (encoder): LSTM(200 -> 200.0, TNC, num_layers=2)\n  (embedding): HybridSequential(\n    (0): Embedding(33278 -> 200, float32)\n  )\n  (out_layer): HybridSequential(\n    (0): Dropout(p = 0, axes=())\n    (1): Dense(None -> 1, linear)\n  )\n)\n"
 }
]
```

In the above code, we first acquire a pretrained model on Wikitext-2 dataset using nlp.model.get_model. We then construct a SentimentNet object, which takes as input the embedding layer and encoder of the pretrained model.

As we employ the pretrained embedding layer and encoder, we only need to initialize the output layer using net.out_layer.initialize(mx.init.Xavier(), ctx=context).

## Data pipeline

### Load sentiment analysis dataset -- IMDB reviews

```{.python .input  n=5}
# train_dataset and test_dataset are both SimpleDataset objects,
# which is a wrapper for lists and arrays.
train_dataset, test_dataset = [nlp.data.IMDB(segment=segment)
                               for segment in ('train', 'test')]
print("Tokenize using spaCy...")
# tokenizer takes as input a string and outputs a list of tokens.
tokenizer = nlp.data.SpacyTokenizer('en')
# length_clip takes as input a list and outputs a list with maximum length 500.
length_clip = nlp.data.ClipSequence(500)

def preprocess(x):
    data, label = x
    # In the labeled train/test sets, a negative review has a score <= 4
    # out of 10, and a positive review has a score >= 7 out of 10. Thus
    # reviews with more neutral ratings are not included in the train/test
    # sets. We labeled a negative review whose score <= 4 as 0, and a
    # positive review whose score >= 7 as 1. As the neural ratings are not
    # included in the datasets, we can simply use 5 as our threshold.
    label = int(label > 5)
    # A token index or a list of token indices is
    # returned according to the vocabulary.
    data = vocab[length_clip(tokenizer(data))]
    return data, label, float(len(data))

def get_length(x):
    return x[2]

def preprocess_dataset(dataset):
    start = time.time()
    with mp.Pool() as pool:
        # Each sample is processed in an asynchronous manner.
        dataset = gluon.data.SimpleDataset(pool.map(preprocess, dataset))
        lengths = gluon.data.SimpleDataset(pool.map(get_length, dataset))
    end = time.time()
    print('Done! Tokenizing Time={:.2f}s, #Sentences={}'
          .format(end - start, len(dataset)))
    return dataset, lengths

train_dataset, train_data_lengths = preprocess_dataset(train_dataset)
test_dataset, test_data_lengths = preprocess_dataset(test_dataset)
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Tokenize using spaCy...\nDone! Tokenizing Time=13.36s, #Sentences=25000\nDone! Tokenizing Time=12.95s, #Sentences=25000\n"
 }
]
```


## Training

### Evaluation using loss and accuracy

```{.python .input  n=6}
def evaluate(net, dataloader, context):
    loss = gluon.loss.SigmoidBCELoss()
    total_L = 0.0
    total_sample_num = 0
    total_correct_num = 0
    start_log_interval_time = time.time()
    print('Begin Testing...')
    for i, (data, label, valid_length) in enumerate(dataloader):
        data = mx.nd.transpose(data.as_in_context(context))
        valid_length = valid_length.as_in_context(context).astype(np.float32)
        label = label.as_in_context(context)
        output = net(data, valid_length)
        L = loss(output, label)
        pred = (output > 0.5).reshape(-1)
        total_L += L.sum().asscalar()
        total_sample_num += label.shape[0]
        total_correct_num += (pred == label).sum().asscalar()
        if (i + 1) % log_interval == 0:
            print('[Batch {}/{}] elapsed {:.2f} s'.format(
                i + 1, len(dataloader),
                time.time() - start_log_interval_time))
            start_log_interval_time = time.time()
    avg_L = total_L / float(total_sample_num)
    acc = total_correct_num / float(total_sample_num)
    return avg_L, acc
```

In the following code, we use FixedBucketSampler, which assigns each data sample to a fixed bucket based on its length. The bucket keys are either given or generated from the input sequence lengths and number of the buckets.

```{.python .input  n=7}
def train(net, context, epochs):
    trainer = gluon.Trainer(net.collect_params(),
                            'ftml',
                            {'learning_rate': learning_rate})
    loss = gluon.loss.SigmoidBCELoss()

    # Construct the DataLoader
    # Pad data, stack label and lengths
    batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Pad(axis=0),
                                          nlp.data.batchify.Stack(),
                                          nlp.data.batchify.Stack())
    batch_sampler = nlp.data.sampler.FixedBucketSampler(train_data_lengths,
                                                        batch_size=batch_size,
                                                        num_buckets=bucket_num,
                                                        ratio=bucket_ratio,
                                                        shuffle=True)
    print(batch_sampler.stats())
    train_dataloader = gluon.data.DataLoader(dataset=train_dataset,
                                             batch_sampler=batch_sampler,
                                             batchify_fn=batchify_fn)
    test_dataloader = gluon.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            batchify_fn=batchify_fn)
    parameters = net.collect_params().values()

    # Training/Testing
    for epoch in range(epochs):
        # Epoch training stats
        start_epoch_time = time.time()
        epoch_L = 0.0
        epoch_sent_num = 0
        epoch_wc = 0
        # Log interval training stats
        start_log_interval_time = time.time()
        log_interval_wc = 0
        log_interval_sent_num = 0
        log_interval_L = 0.0

        for i, (data, label, length) in enumerate(train_dataloader):
            if data.shape[0] > len(context):
                # Multi-gpu training.
                data_list, label_list, length_list \
                = [gluon.utils.split_and_load(x,
                                              context,
                                              batch_axis=0,
                                              even_split=False)
                   for x in [data, label, length]]
            else:
                data_list = [data.as_in_context(context[0])]
                label_list = [label.as_in_context(context[0])]
                length_list = [length.as_in_context(context[0])]
            L = 0
            wc = length.sum().asscalar()
            log_interval_wc += wc
            epoch_wc += wc
            log_interval_sent_num += data.shape[1]
            epoch_sent_num += data.shape[1]
            for data, label, valid_length in zip(data_list, label_list, length_list):
                valid_length = valid_length
                with autograd.record():
                    output = net(data.T, valid_length)
                    L = L + loss(output, label).mean().as_in_context(context[0])
            L.backward()
            # Clip gradient
            if grad_clip:
                gluon.utils.clip_global_norm([p.grad(x.context)
                                              for p in parameters for x in data_list],
                                             grad_clip)
            # Update parameter
            trainer.step(1)
            log_interval_L += L.asscalar()
            epoch_L += L.asscalar()
            if (i + 1) % log_interval == 0:
                print('[Epoch {} Batch {}/{}] elapsed {:.2f} s, \
                      avg loss {:.6f}, throughput {:.2f}K wps'.format(
                    epoch, i + 1, len(train_dataloader),
                    time.time() - start_log_interval_time,
                    log_interval_L / log_interval_sent_num,
                    log_interval_wc / 1000 / (time.time() - start_log_interval_time)))
                # Clear log interval training stats
                start_log_interval_time = time.time()
                log_interval_wc = 0
                log_interval_sent_num = 0
                log_interval_L = 0
        end_epoch_time = time.time()
        test_avg_L, test_acc = evaluate(net, test_dataloader, context[0])
        print('[Epoch {}] train avg loss {:.6f}, test acc {:.2f}, \
        test avg loss {:.6f}, throughput {:.2f}K wps'.format(
            epoch, epoch_L / epoch_sent_num,
            test_acc, test_avg_L, epoch_wc / 1000 /
            (end_epoch_time - start_epoch_time)))
```

```{.python .input  n=8}
train(net, context, epochs)
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "FixedBucketSampler:\n  sample_num=25000, batch_num=1548\n  key=[68, 116, 164, 212, 260, 308, 356, 404, 452, 500]\n  cnt=[981, 1958, 5686, 4614, 2813, 2000, 1411, 1129, 844, 3564]\n  batch_size=[23, 16, 16, 16, 16, 16, 16, 16, 16, 16]\n[Epoch 0 Batch 100/1548] elapsed 4.21 s,                       avg loss 0.002488, throughput 94.89K wps\n[Epoch 0 Batch 200/1548] elapsed 4.17 s,                       avg loss 0.002098, throughput 99.37K wps\n[Epoch 0 Batch 300/1548] elapsed 4.16 s,                       avg loss 0.002196, throughput 86.69K wps\n[Epoch 0 Batch 400/1548] elapsed 4.32 s,                       avg loss 0.001733, throughput 93.43K wps\n[Epoch 0 Batch 500/1548] elapsed 4.23 s,                       avg loss 0.001605, throughput 98.33K wps\n[Epoch 0 Batch 600/1548] elapsed 4.35 s,                       avg loss 0.001525, throughput 95.50K wps\n[Epoch 0 Batch 700/1548] elapsed 4.24 s,                       avg loss 0.001423, throughput 101.45K wps\n[Epoch 0 Batch 800/1548] elapsed 4.16 s,                       avg loss 0.001371, throughput 103.64K wps\n[Epoch 0 Batch 900/1548] elapsed 4.24 s,                       avg loss 0.001391, throughput 97.95K wps\n[Epoch 0 Batch 1000/1548] elapsed 4.39 s,                       avg loss 0.001463, throughput 81.96K wps\n[Epoch 0 Batch 1100/1548] elapsed 4.26 s,                       avg loss 0.001424, throughput 88.20K wps\n[Epoch 0 Batch 1200/1548] elapsed 4.10 s,                       avg loss 0.001319, throughput 94.00K wps\n[Epoch 0 Batch 1300/1548] elapsed 4.40 s,                       avg loss 0.001346, throughput 84.93K wps\n[Epoch 0 Batch 1400/1548] elapsed 4.11 s,                       avg loss 0.001259, throughput 94.36K wps\n[Epoch 0 Batch 1500/1548] elapsed 4.26 s,                       avg loss 0.001223, throughput 93.39K wps\nBegin Testing...\n[Batch 100/1563] elapsed 4.36 s\n[Batch 200/1563] elapsed 4.21 s\n[Batch 300/1563] elapsed 4.30 s\n[Batch 400/1563] elapsed 4.37 s\n[Batch 500/1563] elapsed 4.30 s\n[Batch 600/1563] elapsed 4.72 s\n[Batch 700/1563] elapsed 4.80 s\n[Batch 800/1563] elapsed 4.80 s\n[Batch 900/1563] elapsed 5.61 s\n[Batch 1000/1563] elapsed 4.23 s\n[Batch 1100/1563] elapsed 4.15 s\n[Batch 1200/1563] elapsed 4.31 s\n[Batch 1300/1563] elapsed 4.12 s\n[Batch 1400/1563] elapsed 4.35 s\n[Batch 1500/1563] elapsed 4.16 s\n[Epoch 0] train avg loss 0.001580, test acc 0.86,         test avg loss 0.314616, throughput 93.80K wps\n"
 }
]
```

```{.python .input  n=9}
net(mx.nd.reshape(mx.nd.array(vocab[['This', 'movie', 'is', 'amazing']],
                              ctx=context[0]), shape=(-1, 1)),
    mx.nd.array([4], ctx=context[0])).sigmoid()
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "\n[[0.7124313]]\n<NDArray 1x1 @gpu(0)>"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Conclusion

In summary, we have built a SA model using gluonnlp. It is:

1) easy to use.

2) simple to customize.

3) fast to build the NLP prototype.

Gluonnlp documentation is here: http://gluon-nlp.mxnet.io/index.html
