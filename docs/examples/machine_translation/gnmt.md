# Google Neural Machine Translation (GNMT)

In this notebook, we are going to train Google NMT on IWSLT 2015 English-Vietnamese
Dataset. The building process includes four key steps:

1. Load and preprocess the dataset

2. Create a sampler and `DataLoader`

3. Build the actual model

4. Write the training algorithm

This tutorial will guide you through each of the steps and explain briefly how each works. Please remember to click the download button at the top of the page to download the necessary files to follow this tutorial.

## Setup

Firstly, we need to setup the environment and import the necessary modules. For this tutorial, a GPU is highly important.

```{.python .input}
import warnings
warnings.filterwarnings('ignore')

import argparse
import time
import random
import os
import io
import logging
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
import nmt
nlp.utils.check_version('0.7.0')
```

Next, we need to specify the hyperparameters for the dataset, the model, and for training and testing time.

```{.python .input}
np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
ctx = mx.gpu(0)

# parameters for dataset
dataset = 'IWSLT2015'
src_lang, tgt_lang = 'en', 'vi'
src_max_len, tgt_max_len = 50, 50

# parameters for model
num_hidden = 512
num_layers = 2
num_bi_layers = 1
dropout = 0.2

# parameters for training
batch_size, test_batch_size = 128, 32
num_buckets = 5
epochs = 1
clip = 5
lr = 0.001
lr_update_factor = 0.5
log_interval = 10
save_dir = 'gnmt_en_vi_u512'

#parameters for testing
beam_size = 10
lp_alpha = 1.0
lp_k = 5

nmt.utils.logging_config(save_dir)
```

## Loading and processing the dataset

The following shows how to process the dataset and cache the processed dataset
for future use. The processing steps include the following:

1. Clipping the source and target sequences
2. Splitting the string input to a list of tokens
3. Mapping the string token onto its integer index in the vocabulary
4. Appending the end-of-sentence (EOS) token to source sentence and adding BOS and EOS tokens to the target sentence


Firstly, we load and cache the dataset with the two helper functions `cache_dataset` and `load_cached_dataset`. The functions are straightforward and well commented so no further explanation will be given.

```{.python .input}
def cache_dataset(dataset, prefix):
    """Cache the processed npy dataset  the dataset into an npz file

    Parameters
    ----------
    dataset : gluon.data.SimpleDataset
    file_path : str
    """
    if not os.path.exists(nmt._constants.CACHE_PATH):
        os.makedirs(nmt._constants.CACHE_PATH)
    src_data = np.concatenate([e[0] for e in dataset])
    tgt_data = np.concatenate([e[1] for e in dataset])
    src_cumlen = np.cumsum([0]+[len(e[0]) for e in dataset])
    tgt_cumlen = np.cumsum([0]+[len(e[1]) for e in dataset])
    np.savez(os.path.join(nmt._constants.CACHE_PATH, prefix + '.npz'),
             src_data=src_data, tgt_data=tgt_data,
             src_cumlen=src_cumlen, tgt_cumlen=tgt_cumlen)


def load_cached_dataset(prefix):
    cached_file_path = os.path.join(nmt._constants.CACHE_PATH, prefix + '.npz')
    if os.path.exists(cached_file_path):
        print('Load cached data from {}'.format(cached_file_path))
        npz_data = np.load(cached_file_path)
        src_data, tgt_data, src_cumlen, tgt_cumlen = [npz_data[n] for n in
                ['src_data', 'tgt_data', 'src_cumlen', 'tgt_cumlen']]
        src_data = np.array([src_data[low:high] for low, high in zip(src_cumlen[:-1], src_cumlen[1:])])
        tgt_data = np.array([tgt_data[low:high] for low, high in zip(tgt_cumlen[:-1], tgt_cumlen[1:])])
        return gluon.data.ArrayDataset(np.array(src_data), np.array(tgt_data))
    else:
        return None

```

Next, we write the class `TrainValDataTransform` to have easy access to transforming and clipping the source and target sentences. This class also adds the EOS and BOS tokens for cleaner data. Please refer to the comments in the code for more details.

```{.python .input}
class TrainValDataTransform(object):
    """Transform the machine translation dataset.

    Clip source and the target sentences to the maximum length. For the source sentence, append the
    EOS. For the target sentence, append BOS and EOS.

    Parameters
    ----------
    src_vocab : Vocab
    tgt_vocab : Vocab
    src_max_len : int
    tgt_max_len : int
    """

    def __init__(self, src_vocab, tgt_vocab, src_max_len, tgt_max_len):
        # On initialization of the class, we set the class variables
        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab
        self._src_max_len = src_max_len
        self._tgt_max_len = tgt_max_len

    def __call__(self, src, tgt):
        # On actual calling of the class, we perform the clipping then the appending of the EOS and BOS tokens.
        if self._src_max_len > 0:
            src_sentence = self._src_vocab[src.split()[:self._src_max_len]]
        else:
            src_sentence = self._src_vocab[src.split()]
        if self._tgt_max_len > 0:
            tgt_sentence = self._tgt_vocab[tgt.split()[:self._tgt_max_len]]
        else:
            tgt_sentence = self._tgt_vocab[tgt.split()]
        src_sentence.append(self._src_vocab[self._src_vocab.eos_token])
        tgt_sentence.insert(0, self._tgt_vocab[self._tgt_vocab.bos_token])
        tgt_sentence.append(self._tgt_vocab[self._tgt_vocab.eos_token])
        src_npy = np.array(src_sentence, dtype=np.int32)
        tgt_npy = np.array(tgt_sentence, dtype=np.int32)
        return src_npy, tgt_npy
```

We leverage the class written above to create a helper function that processes the dataset in very few lines of code.

```{.python .input}
def process_dataset(dataset, src_vocab, tgt_vocab, src_max_len=-1, tgt_max_len=-1):
    start = time.time()
    dataset_processed = dataset.transform(TrainValDataTransform(src_vocab, tgt_vocab,
                                                                src_max_len,
                                                                tgt_max_len), lazy=False)
    end = time.time()
    print('Processing time spent: {}'.format(end - start))
    return dataset_processed
```

Here we define a function `load_translation_data` that combines all the above steps to load the data, check if it's been processed, and if not, process the data. The method returns all of the required data for training, validating, and testing our model. Please refer to the comments in the code for more information on what each piece does.

```{.python .input}
def load_translation_data(dataset, src_lang='en', tgt_lang='vi'):
    """Load translation dataset

    Parameters
    ----------
    dataset : str
    src_lang : str, default 'en'
    tgt_lang : str, default 'vi'

    Returns
    -------
    data_train_processed : Dataset
        The preprocessed training sentence pairs
    data_val_processed : Dataset
        The preprocessed validation sentence pairs
    data_test_processed : Dataset
        The preprocessed test sentence pairs
    val_tgt_sentences : list
        The target sentences in the validation set
    test_tgt_sentences : list
        The target sentences in the test set
    src_vocab : Vocab
        Vocabulary of the source language
    tgt_vocab : Vocab
        Vocabulary of the target language
    """
    common_prefix = 'IWSLT2015_{}_{}_{}_{}'.format(src_lang, tgt_lang,
                                                   src_max_len, tgt_max_len)

    # Load the three datasets from files
    data_train = nlp.data.IWSLT2015('train', src_lang=src_lang, tgt_lang=tgt_lang)
    data_val = nlp.data.IWSLT2015('val', src_lang=src_lang, tgt_lang=tgt_lang)
    data_test = nlp.data.IWSLT2015('test', src_lang=src_lang, tgt_lang=tgt_lang)
    src_vocab, tgt_vocab = data_train.src_vocab, data_train.tgt_vocab
    data_train_processed = load_cached_dataset(common_prefix + '_train')

    # Check if each dataset has been processed or not, and if not, process and cache them.
    if not data_train_processed:
        data_train_processed = process_dataset(data_train, src_vocab, tgt_vocab,
                                               src_max_len, tgt_max_len)
        cache_dataset(data_train_processed, common_prefix + '_train')
    data_val_processed = load_cached_dataset(common_prefix + '_val')
    if not data_val_processed:
        data_val_processed = process_dataset(data_val, src_vocab, tgt_vocab)
        cache_dataset(data_val_processed, common_prefix + '_val')
    data_test_processed = load_cached_dataset(common_prefix + '_test')
    if not data_test_processed:
        data_test_processed = process_dataset(data_test, src_vocab, tgt_vocab)
        cache_dataset(data_test_processed, common_prefix + '_test')

    # Pull out the target sentences for both test and validation
    fetch_tgt_sentence = lambda src, tgt: tgt.split()
    val_tgt_sentences = list(data_val.transform(fetch_tgt_sentence))
    test_tgt_sentences = list(data_test.transform(fetch_tgt_sentence))

    # Return all of the necessary pieces we can extract from the data for training our model
    return data_train_processed, data_val_processed, data_test_processed, \
           val_tgt_sentences, test_tgt_sentences, src_vocab, tgt_vocab
```

We define a last helper function `get_data_lengths` to get the length of the datasets, again, for simplified cleaner code later.
```{.python .input}
def get_data_lengths(dataset):
    return list(dataset.transform(lambda srg, tgt: (len(srg), len(tgt))))

```

And for the last step of processing, we leverage all of our helper functions to keep the code concise and to these 15-20 lines for use in our main. This does all of the aforementioned processing along with storing the necessary information in memory for training our model.

```{.python .input}
data_train, data_val, data_test, val_tgt_sentences, test_tgt_sentences, src_vocab, tgt_vocab\
    = load_translation_data(dataset=dataset, src_lang=src_lang, tgt_lang=tgt_lang)
data_train_lengths = get_data_lengths(data_train)
data_val_lengths = get_data_lengths(data_val)
data_test_lengths = get_data_lengths(data_test)

with io.open(os.path.join(save_dir, 'val_gt.txt'), 'w', encoding='utf-8') as of:
    for ele in val_tgt_sentences:
        of.write(' '.join(ele) + '\n')

with io.open(os.path.join(save_dir, 'test_gt.txt'), 'w', encoding='utf-8') as of:
    for ele in test_tgt_sentences:
        of.write(' '.join(ele) + '\n')


data_train = data_train.transform(lambda src, tgt: (src, tgt, len(src), len(tgt)), lazy=False)
data_val = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                                     for i, ele in enumerate(data_val)])
data_test = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                                      for i, ele in enumerate(data_test)])
```

## Sampler and `DataLoader` construction

Now, we have obtained and stored all of the relevant data information. The next step
is to construct the sampler and `DataLoader`. The first step is to use the `batchify`
function, which pads and stacks sequences to form mini-batches.

```{.python .input}
train_batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Pad(pad_val=0),
                                            nlp.data.batchify.Pad(pad_val=0),
                                            nlp.data.batchify.Stack(dtype='float32'),
                                            nlp.data.batchify.Stack(dtype='float32'))
test_batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Pad(pad_val=0),
                                           nlp.data.batchify.Pad(pad_val=0),
                                           nlp.data.batchify.Stack(dtype='float32'),
                                           nlp.data.batchify.Stack(dtype='float32'),
                                           nlp.data.batchify.Stack())
```

We can then construct bucketing samplers, which generate batches by grouping
sequences with similar lengths. Here, the bucketing scheme is empirically determined.

```{.python .input}
bucket_scheme = nlp.data.ExpWidthBucket(bucket_len_step=1.2)
train_batch_sampler = nlp.data.FixedBucketSampler(lengths=data_train_lengths,
                                                  batch_size=batch_size,
                                                  num_buckets=num_buckets,
                                                  shuffle=True,
                                                  bucket_scheme=bucket_scheme)
logging.info('Train Batch Sampler:\n{}'.format(train_batch_sampler.stats()))
val_batch_sampler = nlp.data.FixedBucketSampler(lengths=data_val_lengths,
                                                batch_size=test_batch_size,
                                                num_buckets=num_buckets,
                                                shuffle=False)
logging.info('Valid Batch Sampler:\n{}'.format(val_batch_sampler.stats()))
test_batch_sampler = nlp.data.FixedBucketSampler(lengths=data_test_lengths,
                                                 batch_size=test_batch_size,
                                                 num_buckets=num_buckets,
                                                 shuffle=False)
logging.info('Test Batch Sampler:\n{}'.format(test_batch_sampler.stats()))
```

Given the samplers, we can create a `DataLoader`, which is iterable. This simply is a data construct (an iterator) that can feed the model batches at a time. For more information refer to [this](https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/datasets.html) page.

```{.python .input}
train_data_loader = gluon.data.DataLoader(data_train,
                                          batch_sampler=train_batch_sampler,
                                          batchify_fn=train_batchify_fn,
                                          num_workers=4)
val_data_loader = gluon.data.DataLoader(data_val,
                                        batch_sampler=val_batch_sampler,
                                        batchify_fn=test_batchify_fn,
                                        num_workers=4)
test_data_loader = gluon.data.DataLoader(data_test,
                                         batch_sampler=test_batch_sampler,
                                         batchify_fn=test_batchify_fn,
                                         num_workers=4)
```

## Building the GNMT model

After obtaining the DataLoader, we can finally build the model. The GNMT encoder and decoder
can be easily constructed by calling `get_gnmt_encoder_decoder` function. Then, we
feed the encoder and decoder to the `NMTModel` to construct the GNMT model.

`model.hybridize` allows computation to be done using the symbolic backend. To understand what it means to be "hybridized," please refer to [this](https://mxnet.incubator.apache.org/versions/master/tutorials/gluon/hybrid.html) page on MXNet hybridization and its advantages.

```{.python .input}
encoder, decoder, one_step_ahead_decoder = nmt.gnmt.get_gnmt_encoder_decoder(
    hidden_size=num_hidden, dropout=dropout, num_layers=num_layers,
    num_bi_layers=num_bi_layers)
model = nlp.model.translation.NMTModel(src_vocab=src_vocab, tgt_vocab=tgt_vocab, encoder=encoder,
                                       decoder=decoder, one_step_ahead_decoder=one_step_ahead_decoder,
                                       embed_size=num_hidden, prefix='gnmt_')
model.initialize(init=mx.init.Uniform(0.1), ctx=ctx)
static_alloc = True
model.hybridize(static_alloc=static_alloc)
logging.info(model)

# Due to the paddings, we need to mask out the losses corresponding to padding tokens.
loss_function = nlp.loss.MaskedSoftmaxCELoss()
loss_function.hybridize(static_alloc=static_alloc)
```

Here, we build the `BeamSearchTranslator` and define a predetermined `BeamSearchScorer` as the heuristical mechanism for the search. For more information on Beam Search and its applications to NLP, check [here](https://en.wikipedia.org/wiki/Beam_search).

```{.python .input}
translator = nmt.translation.BeamSearchTranslator(model=model, beam_size=beam_size,
                                                  scorer=nlp.model.BeamSearchScorer(alpha=lp_alpha,
                                                                                    K=lp_k),
                                                  max_length=tgt_max_len + 100)
logging.info('Use beam_size={}, alpha={}, K={}'.format(beam_size, lp_alpha, lp_k))
```

We define the evaluation function as shown in the code block below. The `evaluate` function uses the beam
search translator to generate outputs for the validation and testing datasets. Please refer to the comments in the code for more information on what each piece does. In addition, we add the `write_sentences` helper method to easily output the sentences.

```{.python .input}
def evaluate(data_loader):
    """Evaluate given the data loader

    Parameters
    ----------
    data_loader : gluon.data.DataLoader

    Returns
    -------
    avg_loss : float
        Average loss
    real_translation_out : list of list of str
        The translation output
    """
    translation_out = []
    all_inst_ids = []
    avg_loss_denom = 0
    avg_loss = 0.0

    for _, (src_seq, tgt_seq, src_valid_length, tgt_valid_length, inst_ids) \
            in enumerate(data_loader):
        src_seq = src_seq.as_in_context(ctx)
        tgt_seq = tgt_seq.as_in_context(ctx)
        src_valid_length = src_valid_length.as_in_context(ctx)
        tgt_valid_length = tgt_valid_length.as_in_context(ctx)

        # Calculate Loss
        out, _ = model(src_seq, tgt_seq[:, :-1], src_valid_length, tgt_valid_length - 1)
        loss = loss_function(out, tgt_seq[:, 1:], tgt_valid_length - 1).mean().asscalar()
        all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())
        avg_loss += loss * (tgt_seq.shape[1] - 1)
        avg_loss_denom += (tgt_seq.shape[1] - 1)

        # Translate the sequences and score them
        samples, _, sample_valid_length =\
            translator.translate(src_seq=src_seq, src_valid_length=src_valid_length)
        max_score_sample = samples[:, 0, :].asnumpy()
        sample_valid_length = sample_valid_length[:, 0].asnumpy()

        # Iterate through the tokens and stitch the tokens together for the sentence
        for i in range(max_score_sample.shape[0]):
            translation_out.append(
                [tgt_vocab.idx_to_token[ele] for ele in
                 max_score_sample[i][1:(sample_valid_length[i] - 1)]])

    # Calculate the average loss and initialize a None-filled translation list
    avg_loss = avg_loss / avg_loss_denom
    real_translation_out = [None for _ in range(len(all_inst_ids))]

    # Combine all the words/tokens into a sentence for the final translation
    for ind, sentence in zip(all_inst_ids, translation_out):
        real_translation_out[ind] = sentence

    # Return the loss and the translation
    return avg_loss, real_translation_out


def write_sentences(sentences, file_path):
    with io.open(file_path, 'w', encoding='utf-8') as of:
        for sent in sentences:
            of.write(' '.join(sent) + '\n')
```

## Training

Before entering the training stage, we need to create a trainer for updating the
parameters based on the loss. In the following example, we create a trainer that uses the ADAM
optimizer.

```{.python .input}
trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr})
```

We can then write the training loop. During the training, we evaluate on the validation and testing datasets every epoch, and record the
parameters that give the highest [Bilingual Evaluation Understudy Score (BLEU)](https://www.aclweb.org/anthology/P02-1040.pdf) score on the validation dataset. Before
performing forward and backward computation, we first use the `as_in_context` function to copy
the mini-batch to the GPU. The statement `with mx.autograd.record()` tells Gluon's
backend to compute the gradients for the part inside the block.

```{.python .input}
best_valid_bleu = 0.0

# Run through each epoch
for epoch_id in range(epochs):
    log_avg_loss = 0
    log_avg_gnorm = 0
    log_wc = 0
    log_start_time = time.time()

    # Iterate through each batch
    for batch_id, (src_seq, tgt_seq, src_valid_length, tgt_valid_length)\
            in enumerate(train_data_loader):

        src_seq = src_seq.as_in_context(ctx)
        tgt_seq = tgt_seq.as_in_context(ctx)
        src_valid_length = src_valid_length.as_in_context(ctx)
        tgt_valid_length = tgt_valid_length.as_in_context(ctx)

        # Compute gradients and losses
        with mx.autograd.record():
            out, _ = model(src_seq, tgt_seq[:, :-1], src_valid_length, tgt_valid_length - 1)
            loss = loss_function(out, tgt_seq[:, 1:], tgt_valid_length - 1).mean()
            loss = loss * (tgt_seq.shape[1] - 1) / (tgt_valid_length - 1).mean()
            loss.backward()

        grads = [p.grad(ctx) for p in model.collect_params().values()]
        gnorm = gluon.utils.clip_global_norm(grads, clip)
        trainer.step(1)
        src_wc = src_valid_length.sum().asscalar()
        tgt_wc = (tgt_valid_length - 1).sum().asscalar()
        step_loss = loss.asscalar()
        log_avg_loss += step_loss
        log_avg_gnorm += gnorm
        log_wc += src_wc + tgt_wc
        if (batch_id + 1) % log_interval == 0:
            wps = log_wc / (time.time() - log_start_time)
            logging.info('[Epoch {} Batch {}/{}] loss={:.4f}, ppl={:.4f}, gnorm={:.4f}, '
                         'throughput={:.2f}K wps, wc={:.2f}K'
                         .format(epoch_id, batch_id + 1, len(train_data_loader),
                                 log_avg_loss / log_interval,
                                 np.exp(log_avg_loss / log_interval),
                                 log_avg_gnorm / log_interval,
                                 wps / 1000, log_wc / 1000))
            log_start_time = time.time()
            log_avg_loss = 0
            log_avg_gnorm = 0
            log_wc = 0

    # Evaluate the losses on validation and test datasets and find the corresponding BLEU score and log it
    valid_loss, valid_translation_out = evaluate(val_data_loader)
    valid_bleu_score, _, _, _, _ = nmt.bleu.compute_bleu([val_tgt_sentences], valid_translation_out)
    logging.info('[Epoch {}] valid Loss={:.4f}, valid ppl={:.4f}, valid bleu={:.2f}'
                 .format(epoch_id, valid_loss, np.exp(valid_loss), valid_bleu_score * 100))
    test_loss, test_translation_out = evaluate(test_data_loader)
    test_bleu_score, _, _, _, _ = nmt.bleu.compute_bleu([test_tgt_sentences], test_translation_out)
    logging.info('[Epoch {}] test Loss={:.4f}, test ppl={:.4f}, test bleu={:.2f}'
                 .format(epoch_id, test_loss, np.exp(test_loss), test_bleu_score * 100))

    # Output the sentences we predicted on the validation and test datasets             
    write_sentences(valid_translation_out,
                    os.path.join(save_dir, 'epoch{:d}_valid_out.txt').format(epoch_id))
    write_sentences(test_translation_out,
                    os.path.join(save_dir, 'epoch{:d}_test_out.txt').format(epoch_id))

    # Save the model if the BLEU score is better than the previous best
    if valid_bleu_score > best_valid_bleu:
        best_valid_bleu = valid_bleu_score
        save_path = os.path.join(save_dir, 'valid_best.params')
        logging.info('Save best parameters to {}'.format(save_path))
        model.save_parameters(save_path)

    # Update the learning rate based on the number of epochs that have passed
    if epoch_id + 1 >= (epochs * 2) // 3:
        new_lr = trainer.learning_rate * lr_update_factor
        logging.info('Learning rate change to {}'.format(new_lr))
        trainer.set_learning_rate(new_lr)
```

## Conclusion
In this notebook, we have shown how to train a GNMT model on the IWSLT 2015 English-Vietnamese dataset using the Gluon NLP toolkit.
The complete training script can be found [here](https://github.com/dmlc/gluon-nlp/blob/master/scripts/machine_translation/train_gnmt.py).
The code sequence to reproduce the results can be seen on the [machine translation page](http://gluon-nlp.mxnet.io/model_zoo/machine_translation/index.html).
