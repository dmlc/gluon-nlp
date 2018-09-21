# Google NMT on IWSLT 2015 English-Vietnamese Translation

In this notebook, we are going to train Google NMT on IWSLT 2015 English-Vietnamese
Dataset. The building prcoess includes four steps: 1) load and process dataset, 2)
create sampler and DataLoader, 3) build model, and 4) write training epochs.

## Load MXNET and Gluon

```{.python .input  n=1}
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
```

## Hyper-parameters

```{.python .input  n=2}
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
epochs = 2
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

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "All Logs will be saved to gnmt_en_vi_u512/<ipython-input-2-df5a36649d80>.log\n"
 },
 {
  "data": {
   "text/plain": "'gnmt_en_vi_u512'"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Load and Preprocess Dataset

The following shows how to process the dataset and cache the processed dataset
for future use. The processing steps include: 1) clip the source and target
sequences, 2) split the string input to a list of tokens, 3) map the
string token into its integer index in the vocabulary, and 4) append end-of-sentence (EOS) token to source
sentence and add BOS and EOS tokens to target sentence.

```{.python .input  n=3}
def cache_dataset(dataset, prefix):
    """Cache the processed npy dataset  the dataset into a npz

    Parameters
    ----------
    dataset : gluon.data.SimpleDataset
    file_path : str
    """
    if not os.path.exists(nmt._constants.CACHE_PATH):
        os.makedirs(nmt._constants.CACHE_PATH)
    src_data = np.array([ele[0] for ele in dataset])
    tgt_data = np.array([ele[1] for ele in dataset])
    np.savez(os.path.join(nmt._constants.CACHE_PATH, prefix + '.npz'), src_data=src_data, tgt_data=tgt_data)


def load_cached_dataset(prefix):
    cached_file_path = os.path.join(nmt._constants.CACHE_PATH, prefix + '.npz')
    if os.path.exists(cached_file_path):
        print('Load cached data from {}'.format(cached_file_path))
        dat = np.load(cached_file_path)
        return gluon.data.ArrayDataset(np.array(dat['src_data']), np.array(dat['tgt_data']))
    else:
        return None


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
        self._src_vocab = src_vocab
        self._tgt_vocab = tgt_vocab
        self._src_max_len = src_max_len
        self._tgt_max_len = tgt_max_len

    def __call__(self, src, tgt):
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


def process_dataset(dataset, src_vocab, tgt_vocab, src_max_len=-1, tgt_max_len=-1):
    start = time.time()
    dataset_processed = dataset.transform(TrainValDataTransform(src_vocab, tgt_vocab,
                                                                src_max_len,
                                                                tgt_max_len), lazy=False)
    end = time.time()
    print('Processing time spent: {}'.format(end - start))
    return dataset_processed


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
    data_train = nlp.data.IWSLT2015('train', src_lang=src_lang, tgt_lang=tgt_lang)
    data_val = nlp.data.IWSLT2015('val', src_lang=src_lang, tgt_lang=tgt_lang)
    data_test = nlp.data.IWSLT2015('test', src_lang=src_lang, tgt_lang=tgt_lang)
    src_vocab, tgt_vocab = data_train.src_vocab, data_train.tgt_vocab
    data_train_processed = load_cached_dataset(common_prefix + '_train')
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
    fetch_tgt_sentence = lambda src, tgt: tgt.split()
    val_tgt_sentences = list(data_val.transform(fetch_tgt_sentence))
    test_tgt_sentences = list(data_test.transform(fetch_tgt_sentence))
    return data_train_processed, data_val_processed, data_test_processed, \
           val_tgt_sentences, test_tgt_sentences, src_vocab, tgt_vocab


def get_data_lengths(dataset):
    return list(dataset.transform(lambda srg, tgt: (len(srg), len(tgt))))


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

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Load cached data from /home/ubuntu/cgraywang/gluon-nlp/docs/examples/nmt/nmt/cached/IWSLT2015_en_vi_50_50_train.npz\nLoad cached data from /home/ubuntu/cgraywang/gluon-nlp/docs/examples/nmt/nmt/cached/IWSLT2015_en_vi_50_50_val.npz\nLoad cached data from /home/ubuntu/cgraywang/gluon-nlp/docs/examples/nmt/nmt/cached/IWSLT2015_en_vi_50_50_test.npz\n"
 }
]
```

## Create Sampler and DataLoader

Now, we have obtained `data_train`, `data_val`, and `data_test`. The next step
is to construct sampler and DataLoader. The first step is to construct batchify
function, which pads and stacks sequences to form mini-batch.

```{.python .input  n=4}
train_batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Pad(),
                                            nlp.data.batchify.Pad(),
                                            nlp.data.batchify.Stack(dtype='float32'),
                                            nlp.data.batchify.Stack(dtype='float32'))
test_batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Pad(),
                                           nlp.data.batchify.Pad(),
                                           nlp.data.batchify.Stack(dtype='float32'),
                                           nlp.data.batchify.Stack(dtype='float32'),
                                           nlp.data.batchify.Stack())
```

We can then construct bucketing samplers, which generate batches by grouping
sequences with similar lengths. Here, the bucketing scheme is empirically determined.

```{.python .input  n=5}
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

```{.json .output n=5}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "2018-09-21 22:05:23,029 - root - Train Batch Sampler:\nFixedBucketSampler:\n  sample_num=133166, batch_num=1043\n  key=[(9, 10), (16, 17), (26, 27), (37, 38), (51, 52)]\n  cnt=[11414, 34897, 37760, 23480, 25615]\n  batch_size=[128, 128, 128, 128, 128]\n2018-09-21 22:05:23,035 - root - Valid Batch Sampler:\nFixedBucketSampler:\n  sample_num=1553, batch_num=52\n  key=[(22, 28), (40, 52), (58, 76), (76, 100), (94, 124)]\n  cnt=[1037, 432, 67, 10, 7]\n  batch_size=[32, 32, 32, 32, 32]\n2018-09-21 22:05:23,039 - root - Test Batch Sampler:\nFixedBucketSampler:\n  sample_num=1268, batch_num=42\n  key=[(23, 29), (43, 53), (63, 77), (83, 101), (103, 125)]\n  cnt=[770, 381, 84, 26, 7]\n  batch_size=[32, 32, 32, 32, 32]\n"
 }
]
```

Given the samplers, we can create DataLoader, which is iterable.

```{.python .input  n=6}
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

## Build GNMT Model

After obtaining DataLoader, we can build the model. The GNMT encoder and decoder
can be easily constructed by calling `get_gnmt_encoder_decoder` function. Then, we
feed the encoder and decoder to `NMTModel` to construct the GNMT model.
`model.hybridize` allows computation to be done using the symbolic backend.

```{.python .input  n=7}
encoder, decoder = nmt.gnmt.get_gnmt_encoder_decoder(hidden_size=num_hidden,
                                                     dropout=dropout,
                                                     num_layers=num_layers,
                                                     num_bi_layers=num_bi_layers)
model = nmt.translation.NMTModel(src_vocab=src_vocab, tgt_vocab=tgt_vocab, encoder=encoder, decoder=decoder,
                                 embed_size=num_hidden, prefix='gnmt_')
model.initialize(init=mx.init.Uniform(0.1), ctx=ctx)
static_alloc = True
model.hybridize(static_alloc=static_alloc)
logging.info(model)

# Due to the paddings, we need to mask out the losses corresponding to padding tokens.
loss_function = nmt.loss.SoftmaxCEMaskedLoss()
loss_function.hybridize(static_alloc=static_alloc)
```

```{.json .output n=7}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "2018-09-21 22:05:32,647 - root - NMTModel(\n  (encoder): GNMTEncoder(\n    (dropout_layer): Dropout(p = 0.2, axes=())\n    (rnn_cells): HybridSequential(\n      (0): BidirectionalCell(forward=LSTMCell(None -> 2048), backward=LSTMCell(None -> 2048))\n      (1): LSTMCell(None -> 2048)\n    )\n  )\n  (decoder): GNMTDecoder(\n    (attention_cell): DotProductAttentionCell(\n      (_dropout_layer): Dropout(p = 0.0, axes=())\n      (_proj_query): Dense(None -> 512, linear)\n    )\n    (dropout_layer): Dropout(p = 0.2, axes=())\n    (rnn_cells): HybridSequential(\n      (0): LSTMCell(None -> 2048)\n      (1): LSTMCell(None -> 2048)\n    )\n  )\n  (src_embed): HybridSequential(\n    (0): Embedding(17191 -> 512, float32)\n    (1): Dropout(p = 0.0, axes=())\n  )\n  (tgt_embed): HybridSequential(\n    (0): Embedding(7709 -> 512, float32)\n    (1): Dropout(p = 0.0, axes=())\n  )\n  (tgt_proj): Dense(None -> 7709, linear)\n)\n"
 }
]
```

We also build the beam search translator.

```{.python .input  n=8}
translator = nmt.translation.BeamSearchTranslator(model=model, beam_size=beam_size,
                                                  scorer=nlp.model.BeamSearchScorer(alpha=lp_alpha,
                                                                                    K=lp_k),
                                                  max_length=tgt_max_len + 100)
logging.info('Use beam_size={}, alpha={}, K={}'.format(beam_size, lp_alpha, lp_k))
```

```{.json .output n=8}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "2018-09-21 22:05:32,673 - root - Use beam_size=10, alpha=1.0, K=5\n"
 }
]
```

We define evaluation function as follows. The `evaluate` function use beam
search translator to generate outputs for the validation and testing datasets.

```{.python .input  n=9}
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
        # Calculating Loss
        out, _ = model(src_seq, tgt_seq[:, :-1], src_valid_length, tgt_valid_length - 1)
        loss = loss_function(out, tgt_seq[:, 1:], tgt_valid_length - 1).mean().asscalar()
        all_inst_ids.extend(inst_ids.asnumpy().astype(np.int32).tolist())
        avg_loss += loss * (tgt_seq.shape[1] - 1)
        avg_loss_denom += (tgt_seq.shape[1] - 1)
        # Translate
        samples, _, sample_valid_length =\
            translator.translate(src_seq=src_seq, src_valid_length=src_valid_length)
        max_score_sample = samples[:, 0, :].asnumpy()
        sample_valid_length = sample_valid_length[:, 0].asnumpy()
        for i in range(max_score_sample.shape[0]):
            translation_out.append(
                [tgt_vocab.idx_to_token[ele] for ele in
                 max_score_sample[i][1:(sample_valid_length[i] - 1)]])
    avg_loss = avg_loss / avg_loss_denom
    real_translation_out = [None for _ in range(len(all_inst_ids))]
    for ind, sentence in zip(all_inst_ids, translation_out):
        real_translation_out[ind] = sentence
    return avg_loss, real_translation_out


def write_sentences(sentences, file_path):
    with io.open(file_path, 'w', encoding='utf-8') as of:
        for sent in sentences:
            of.write(' '.join(sent) + '\n')
```

## Training Epochs

Before entering the training stage, we need to create trainer for updating the
parameters. In the following example, we create a trainer that uses ADAM
optimzier.

```{.python .input  n=10}
trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr})
```

We can then write the training loop. During the training, we evaluate on the validation and testing datasets every epoch, and record the
parameters that give the hightest BLEU score on the validation dataset. Before
performing forward and backward, we first use `as_in_context` function to copy
the mini-batch to GPU. The statement `with mx.autograd.record()` tells Gluon
backend to compute the gradients for the part inside the block.

```{.python .input  n=11}
best_valid_bleu = 0.0
for epoch_id in range(epochs):
    log_avg_loss = 0
    log_avg_gnorm = 0
    log_wc = 0
    log_start_time = time.time()
    for batch_id, (src_seq, tgt_seq, src_valid_length, tgt_valid_length)\
            in enumerate(train_data_loader):
        # logging.info(src_seq.context) Context suddenly becomes GPU.
        src_seq = src_seq.as_in_context(ctx)
        tgt_seq = tgt_seq.as_in_context(ctx)
        src_valid_length = src_valid_length.as_in_context(ctx)
        tgt_valid_length = tgt_valid_length.as_in_context(ctx)
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
    valid_loss, valid_translation_out = evaluate(val_data_loader)
    valid_bleu_score, _, _, _, _ = nmt.bleu.compute_bleu([val_tgt_sentences], valid_translation_out)
    logging.info('[Epoch {}] valid Loss={:.4f}, valid ppl={:.4f}, valid bleu={:.2f}'
                 .format(epoch_id, valid_loss, np.exp(valid_loss), valid_bleu_score * 100))
    test_loss, test_translation_out = evaluate(test_data_loader)
    test_bleu_score, _, _, _, _ = nmt.bleu.compute_bleu([test_tgt_sentences], test_translation_out)
    logging.info('[Epoch {}] test Loss={:.4f}, test ppl={:.4f}, test bleu={:.2f}'
                 .format(epoch_id, test_loss, np.exp(test_loss), test_bleu_score * 100))
    write_sentences(valid_translation_out,
                    os.path.join(save_dir, 'epoch{:d}_valid_out.txt').format(epoch_id))
    write_sentences(test_translation_out,
                    os.path.join(save_dir, 'epoch{:d}_test_out.txt').format(epoch_id))
    if valid_bleu_score > best_valid_bleu:
        best_valid_bleu = valid_bleu_score
        save_path = os.path.join(save_dir, 'valid_best.params')
        logging.info('Save best parameters to {}'.format(save_path))
        model.save_parameters(save_path)
    if epoch_id + 1 >= (epochs * 2) // 3:
        new_lr = trainer.learning_rate * lr_update_factor
        logging.info('Learning rate change to {}'.format(new_lr))
        trainer.set_learning_rate(new_lr)
```

```{.json .output n=11}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "2018-09-21 22:05:38,346 - root - [Epoch 0 Batch 10/1043] loss=7.7375, ppl=2292.6591, gnorm=1.4907, throughput=9.66K wps, wc=54.27K\n2018-09-21 22:05:41,141 - root - [Epoch 0 Batch 20/1043] loss=6.3590, ppl=577.6408, gnorm=1.5744, throughput=18.01K wps, wc=50.20K\n2018-09-21 22:05:44,552 - root - [Epoch 0 Batch 30/1043] loss=6.3708, ppl=584.5351, gnorm=0.8043, throughput=19.91K wps, wc=67.78K\n2018-09-21 22:05:47,829 - root - [Epoch 0 Batch 40/1043] loss=6.1793, ppl=482.6752, gnorm=0.6221, throughput=19.31K wps, wc=63.19K\n2018-09-21 22:05:51,120 - root - [Epoch 0 Batch 50/1043] loss=6.1887, ppl=487.2141, gnorm=0.4079, throughput=18.85K wps, wc=61.93K\n2018-09-21 22:05:54,096 - root - [Epoch 0 Batch 60/1043] loss=6.1112, ppl=450.8753, gnorm=0.6981, throughput=19.92K wps, wc=59.19K\n2018-09-21 22:05:57,749 - root - [Epoch 0 Batch 70/1043] loss=6.1591, ppl=473.0155, gnorm=0.4729, throughput=20.00K wps, wc=72.99K\n2018-09-21 22:06:01,083 - root - [Epoch 0 Batch 80/1043] loss=6.0703, ppl=432.8200, gnorm=0.4039, throughput=19.41K wps, wc=64.58K\n2018-09-21 22:06:03,923 - root - [Epoch 0 Batch 90/1043] loss=5.9390, ppl=379.5640, gnorm=0.3632, throughput=18.68K wps, wc=53.02K\n2018-09-21 22:06:06,979 - root - [Epoch 0 Batch 100/1043] loss=5.8791, ppl=357.4793, gnorm=0.4188, throughput=19.47K wps, wc=59.42K\n2018-09-21 22:06:10,266 - root - [Epoch 0 Batch 110/1043] loss=5.8729, ppl=355.2918, gnorm=0.3631, throughput=19.96K wps, wc=65.50K\n2018-09-21 22:06:13,233 - root - [Epoch 0 Batch 120/1043] loss=5.8700, ppl=354.2508, gnorm=0.3343, throughput=19.70K wps, wc=58.43K\n2018-09-21 22:06:16,941 - root - [Epoch 0 Batch 130/1043] loss=5.9359, ppl=378.3900, gnorm=0.3467, throughput=16.03K wps, wc=59.39K\n2018-09-21 22:06:20,179 - root - [Epoch 0 Batch 140/1043] loss=5.8855, ppl=359.7728, gnorm=0.2827, throughput=18.92K wps, wc=61.18K\n2018-09-21 22:06:23,244 - root - [Epoch 0 Batch 150/1043] loss=5.8198, ppl=336.9059, gnorm=0.2843, throughput=18.40K wps, wc=56.34K\n2018-09-21 22:06:26,700 - root - [Epoch 0 Batch 160/1043] loss=5.7556, ppl=315.9615, gnorm=0.4015, throughput=16.77K wps, wc=57.93K\n2018-09-21 22:06:30,085 - root - [Epoch 0 Batch 170/1043] loss=5.7772, ppl=322.8600, gnorm=0.3006, throughput=19.06K wps, wc=64.42K\n2018-09-21 22:06:32,672 - root - [Epoch 0 Batch 180/1043] loss=5.4695, ppl=237.3432, gnorm=0.3297, throughput=17.17K wps, wc=44.31K\n2018-09-21 22:06:36,042 - root - [Epoch 0 Batch 190/1043] loss=5.6266, ppl=277.7225, gnorm=0.3634, throughput=18.57K wps, wc=62.45K\n2018-09-21 22:06:38,986 - root - [Epoch 0 Batch 200/1043] loss=5.5116, ppl=247.5500, gnorm=0.3427, throughput=18.45K wps, wc=54.24K\n2018-09-21 22:06:41,814 - root - [Epoch 0 Batch 210/1043] loss=5.3337, ppl=207.1959, gnorm=0.4309, throughput=18.66K wps, wc=52.68K\n2018-09-21 22:06:44,569 - root - [Epoch 0 Batch 220/1043] loss=5.3188, ppl=204.1433, gnorm=0.3864, throughput=18.34K wps, wc=50.47K\n2018-09-21 22:06:47,729 - root - [Epoch 0 Batch 230/1043] loss=5.3171, ppl=203.7825, gnorm=0.4236, throughput=19.54K wps, wc=61.65K\n2018-09-21 22:06:50,659 - root - [Epoch 0 Batch 240/1043] loss=5.3093, ppl=202.1996, gnorm=0.3478, throughput=19.92K wps, wc=58.32K\n2018-09-21 22:06:54,197 - root - [Epoch 0 Batch 250/1043] loss=5.4303, ppl=228.2228, gnorm=0.2912, throughput=20.04K wps, wc=70.84K\n2018-09-21 22:06:57,309 - root - [Epoch 0 Batch 260/1043] loss=5.2876, ppl=197.8658, gnorm=0.3685, throughput=19.38K wps, wc=60.22K\n2018-09-21 22:07:00,958 - root - [Epoch 0 Batch 270/1043] loss=5.3883, ppl=218.8361, gnorm=0.2981, throughput=20.01K wps, wc=72.96K\n2018-09-21 22:07:04,223 - root - [Epoch 0 Batch 280/1043] loss=5.2595, ppl=192.3885, gnorm=0.2640, throughput=18.64K wps, wc=60.80K\n2018-09-21 22:07:06,808 - root - [Epoch 0 Batch 290/1043] loss=4.9837, ppl=146.0157, gnorm=0.3421, throughput=17.74K wps, wc=45.79K\n2018-09-21 22:07:09,829 - root - [Epoch 0 Batch 300/1043] loss=5.0936, ppl=162.9761, gnorm=0.3339, throughput=19.57K wps, wc=59.05K\n2018-09-21 22:07:13,136 - root - [Epoch 0 Batch 310/1043] loss=5.1146, ppl=166.4307, gnorm=0.2916, throughput=18.64K wps, wc=61.58K\n2018-09-21 22:07:15,991 - root - [Epoch 0 Batch 320/1043] loss=4.9891, ppl=146.8112, gnorm=0.3041, throughput=18.63K wps, wc=53.10K\n2018-09-21 22:07:19,183 - root - [Epoch 0 Batch 330/1043] loss=5.0453, ppl=155.2869, gnorm=0.3002, throughput=19.24K wps, wc=61.37K\n2018-09-21 22:07:22,233 - root - [Epoch 0 Batch 340/1043] loss=5.0680, ppl=158.8540, gnorm=0.2632, throughput=18.66K wps, wc=56.88K\n2018-09-21 22:07:25,053 - root - [Epoch 0 Batch 350/1043] loss=4.9312, ppl=138.5427, gnorm=0.3213, throughput=19.48K wps, wc=54.86K\n2018-09-21 22:07:28,359 - root - [Epoch 0 Batch 360/1043] loss=5.0529, ppl=156.4758, gnorm=0.2851, throughput=19.54K wps, wc=64.55K\n2018-09-21 22:07:31,879 - root - [Epoch 0 Batch 370/1043] loss=4.9171, ppl=136.6010, gnorm=0.2945, throughput=19.05K wps, wc=66.97K\n2018-09-21 22:07:34,842 - root - [Epoch 0 Batch 380/1043] loss=4.8253, ppl=124.6236, gnorm=0.3033, throughput=17.86K wps, wc=52.79K\n2018-09-21 22:07:37,656 - root - [Epoch 0 Batch 390/1043] loss=4.8032, ppl=121.8960, gnorm=0.3442, throughput=18.13K wps, wc=50.94K\n2018-09-21 22:07:40,299 - root - [Epoch 0 Batch 400/1043] loss=4.6603, ppl=105.6630, gnorm=0.3571, throughput=18.27K wps, wc=48.22K\n2018-09-21 22:07:43,038 - root - [Epoch 0 Batch 410/1043] loss=4.8334, ppl=125.6321, gnorm=0.3059, throughput=17.65K wps, wc=48.27K\n2018-09-21 22:07:46,112 - root - [Epoch 0 Batch 420/1043] loss=4.8602, ppl=129.0516, gnorm=0.3092, throughput=18.29K wps, wc=56.14K\n2018-09-21 22:07:49,505 - root - [Epoch 0 Batch 430/1043] loss=4.9080, ppl=135.3675, gnorm=0.2933, throughput=20.46K wps, wc=69.33K\n2018-09-21 22:07:52,843 - root - [Epoch 0 Batch 440/1043] loss=4.8004, ppl=121.5644, gnorm=0.2689, throughput=20.11K wps, wc=67.08K\n2018-09-21 22:07:55,728 - root - [Epoch 0 Batch 450/1043] loss=4.6993, ppl=109.8657, gnorm=0.2928, throughput=18.64K wps, wc=53.68K\n2018-09-21 22:07:58,448 - root - [Epoch 0 Batch 460/1043] loss=4.4873, ppl=88.8854, gnorm=0.3431, throughput=18.56K wps, wc=50.38K\n2018-09-21 22:08:01,641 - root - [Epoch 0 Batch 470/1043] loss=4.7631, ppl=117.1051, gnorm=0.2814, throughput=19.02K wps, wc=60.70K\n2018-09-21 22:08:04,526 - root - [Epoch 0 Batch 480/1043] loss=4.4122, ppl=82.4478, gnorm=0.3448, throughput=18.76K wps, wc=54.04K\n2018-09-21 22:08:07,151 - root - [Epoch 0 Batch 490/1043] loss=4.5267, ppl=92.4517, gnorm=0.4326, throughput=17.66K wps, wc=46.32K\n2018-09-21 22:08:09,896 - root - [Epoch 0 Batch 500/1043] loss=4.5806, ppl=97.5688, gnorm=0.3167, throughput=17.70K wps, wc=48.55K\n2018-09-21 22:08:12,243 - root - [Epoch 0 Batch 510/1043] loss=4.3424, ppl=76.8895, gnorm=0.3342, throughput=17.77K wps, wc=41.62K\n2018-09-21 22:08:14,481 - root - [Epoch 0 Batch 520/1043] loss=4.2492, ppl=70.0514, gnorm=0.3677, throughput=17.88K wps, wc=39.98K\n2018-09-21 22:08:17,630 - root - [Epoch 0 Batch 530/1043] loss=4.6692, ppl=106.6142, gnorm=0.3126, throughput=18.62K wps, wc=58.58K\n2018-09-21 22:08:20,432 - root - [Epoch 0 Batch 540/1043] loss=4.5282, ppl=92.5930, gnorm=0.3146, throughput=18.12K wps, wc=50.72K\n2018-09-21 22:08:23,625 - root - [Epoch 0 Batch 550/1043] loss=4.5519, ppl=94.8168, gnorm=0.3138, throughput=19.73K wps, wc=62.95K\n2018-09-21 22:08:26,168 - root - [Epoch 0 Batch 560/1043] loss=4.3375, ppl=76.5171, gnorm=0.3368, throughput=18.15K wps, wc=46.13K\n2018-09-21 22:08:29,203 - root - [Epoch 0 Batch 570/1043] loss=4.3641, ppl=78.5776, gnorm=0.3007, throughput=20.17K wps, wc=61.12K\n2018-09-21 22:08:32,148 - root - [Epoch 0 Batch 580/1043] loss=4.3685, ppl=78.9221, gnorm=0.2993, throughput=18.84K wps, wc=55.43K\n2018-09-21 22:08:35,639 - root - [Epoch 0 Batch 590/1043] loss=4.5959, ppl=99.0810, gnorm=0.2571, throughput=21.20K wps, wc=73.93K\n2018-09-21 22:08:38,621 - root - [Epoch 0 Batch 600/1043] loss=4.5030, ppl=90.2874, gnorm=0.2763, throughput=18.73K wps, wc=55.80K\n2018-09-21 22:08:41,491 - root - [Epoch 0 Batch 610/1043] loss=4.3272, ppl=75.7355, gnorm=0.3350, throughput=18.23K wps, wc=52.28K\n2018-09-21 22:08:45,007 - root - [Epoch 0 Batch 620/1043] loss=4.5685, ppl=96.4009, gnorm=0.2640, throughput=20.60K wps, wc=72.39K\n2018-09-21 22:08:47,087 - root - [Epoch 0 Batch 630/1043] loss=4.0605, ppl=58.0019, gnorm=0.3316, throughput=16.59K wps, wc=34.44K\n2018-09-21 22:08:50,032 - root - [Epoch 0 Batch 640/1043] loss=4.3434, ppl=76.9717, gnorm=0.3179, throughput=19.50K wps, wc=57.32K\n2018-09-21 22:08:53,473 - root - [Epoch 0 Batch 650/1043] loss=4.4893, ppl=89.0569, gnorm=0.2840, throughput=19.32K wps, wc=66.42K\n2018-09-21 22:08:55,951 - root - [Epoch 0 Batch 660/1043] loss=4.2164, ppl=67.7878, gnorm=0.3611, throughput=17.95K wps, wc=44.42K\n2018-09-21 22:08:59,920 - root - [Epoch 0 Batch 670/1043] loss=4.6150, ppl=100.9885, gnorm=0.2676, throughput=19.59K wps, wc=77.68K\n2018-09-21 22:09:03,166 - root - [Epoch 0 Batch 680/1043] loss=4.4367, ppl=84.4951, gnorm=0.2724, throughput=18.21K wps, wc=59.02K\n2018-09-21 22:09:05,967 - root - [Epoch 0 Batch 690/1043] loss=4.2191, ppl=67.9725, gnorm=0.3161, throughput=18.07K wps, wc=50.54K\n2018-09-21 22:09:08,868 - root - [Epoch 0 Batch 700/1043] loss=4.3094, ppl=74.3939, gnorm=0.2945, throughput=18.12K wps, wc=52.45K\n2018-09-21 22:09:11,332 - root - [Epoch 0 Batch 710/1043] loss=4.2017, ppl=66.8025, gnorm=0.3359, throughput=16.80K wps, wc=41.32K\n2018-09-21 22:09:14,070 - root - [Epoch 0 Batch 720/1043] loss=4.2778, ppl=72.0782, gnorm=0.2971, throughput=18.32K wps, wc=50.09K\n2018-09-21 22:09:16,943 - root - [Epoch 0 Batch 730/1043] loss=4.2019, ppl=66.8124, gnorm=0.3055, throughput=18.13K wps, wc=52.02K\n2018-09-21 22:09:20,146 - root - [Epoch 0 Batch 740/1043] loss=4.3440, ppl=77.0135, gnorm=0.2719, throughput=18.71K wps, wc=59.86K\n2018-09-21 22:09:23,066 - root - [Epoch 0 Batch 750/1043] loss=4.2010, ppl=66.7560, gnorm=0.2858, throughput=18.30K wps, wc=53.35K\n2018-09-21 22:09:26,538 - root - [Epoch 0 Batch 760/1043] loss=4.3022, ppl=73.8630, gnorm=0.2944, throughput=20.20K wps, wc=70.09K\n2018-09-21 22:09:29,021 - root - [Epoch 0 Batch 770/1043] loss=4.1313, ppl=62.2570, gnorm=0.3020, throughput=17.43K wps, wc=43.21K\n2018-09-21 22:09:32,738 - root - [Epoch 0 Batch 780/1043] loss=4.3086, ppl=74.3333, gnorm=0.2551, throughput=20.45K wps, wc=75.93K\n2018-09-21 22:09:35,368 - root - [Epoch 0 Batch 790/1043] loss=4.1472, ppl=63.2548, gnorm=0.3048, throughput=17.83K wps, wc=46.81K\n2018-09-21 22:09:38,523 - root - [Epoch 0 Batch 800/1043] loss=4.2139, ppl=67.6212, gnorm=0.3017, throughput=18.64K wps, wc=58.72K\n2018-09-21 22:09:41,529 - root - [Epoch 0 Batch 810/1043] loss=4.0840, ppl=59.3807, gnorm=0.2963, throughput=19.13K wps, wc=57.38K\n2018-09-21 22:09:44,549 - root - [Epoch 0 Batch 820/1043] loss=3.9759, ppl=53.2957, gnorm=0.3254, throughput=19.41K wps, wc=58.52K\n2018-09-21 22:09:47,623 - root - [Epoch 0 Batch 830/1043] loss=4.1327, ppl=62.3458, gnorm=0.3448, throughput=18.64K wps, wc=57.24K\n2018-09-21 22:09:50,443 - root - [Epoch 0 Batch 840/1043] loss=4.0973, ppl=60.1790, gnorm=0.3150, throughput=18.68K wps, wc=52.57K\n2018-09-21 22:09:53,676 - root - [Epoch 0 Batch 850/1043] loss=4.1487, ppl=63.3536, gnorm=0.3111, throughput=19.86K wps, wc=64.14K\n2018-09-21 22:09:56,622 - root - [Epoch 0 Batch 860/1043] loss=4.1350, ppl=62.4871, gnorm=0.2901, throughput=18.58K wps, wc=54.64K\n2018-09-21 22:09:59,971 - root - [Epoch 0 Batch 870/1043] loss=4.1729, ppl=64.9006, gnorm=0.3284, throughput=19.67K wps, wc=65.81K\n2018-09-21 22:10:02,728 - root - [Epoch 0 Batch 880/1043] loss=4.1088, ppl=60.8733, gnorm=0.3028, throughput=18.27K wps, wc=50.27K\n2018-09-21 22:10:05,733 - root - [Epoch 0 Batch 890/1043] loss=4.1634, ppl=64.2903, gnorm=0.2777, throughput=18.70K wps, wc=56.11K\n2018-09-21 22:10:08,929 - root - [Epoch 0 Batch 900/1043] loss=4.1853, ppl=65.7101, gnorm=0.2858, throughput=19.08K wps, wc=60.91K\n2018-09-21 22:10:11,791 - root - [Epoch 0 Batch 910/1043] loss=3.9864, ppl=53.8625, gnorm=0.2996, throughput=18.07K wps, wc=51.65K\n2018-09-21 22:10:14,985 - root - [Epoch 0 Batch 920/1043] loss=4.1712, ppl=64.7910, gnorm=0.2745, throughput=18.98K wps, wc=60.52K\n2018-09-21 22:10:17,548 - root - [Epoch 0 Batch 930/1043] loss=3.9789, ppl=53.4599, gnorm=0.2938, throughput=17.00K wps, wc=43.51K\n2018-09-21 22:10:20,276 - root - [Epoch 0 Batch 940/1043] loss=3.9308, ppl=50.9497, gnorm=0.3176, throughput=18.25K wps, wc=49.71K\n2018-09-21 22:10:23,738 - root - [Epoch 0 Batch 950/1043] loss=4.2074, ppl=67.1815, gnorm=0.2732, throughput=20.50K wps, wc=70.92K\n2018-09-21 22:10:27,456 - root - [Epoch 0 Batch 960/1043] loss=4.1580, ppl=63.9440, gnorm=0.2796, throughput=19.95K wps, wc=74.06K\n2018-09-21 22:10:29,996 - root - [Epoch 0 Batch 970/1043] loss=3.8830, ppl=48.5707, gnorm=0.3559, throughput=17.37K wps, wc=44.03K\n2018-09-21 22:10:33,595 - root - [Epoch 0 Batch 980/1043] loss=4.1896, ppl=65.9942, gnorm=0.2912, throughput=20.22K wps, wc=72.73K\n2018-09-21 22:10:36,870 - root - [Epoch 0 Batch 990/1043] loss=4.0976, ppl=60.1968, gnorm=0.2941, throughput=19.99K wps, wc=65.39K\n2018-09-21 22:10:39,876 - root - [Epoch 0 Batch 1000/1043] loss=3.9923, ppl=54.1769, gnorm=0.3028, throughput=18.53K wps, wc=55.65K\n2018-09-21 22:10:43,150 - root - [Epoch 0 Batch 1010/1043] loss=4.0832, ppl=59.3326, gnorm=0.2839, throughput=20.06K wps, wc=65.61K\n2018-09-21 22:10:45,743 - root - [Epoch 0 Batch 1020/1043] loss=3.8192, ppl=45.5696, gnorm=0.3245, throughput=17.66K wps, wc=45.72K\n2018-09-21 22:10:48,831 - root - [Epoch 0 Batch 1030/1043] loss=3.9808, ppl=53.5582, gnorm=0.3004, throughput=18.15K wps, wc=55.97K\n2018-09-21 22:10:51,716 - root - [Epoch 0 Batch 1040/1043] loss=3.9463, ppl=51.7413, gnorm=0.3653, throughput=18.49K wps, wc=53.27K\n2018-09-21 22:11:19,017 - root - [Epoch 0] valid Loss=2.8567, valid ppl=17.4045, valid bleu=2.95\n2018-09-21 22:11:43,356 - root - [Epoch 0] test Loss=3.0003, test ppl=20.0925, test bleu=2.68\n2018-09-21 22:11:43,368 - root - Save best parameters to gnmt_en_vi_u512/valid_best.params\n2018-09-21 22:11:43,604 - root - Learning rate change to 0.0005\n2018-09-21 22:11:47,597 - root - [Epoch 1 Batch 10/1043] loss=4.0216, ppl=55.7914, gnorm=0.3151, throughput=16.31K wps, wc=65.08K\n2018-09-21 22:11:51,280 - root - [Epoch 1 Batch 20/1043] loss=4.0221, ppl=55.8157, gnorm=0.2725, throughput=19.51K wps, wc=71.73K\n2018-09-21 22:11:54,471 - root - [Epoch 1 Batch 30/1043] loss=3.8743, ppl=48.1497, gnorm=0.2847, throughput=19.28K wps, wc=61.44K\n2018-09-21 22:11:56,727 - root - [Epoch 1 Batch 40/1043] loss=3.5596, ppl=35.1497, gnorm=0.3121, throughput=16.75K wps, wc=37.67K\n2018-09-21 22:11:58,956 - root - [Epoch 1 Batch 50/1043] loss=3.4704, ppl=32.1484, gnorm=0.3277, throughput=16.98K wps, wc=37.77K\n2018-09-21 22:12:02,582 - root - [Epoch 1 Batch 60/1043] loss=4.0301, ppl=56.2675, gnorm=0.2729, throughput=20.76K wps, wc=75.14K\n2018-09-21 22:12:05,806 - root - [Epoch 1 Batch 70/1043] loss=3.9115, ppl=49.9753, gnorm=0.2712, throughput=18.80K wps, wc=60.52K\n2018-09-21 22:12:09,006 - root - [Epoch 1 Batch 80/1043] loss=3.8136, ppl=45.3112, gnorm=0.2917, throughput=19.90K wps, wc=63.60K\n2018-09-21 22:12:12,229 - root - [Epoch 1 Batch 90/1043] loss=3.9610, ppl=52.5084, gnorm=0.3412, throughput=19.59K wps, wc=63.07K\n2018-09-21 22:12:15,242 - root - [Epoch 1 Batch 100/1043] loss=3.6872, ppl=39.9347, gnorm=0.3091, throughput=19.16K wps, wc=57.64K\n2018-09-21 22:12:18,366 - root - [Epoch 1 Batch 110/1043] loss=3.8353, ppl=46.3090, gnorm=0.2950, throughput=18.53K wps, wc=57.82K\n2018-09-21 22:12:21,381 - root - [Epoch 1 Batch 120/1043] loss=3.7743, ppl=43.5659, gnorm=0.3473, throughput=18.62K wps, wc=56.05K\n2018-09-21 22:12:25,227 - root - [Epoch 1 Batch 130/1043] loss=4.0320, ppl=56.3763, gnorm=0.2871, throughput=20.17K wps, wc=77.45K\n2018-09-21 22:12:28,940 - root - [Epoch 1 Batch 140/1043] loss=3.8535, ppl=47.1555, gnorm=0.2850, throughput=19.24K wps, wc=71.37K\n2018-09-21 22:12:32,939 - root - [Epoch 1 Batch 150/1043] loss=4.0917, ppl=59.8412, gnorm=0.2456, throughput=20.68K wps, wc=82.65K\n2018-09-21 22:12:35,910 - root - [Epoch 1 Batch 160/1043] loss=3.8522, ppl=47.0983, gnorm=0.2870, throughput=18.41K wps, wc=54.62K\n2018-09-21 22:12:39,161 - root - [Epoch 1 Batch 170/1043] loss=3.8073, ppl=45.0271, gnorm=0.3042, throughput=20.86K wps, wc=67.68K\n2018-09-21 22:12:42,041 - root - [Epoch 1 Batch 180/1043] loss=3.7661, ppl=43.2118, gnorm=0.3045, throughput=18.45K wps, wc=53.09K\n2018-09-21 22:12:44,714 - root - [Epoch 1 Batch 190/1043] loss=3.6400, ppl=38.0931, gnorm=0.3272, throughput=18.08K wps, wc=48.29K\n2018-09-21 22:12:47,882 - root - [Epoch 1 Batch 200/1043] loss=3.8518, ppl=47.0794, gnorm=0.2976, throughput=19.78K wps, wc=62.59K\n2018-09-21 22:12:50,924 - root - [Epoch 1 Batch 210/1043] loss=3.7456, ppl=42.3332, gnorm=0.3093, throughput=19.32K wps, wc=58.70K\n2018-09-21 22:12:54,697 - root - [Epoch 1 Batch 220/1043] loss=4.0036, ppl=54.7936, gnorm=0.3090, throughput=17.79K wps, wc=67.07K\n2018-09-21 22:12:57,564 - root - [Epoch 1 Batch 230/1043] loss=3.7653, ppl=43.1781, gnorm=0.3132, throughput=18.37K wps, wc=52.58K\n2018-09-21 22:13:00,316 - root - [Epoch 1 Batch 240/1043] loss=3.6769, ppl=39.5236, gnorm=0.3120, throughput=18.86K wps, wc=51.83K\n2018-09-21 22:13:03,198 - root - [Epoch 1 Batch 250/1043] loss=3.7738, ppl=43.5456, gnorm=0.3416, throughput=19.60K wps, wc=56.43K\n2018-09-21 22:13:05,992 - root - [Epoch 1 Batch 260/1043] loss=3.6796, ppl=39.6317, gnorm=0.3071, throughput=17.44K wps, wc=48.67K\n2018-09-21 22:13:09,146 - root - [Epoch 1 Batch 270/1043] loss=3.8355, ppl=46.3144, gnorm=0.2962, throughput=20.29K wps, wc=63.93K\n2018-09-21 22:13:12,444 - root - [Epoch 1 Batch 280/1043] loss=3.8152, ppl=45.3864, gnorm=0.2999, throughput=20.36K wps, wc=67.06K\n2018-09-21 22:13:15,839 - root - [Epoch 1 Batch 290/1043] loss=3.8797, ppl=48.4082, gnorm=0.2804, throughput=19.66K wps, wc=66.64K\n2018-09-21 22:13:19,087 - root - [Epoch 1 Batch 300/1043] loss=3.7924, ppl=44.3617, gnorm=0.2827, throughput=20.57K wps, wc=66.76K\n2018-09-21 22:13:22,025 - root - [Epoch 1 Batch 310/1043] loss=3.7227, ppl=41.3743, gnorm=0.3199, throughput=19.69K wps, wc=57.80K\n2018-09-21 22:13:24,177 - root - [Epoch 1 Batch 320/1043] loss=3.4170, ppl=30.4774, gnorm=0.3457, throughput=16.47K wps, wc=35.38K\n2018-09-21 22:13:27,240 - root - [Epoch 1 Batch 330/1043] loss=3.6920, ppl=40.1253, gnorm=0.3061, throughput=18.47K wps, wc=56.52K\n2018-09-21 22:13:30,916 - root - [Epoch 1 Batch 340/1043] loss=3.8938, ppl=49.0958, gnorm=0.2857, throughput=19.60K wps, wc=72.01K\n2018-09-21 22:13:33,634 - root - [Epoch 1 Batch 350/1043] loss=3.6124, ppl=37.0535, gnorm=0.3279, throughput=19.38K wps, wc=52.60K\n2018-09-21 22:13:37,616 - root - [Epoch 1 Batch 360/1043] loss=3.9422, ppl=51.5298, gnorm=0.2947, throughput=20.68K wps, wc=82.28K\n2018-09-21 22:13:40,531 - root - [Epoch 1 Batch 370/1043] loss=3.5470, ppl=34.7074, gnorm=0.3431, throughput=18.73K wps, wc=54.51K\n2018-09-21 22:13:43,878 - root - [Epoch 1 Batch 380/1043] loss=3.7705, ppl=43.4007, gnorm=0.3105, throughput=19.47K wps, wc=65.12K\n2018-09-21 22:13:47,389 - root - [Epoch 1 Batch 390/1043] loss=3.8030, ppl=44.8367, gnorm=0.3184, throughput=19.58K wps, wc=68.68K\n2018-09-21 22:13:49,806 - root - [Epoch 1 Batch 400/1043] loss=3.4874, ppl=32.7005, gnorm=0.3428, throughput=16.92K wps, wc=40.81K\n2018-09-21 22:13:53,088 - root - [Epoch 1 Batch 410/1043] loss=3.7097, ppl=40.8434, gnorm=0.2982, throughput=18.90K wps, wc=61.97K\n2018-09-21 22:13:56,781 - root - [Epoch 1 Batch 420/1043] loss=3.8401, ppl=46.5322, gnorm=0.2935, throughput=19.92K wps, wc=73.42K\n2018-09-21 22:13:59,524 - root - [Epoch 1 Batch 430/1043] loss=3.5757, ppl=35.7203, gnorm=0.3156, throughput=19.47K wps, wc=53.33K\n2018-09-21 22:14:02,442 - root - [Epoch 1 Batch 440/1043] loss=3.6379, ppl=38.0113, gnorm=0.3129, throughput=19.37K wps, wc=56.45K\n2018-09-21 22:14:04,983 - root - [Epoch 1 Batch 450/1043] loss=3.5024, ppl=33.1967, gnorm=0.3367, throughput=18.54K wps, wc=47.03K\n2018-09-21 22:14:08,035 - root - [Epoch 1 Batch 460/1043] loss=3.6477, ppl=38.3866, gnorm=0.3150, throughput=19.89K wps, wc=60.63K\n2018-09-21 22:14:10,632 - root - [Epoch 1 Batch 470/1043] loss=3.5192, ppl=33.7587, gnorm=0.3360, throughput=17.11K wps, wc=44.36K\n2018-09-21 22:14:13,391 - root - [Epoch 1 Batch 480/1043] loss=3.5442, ppl=34.6125, gnorm=0.3449, throughput=18.46K wps, wc=50.86K\n2018-09-21 22:14:16,173 - root - [Epoch 1 Batch 490/1043] loss=3.5625, ppl=35.2514, gnorm=0.3294, throughput=18.87K wps, wc=52.44K\n2018-09-21 22:14:19,498 - root - [Epoch 1 Batch 500/1043] loss=3.7279, ppl=41.5921, gnorm=0.3308, throughput=20.06K wps, wc=66.65K\n2018-09-21 22:14:22,304 - root - [Epoch 1 Batch 510/1043] loss=3.4703, ppl=32.1463, gnorm=0.3557, throughput=18.53K wps, wc=51.95K\n2018-09-21 22:14:25,104 - root - [Epoch 1 Batch 520/1043] loss=3.5954, ppl=36.4298, gnorm=0.3330, throughput=18.73K wps, wc=52.40K\n2018-09-21 22:14:28,287 - root - [Epoch 1 Batch 530/1043] loss=3.6642, ppl=39.0252, gnorm=0.2974, throughput=19.59K wps, wc=62.28K\n2018-09-21 22:14:30,536 - root - [Epoch 1 Batch 540/1043] loss=3.3345, ppl=28.0653, gnorm=0.3542, throughput=16.40K wps, wc=36.85K\n2018-09-21 22:14:33,619 - root - [Epoch 1 Batch 550/1043] loss=3.4951, ppl=32.9527, gnorm=0.3506, throughput=20.18K wps, wc=62.18K\n2018-09-21 22:14:36,055 - root - [Epoch 1 Batch 560/1043] loss=3.5026, ppl=33.2023, gnorm=0.3399, throughput=18.30K wps, wc=44.51K\n2018-09-21 22:14:39,111 - root - [Epoch 1 Batch 570/1043] loss=3.5692, ppl=35.4887, gnorm=0.3710, throughput=19.16K wps, wc=58.51K\n2018-09-21 22:14:41,928 - root - [Epoch 1 Batch 580/1043] loss=3.5191, ppl=33.7534, gnorm=0.3485, throughput=18.32K wps, wc=51.54K\n2018-09-21 22:14:44,706 - root - [Epoch 1 Batch 590/1043] loss=3.3927, ppl=29.7454, gnorm=0.3475, throughput=18.47K wps, wc=51.26K\n2018-09-21 22:14:47,454 - root - [Epoch 1 Batch 600/1043] loss=3.4244, ppl=30.7057, gnorm=0.3471, throughput=18.27K wps, wc=50.15K\n2018-09-21 22:14:50,927 - root - [Epoch 1 Batch 610/1043] loss=3.6778, ppl=39.5579, gnorm=0.3206, throughput=21.20K wps, wc=73.53K\n2018-09-21 22:14:53,319 - root - [Epoch 1 Batch 620/1043] loss=3.3094, ppl=27.3677, gnorm=0.3554, throughput=17.18K wps, wc=41.04K\n2018-09-21 22:14:56,319 - root - [Epoch 1 Batch 630/1043] loss=3.5596, ppl=35.1506, gnorm=0.3167, throughput=18.33K wps, wc=54.92K\n2018-09-21 22:14:59,217 - root - [Epoch 1 Batch 640/1043] loss=3.4827, ppl=32.5476, gnorm=0.3518, throughput=18.77K wps, wc=54.32K\n2018-09-21 22:15:02,041 - root - [Epoch 1 Batch 650/1043] loss=3.4443, ppl=31.3226, gnorm=0.3421, throughput=17.63K wps, wc=49.69K\n2018-09-21 22:15:05,152 - root - [Epoch 1 Batch 660/1043] loss=3.5432, ppl=34.5771, gnorm=0.3270, throughput=18.29K wps, wc=56.81K\n2018-09-21 22:15:08,442 - root - [Epoch 1 Batch 670/1043] loss=3.4695, ppl=32.1223, gnorm=0.3566, throughput=18.48K wps, wc=60.72K\n2018-09-21 22:15:12,140 - root - [Epoch 1 Batch 680/1043] loss=3.6098, ppl=36.9571, gnorm=0.3254, throughput=20.32K wps, wc=75.07K\n2018-09-21 22:15:15,470 - root - [Epoch 1 Batch 690/1043] loss=3.6118, ppl=37.0332, gnorm=0.3445, throughput=19.75K wps, wc=65.71K\n2018-09-21 22:15:18,746 - root - [Epoch 1 Batch 700/1043] loss=3.5356, ppl=34.3148, gnorm=0.3431, throughput=18.39K wps, wc=60.12K\n2018-09-21 22:15:21,718 - root - [Epoch 1 Batch 710/1043] loss=3.4813, ppl=32.5024, gnorm=0.3511, throughput=18.74K wps, wc=55.62K\n2018-09-21 22:15:25,123 - root - [Epoch 1 Batch 720/1043] loss=3.5338, ppl=34.2549, gnorm=0.3374, throughput=20.09K wps, wc=68.33K\n2018-09-21 22:15:28,059 - root - [Epoch 1 Batch 730/1043] loss=3.4353, ppl=31.0409, gnorm=0.3480, throughput=18.67K wps, wc=54.75K\n2018-09-21 22:15:30,666 - root - [Epoch 1 Batch 740/1043] loss=3.2414, ppl=25.5698, gnorm=0.4012, throughput=18.55K wps, wc=48.33K\n2018-09-21 22:15:33,891 - root - [Epoch 1 Batch 750/1043] loss=3.5470, ppl=34.7102, gnorm=0.3399, throughput=19.36K wps, wc=62.37K\n2018-09-21 22:15:36,468 - root - [Epoch 1 Batch 760/1043] loss=3.2693, ppl=26.2924, gnorm=0.4124, throughput=18.39K wps, wc=47.31K\n2018-09-21 22:15:39,309 - root - [Epoch 1 Batch 770/1043] loss=3.3981, ppl=29.9060, gnorm=0.3663, throughput=18.03K wps, wc=51.20K\n2018-09-21 22:15:42,254 - root - [Epoch 1 Batch 780/1043] loss=3.4692, ppl=32.1125, gnorm=0.3408, throughput=18.65K wps, wc=54.88K\n2018-09-21 22:15:45,555 - root - [Epoch 1 Batch 790/1043] loss=3.4617, ppl=31.8700, gnorm=0.3658, throughput=20.59K wps, wc=67.87K\n2018-09-21 22:15:48,440 - root - [Epoch 1 Batch 800/1043] loss=3.4041, ppl=30.0871, gnorm=0.3600, throughput=18.42K wps, wc=53.09K\n2018-09-21 22:15:51,357 - root - [Epoch 1 Batch 810/1043] loss=3.3979, ppl=29.8998, gnorm=0.3272, throughput=18.14K wps, wc=52.84K\n2018-09-21 22:15:54,171 - root - [Epoch 1 Batch 820/1043] loss=3.3658, ppl=28.9559, gnorm=0.3530, throughput=18.99K wps, wc=53.37K\n2018-09-21 22:15:57,639 - root - [Epoch 1 Batch 830/1043] loss=3.5321, ppl=34.1964, gnorm=0.3084, throughput=19.91K wps, wc=69.01K\n2018-09-21 22:16:01,302 - root - [Epoch 1 Batch 840/1043] loss=3.5586, ppl=35.1157, gnorm=0.2983, throughput=20.87K wps, wc=76.39K\n2018-09-21 22:16:04,269 - root - [Epoch 1 Batch 850/1043] loss=3.4558, ppl=31.6845, gnorm=0.3278, throughput=19.49K wps, wc=57.73K\n2018-09-21 22:16:06,999 - root - [Epoch 1 Batch 860/1043] loss=3.3437, ppl=28.3233, gnorm=0.3400, throughput=18.67K wps, wc=50.93K\n2018-09-21 22:16:09,248 - root - [Epoch 1 Batch 870/1043] loss=3.2422, ppl=25.5897, gnorm=0.3689, throughput=19.22K wps, wc=43.16K\n2018-09-21 22:16:11,689 - root - [Epoch 1 Batch 880/1043] loss=3.2668, ppl=26.2275, gnorm=0.3587, throughput=18.62K wps, wc=45.40K\n2018-09-21 22:16:14,989 - root - [Epoch 1 Batch 890/1043] loss=3.5027, ppl=33.2066, gnorm=0.3278, throughput=21.26K wps, wc=70.10K\n2018-09-21 22:16:17,847 - root - [Epoch 1 Batch 900/1043] loss=3.3733, ppl=29.1758, gnorm=0.3793, throughput=20.72K wps, wc=59.13K\n2018-09-21 22:16:20,613 - root - [Epoch 1 Batch 910/1043] loss=3.2748, ppl=26.4368, gnorm=0.3896, throughput=17.01K wps, wc=46.98K\n2018-09-21 22:16:23,899 - root - [Epoch 1 Batch 920/1043] loss=3.4553, ppl=31.6693, gnorm=0.3272, throughput=20.25K wps, wc=66.44K\n2018-09-21 22:16:26,523 - root - [Epoch 1 Batch 930/1043] loss=3.3658, ppl=28.9571, gnorm=0.3593, throughput=19.10K wps, wc=50.01K\n2018-09-21 22:16:29,232 - root - [Epoch 1 Batch 940/1043] loss=3.2992, ppl=27.0911, gnorm=0.3831, throughput=19.64K wps, wc=53.14K\n2018-09-21 22:16:31,918 - root - [Epoch 1 Batch 950/1043] loss=3.2734, ppl=26.4008, gnorm=0.3673, throughput=19.04K wps, wc=51.06K\n2018-09-21 22:16:34,933 - root - [Epoch 1 Batch 960/1043] loss=3.4178, ppl=30.5035, gnorm=0.3296, throughput=19.94K wps, wc=60.05K\n2018-09-21 22:16:38,072 - root - [Epoch 1 Batch 970/1043] loss=3.4463, ppl=31.3847, gnorm=0.3454, throughput=19.44K wps, wc=60.95K\n2018-09-21 22:16:40,342 - root - [Epoch 1 Batch 980/1043] loss=3.2040, ppl=24.6319, gnorm=0.3692, throughput=17.23K wps, wc=39.06K\n2018-09-21 22:16:43,231 - root - [Epoch 1 Batch 990/1043] loss=3.3610, ppl=28.8194, gnorm=0.3210, throughput=17.97K wps, wc=51.84K\n2018-09-21 22:16:45,733 - root - [Epoch 1 Batch 1000/1043] loss=3.1590, ppl=23.5477, gnorm=0.3982, throughput=17.61K wps, wc=43.98K\n2018-09-21 22:16:48,528 - root - [Epoch 1 Batch 1010/1043] loss=3.2728, ppl=26.3860, gnorm=0.3529, throughput=18.05K wps, wc=50.37K\n2018-09-21 22:16:51,822 - root - [Epoch 1 Batch 1020/1043] loss=3.3868, ppl=29.5702, gnorm=0.3414, throughput=19.88K wps, wc=65.44K\n2018-09-21 22:16:54,571 - root - [Epoch 1 Batch 1030/1043] loss=3.2690, ppl=26.2849, gnorm=0.3682, throughput=18.28K wps, wc=50.17K\n2018-09-21 22:16:56,752 - root - [Epoch 1 Batch 1040/1043] loss=3.0769, ppl=21.6901, gnorm=0.3846, throughput=17.32K wps, wc=37.72K\n2018-09-21 22:17:22,479 - root - [Epoch 1] valid Loss=2.4313, valid ppl=11.3741, valid bleu=8.90\n2018-09-21 22:17:44,531 - root - [Epoch 1] test Loss=2.4986, test ppl=12.1659, test bleu=9.56\n2018-09-21 22:17:44,544 - root - Save best parameters to gnmt_en_vi_u512/valid_best.params\n2018-09-21 22:17:44,912 - root - Learning rate change to 0.00025\n"
 }
]
```

## Summary
In this notebook, we have shown how to train a GNMT model on IWSLT 2015 English-Vietnamese using Gluon NLP toolkit. 
The complete training script can be found [here](https://github.com/dmlc/gluon-nlp/blob/master/scripts/nmt/train_gnmt.py). 
The command to reproduce the result can be seen in the [nmt scripts page](http://gluon-nlp.mxnet.io/scripts/index.html#machine-translation).
