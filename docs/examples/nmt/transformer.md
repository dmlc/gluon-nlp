# Transformer based Machine Translation Using GluonNLP

In this notebook, we will show how to train Transformer introduced in [1] and evaluate the pretrained model using GluonNLP. The model is both more accurate and lighter to train than previous seq2seq models. We will together go through: 

1) Use the state-of-the-art pretrained Transformer model: we will evaluate the pretrained SOTA Transformer model and translate a few sentences ourselves with the `BeamSearchTranslator` using the SOTA model; 

2) Train the Transformer yourself: including loading and processing dataset, define the Transformer model, write train script and evaluate the trained model. Note that in order to obtain the state-of-the-art results on WMT 2014 English-German dataset, it will take around 1 day to have the model. In order to let you run through the Transformer quickly, we suggest you to start with the `TOY` dataset sampled from the WMT dataset (by default in this notebook).

## Preparation

### Load MXNet and GluonNLP

```{.python .input  n=1}
import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
```

### Set Environment

```{.python .input  n=2}
np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
ctx = mx.gpu(0)
```

## Use the SOTA Pretrained Transformer model

In this subsection, we first load the SOTA Transformer model in GluonNLP model zoo; and secondly we load the full WMT 2014 English-German test dataset; and finally evaluate the model.

### Get the SOTA Transformer

Next, we load the pretrained SOTA Transformer using the model API in GluonNLP. In this way, we can easily get access to the SOTA machine translation model and use it in your own application.

```{.python .input  n=3}
import nmt

wmt_model_name = 'transformer_en_de_512'

wmt_transformer_model, wmt_src_vocab, wmt_tgt_vocab = \
    nmt.transformer.get_model(wmt_model_name, 
                              dataset_name='WMT2014', 
                              pretrained=True, 
                              ctx=ctx)

print(wmt_src_vocab)
print(wmt_tgt_vocab)
```

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Vocab(size=36794, unk=\"<unk>\", reserved=\"['<eos>', '<bos>', '<eos>']\")\nVocab(size=36794, unk=\"<unk>\", reserved=\"['<eos>', '<bos>', '<eos>']\")\n"
 }
]
```

The Transformer model architecture is shown as below:

<div style="width: 500px;">![transformer](transformer.png)</div>

```{.python .input  n=4}
print(wmt_transformer_model)
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "NMTModel(\n  (encoder): TransformerEncoder(\n    (dropout_layer): Dropout(p = 0.1, axes=())\n    (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n    (transformer_cells): HybridSequential(\n      (0): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(512 -> 512, linear)\n          (proj_key): Dense(512 -> 512, linear)\n          (proj_value): Dense(512 -> 512, linear)\n        )\n        (proj): Dense(512 -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(512 -> 2048, Activation(relu))\n          (ffn_2): Dense(2048 -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n      )\n      (1): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(512 -> 512, linear)\n          (proj_key): Dense(512 -> 512, linear)\n          (proj_value): Dense(512 -> 512, linear)\n        )\n        (proj): Dense(512 -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(512 -> 2048, Activation(relu))\n          (ffn_2): Dense(2048 -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n      )\n      (2): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(512 -> 512, linear)\n          (proj_key): Dense(512 -> 512, linear)\n          (proj_value): Dense(512 -> 512, linear)\n        )\n        (proj): Dense(512 -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(512 -> 2048, Activation(relu))\n          (ffn_2): Dense(2048 -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n      )\n      (3): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(512 -> 512, linear)\n          (proj_key): Dense(512 -> 512, linear)\n          (proj_value): Dense(512 -> 512, linear)\n        )\n        (proj): Dense(512 -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(512 -> 2048, Activation(relu))\n          (ffn_2): Dense(2048 -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n      )\n      (4): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(512 -> 512, linear)\n          (proj_key): Dense(512 -> 512, linear)\n          (proj_value): Dense(512 -> 512, linear)\n        )\n        (proj): Dense(512 -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(512 -> 2048, Activation(relu))\n          (ffn_2): Dense(2048 -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n      )\n      (5): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(512 -> 512, linear)\n          (proj_key): Dense(512 -> 512, linear)\n          (proj_value): Dense(512 -> 512, linear)\n        )\n        (proj): Dense(512 -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(512 -> 2048, Activation(relu))\n          (ffn_2): Dense(2048 -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n      )\n    )\n  )\n  (decoder): TransformerDecoder(\n    (dropout_layer): Dropout(p = 0.1, axes=())\n    (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n    (transformer_cells): HybridSequential(\n      (0): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(512 -> 512, linear)\n          (proj_key): Dense(512 -> 512, linear)\n          (proj_value): Dense(512 -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(512 -> 512, linear)\n          (proj_key): Dense(512 -> 512, linear)\n          (proj_value): Dense(512 -> 512, linear)\n        )\n        (proj_in): Dense(512 -> 512, linear)\n        (proj_inter): Dense(512 -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(512 -> 2048, Activation(relu))\n          (ffn_2): Dense(2048 -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n      )\n      (1): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(512 -> 512, linear)\n          (proj_key): Dense(512 -> 512, linear)\n          (proj_value): Dense(512 -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(512 -> 512, linear)\n          (proj_key): Dense(512 -> 512, linear)\n          (proj_value): Dense(512 -> 512, linear)\n        )\n        (proj_in): Dense(512 -> 512, linear)\n        (proj_inter): Dense(512 -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(512 -> 2048, Activation(relu))\n          (ffn_2): Dense(2048 -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n      )\n      (2): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(512 -> 512, linear)\n          (proj_key): Dense(512 -> 512, linear)\n          (proj_value): Dense(512 -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(512 -> 512, linear)\n          (proj_key): Dense(512 -> 512, linear)\n          (proj_value): Dense(512 -> 512, linear)\n        )\n        (proj_in): Dense(512 -> 512, linear)\n        (proj_inter): Dense(512 -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(512 -> 2048, Activation(relu))\n          (ffn_2): Dense(2048 -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n      )\n      (3): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(512 -> 512, linear)\n          (proj_key): Dense(512 -> 512, linear)\n          (proj_value): Dense(512 -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(512 -> 512, linear)\n          (proj_key): Dense(512 -> 512, linear)\n          (proj_value): Dense(512 -> 512, linear)\n        )\n        (proj_in): Dense(512 -> 512, linear)\n        (proj_inter): Dense(512 -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(512 -> 2048, Activation(relu))\n          (ffn_2): Dense(2048 -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n      )\n      (4): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(512 -> 512, linear)\n          (proj_key): Dense(512 -> 512, linear)\n          (proj_value): Dense(512 -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(512 -> 512, linear)\n          (proj_key): Dense(512 -> 512, linear)\n          (proj_value): Dense(512 -> 512, linear)\n        )\n        (proj_in): Dense(512 -> 512, linear)\n        (proj_inter): Dense(512 -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(512 -> 2048, Activation(relu))\n          (ffn_2): Dense(2048 -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n      )\n      (5): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(512 -> 512, linear)\n          (proj_key): Dense(512 -> 512, linear)\n          (proj_value): Dense(512 -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(512 -> 512, linear)\n          (proj_key): Dense(512 -> 512, linear)\n          (proj_value): Dense(512 -> 512, linear)\n        )\n        (proj_in): Dense(512 -> 512, linear)\n        (proj_inter): Dense(512 -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(512 -> 2048, Activation(relu))\n          (ffn_2): Dense(2048 -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=512)\n      )\n    )\n  )\n  (src_embed): HybridSequential(\n    (0): Embedding(36794 -> 512, float32)\n    (1): Dropout(p = 0.0, axes=())\n  )\n  (tgt_embed): HybridSequential(\n    (0): Embedding(36794 -> 512, float32)\n    (1): Dropout(p = 0.0, axes=())\n  )\n  (tgt_proj): Dense(512 -> 36794, linear)\n)\n"
 }
]
```

### Load  and Preprocess WMT 2014 Dataset

We then load the WMT 2014 English-German test dataset for evaluation purpose.

The following shows how to process the dataset and cache the processed dataset
for the future use. The processing steps include: 

* 1) clip the source and target sequences 
* 2) split the string input to a list of tokens
* 3) map the string token into its index in the vocabulary
* 4) append EOS token to source sentence and add BOS and EOS tokens to target sentence.

Let's first look at the WMT 2014 corpus.

```{.python .input  n=5}
import hyperparameters as hparams

wmt_data_test = nlp.data.WMT2014BPE('newstest2014',
                                    src_lang=hparams.src_lang,
                                    tgt_lang=hparams.tgt_lang,
                                    full=False)
print('Source language %s, Target language %s' % (hparams.src_lang, hparams.tgt_lang))

wmt_data_test[0]
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Source language en, Target language de\n"
 },
 {
  "data": {
   "text/plain": "('Or@@ land@@ o Blo@@ om and Mir@@ anda Ker@@ r still love each other',\n 'Or@@ land@@ o Blo@@ om und Mir@@ anda Ker@@ r lieben sich noch immer')"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=6}
wmt_test_text = nlp.data.WMT2014('newstest2014', 
                                 src_lang=hparams.src_lang, 
                                 tgt_lang=hparams.tgt_lang,
                                 full=False)
wmt_test_text[0]
```

```{.json .output n=6}
[
 {
  "data": {
   "text/plain": "('Orlando Bloom and Miranda Kerr still love each other',\n 'Orlando Bloom und Miranda Kerr lieben sich noch immer')"
  },
  "execution_count": 6,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

We then generate the target gold translations.

```{.python .input  n=7}
wmt_test_tgt_sentences = list(wmt_test_text.transform(lambda src, tgt: tgt))
wmt_test_tgt_sentences[0]
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "'Orlando Bloom und Miranda Kerr lieben sich noch immer'"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=8}
import dataprocessor

print(dataprocessor.TrainValDataTransform.__doc__)
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Transform the machine translation dataset.\n\n    Clip source and the target sentences to the maximum length. For the source sentence, append the\n    EOS. For the target sentence, append BOS and EOS.\n\n    Parameters\n    ----------\n    src_vocab : Vocab\n    tgt_vocab : Vocab\n    src_max_len : int\n    tgt_max_len : int\n    \n"
 }
]
```

```{.python .input  n=9}
wmt_transform_fn = dataprocessor.TrainValDataTransform(wmt_src_vocab, wmt_tgt_vocab, -1, -1)
wmt_dataset_processed = wmt_data_test.transform(wmt_transform_fn, lazy=False)
print(*wmt_dataset_processed[0], sep='\n')
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[ 7300 21964 23833  1935 24004 11836  6698 11839  5565 25464 27950 22544\n 16202 24272     3]\n[    2  7300 21964 23833  1935 24004 29615  6698 11839  5565 25464 22297\n 27121 23712 20558     3]\n"
 }
]
```

### Create Sampler and DataLoader for TOY Dataset

```{.python .input  n=10}
wmt_data_test_with_len = gluon.data.SimpleDataset([(ele[0], ele[1], len(
    ele[0]), len(ele[1]), i) for i, ele in enumerate(wmt_dataset_processed)])
```

Now, we have obtained data_train, data_val, and data_test. The next step is to construct sampler and DataLoader. The first step is to construct batchify function, which pads and stacks sequences to form mini-batch.

```{.python .input  n=11}
wmt_test_batchify_fn = nlp.data.batchify.Tuple(
    nlp.data.batchify.Pad(),
    nlp.data.batchify.Pad(),
    nlp.data.batchify.Stack(dtype='float32'),
    nlp.data.batchify.Stack(dtype='float32'),
    nlp.data.batchify.Stack())
```

We can then construct bucketing samplers, which generate batches by grouping sequences with similar lengths.

```{.python .input  n=12}
wmt_bucket_scheme = nlp.data.ExpWidthBucket(bucket_len_step=1.2)
```

```{.python .input  n=13}
wmt_test_batch_sampler = nlp.data.FixedBucketSampler(
    lengths=wmt_dataset_processed.transform(lambda src, tgt: len(tgt)),
    use_average_length=True,
    bucket_scheme=wmt_bucket_scheme,
    batch_size=256)
print(wmt_test_batch_sampler.stats())
```

```{.json .output n=13}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "FixedBucketSampler:\n  sample_num=2737, batch_num=364\n  key=[10, 14, 19, 25, 33, 42, 53, 66, 81, 100]\n  cnt=[101, 243, 386, 484, 570, 451, 280, 172, 41, 9]\n  batch_size=[25, 18, 13, 10, 8, 6, 5, 4, 3, 2]\n"
 }
]
```

Given the samplers, we can create DataLoader, which is iterable.

```{.python .input  n=14}
wmt_test_data_loader = gluon.data.DataLoader(
    wmt_data_test_with_len,
    batch_sampler=wmt_test_batch_sampler,
    batchify_fn=wmt_test_batchify_fn,
    num_workers=8)
len(wmt_test_data_loader)
```

```{.json .output n=14}
[
 {
  "data": {
   "text/plain": "364"
  },
  "execution_count": 14,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### Evaluate Transformer

Next, we generate the SOTA results on the WMT test dataset. As we can see from the result, we are able to achieve the SOTA number 27.35 as the BLEU score.


We first define the `BeamSearchTranslator` to generate the actual translations.

```{.python .input  n=15}
wmt_translator = nmt.translation.BeamSearchTranslator(
    model=wmt_transformer_model,
    beam_size=hparams.beam_size, 
    scorer=nlp.model.BeamSearchScorer(alpha=hparams.lp_alpha, K=hparams.lp_k),
    max_length=200)
```

Then we caculate the `loss` as well as the `bleu` score on the WMT 2014 English-German test dataset. Note that the following evalution process will take ~13 mins to complete.

```{.python .input  n=16}
import time
import utils

eval_start_time = time.time()

wmt_test_loss_function = nmt.loss.SoftmaxCEMaskedLoss()
wmt_test_loss_function.hybridize()

wmt_detokenizer = nlp.data.SacreMosesDetokenizer()

wmt_test_loss, wmt_test_translation_out = utils.evaluate(wmt_transformer_model,
                                                         wmt_test_data_loader,
                                                         wmt_test_loss_function,
                                                         wmt_translator,
                                                         wmt_tgt_vocab,
                                                         wmt_detokenizer,
                                                         ctx)

wmt_test_bleu_score, _, _, _, _ = nmt.bleu.compute_bleu([wmt_test_tgt_sentences], 
                                                        wmt_test_translation_out,
                                                        tokenized=False, 
                                                        tokenizer=hparams.bleu,
                                                        split_compound_word=False,
                                                        bpe=False)

print('WMT14 EN-DE SOTA model test loss: %.2f; test bleu score: %.2f; time cost %.2fs'
      %(wmt_test_loss, wmt_test_bleu_score * 100, (time.time() - eval_start_time)))
```

```{.json .output n=16}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "WMT14 EN-DE SOTA model test loss: 1.21; test bleu score: 27.35; time cost 782.25s\n"
 }
]
```

```{.python .input  n=17}
print('Sample translations:')
num_pairs = 3

for i in range(num_pairs):
    print('EN:')
    print(wmt_test_text[i][0])
    print('DE-Candidate:')
    print(wmt_test_translation_out[i])
    print('DE-Reference:')
    print(wmt_test_tgt_sentences[i])
    print('========')
```

```{.json .output n=17}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Sample translations:\nEN:\nOrlando Bloom and Miranda Kerr still love each other\nDE-Candidate:\nOrlando Bloom und Miranda Kerr lieben einander immer noch.\nDE-Reference:\nOrlando Bloom und Miranda Kerr lieben sich noch immer\n========\nEN:\nActors Orlando Bloom and Model Miranda Kerr want to go their separate ways.\nDE-Candidate:\nSchauspieler Orlando Bloom und Model Miranda Kerr wollen ihre eigenen Wege gehen.\nDE-Reference:\nSchauspieler Orlando Bloom und Model Miranda Kerr wollen k\u00fcnftig getrennte Wege gehen.\n========\nEN:\nHowever, in an interview, Bloom has said that he and Kerr still love each other.\nDE-Candidate:\nIn einem Interview hat Bloom jedoch gesagt, dass er und Kerr sich immer noch lieben.\nDE-Reference:\nIn einem Interview sagte Bloom jedoch, dass er und Kerr sich noch immer lieben.\n========\n"
 }
]
```

### Translation Inference

We herein show the actual translation example (EN-DE) when given a source language using the SOTA Transformer model.

```{.python .input  n=18}
import utils

print('Translate the following English sentence into German:')

sample_src_seq = 'We love each other'

print('[\'' + sample_src_seq + '\']')
    
sample_tgt_seq = utils.translate(wmt_translator, 
                                 sample_src_seq, 
                                 wmt_src_vocab, 
                                 wmt_tgt_vocab, 
                                 wmt_detokenizer,
                                 ctx)

print('The German translation is:')
print(sample_tgt_seq)
```

```{.json .output n=18}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Translate the following English sentence into German:\n['We love each other']\nThe German translation is:\n['Wir lieben einander']\n"
 }
]
```

## Train Your Own Transformer

In this subsection, we will go though the whole process about loading translation dataset in a more unified way, and create data sampler and loader, as well as define the Transformer model, finally writing training script to train the model yourself.

### Load and Preprocess TOY Dataset

Note that we use demo mode (`TOY` dataset) by default, since loading the whole WMT 2014 English-German dataset `WMT2014BPE` for the later training will be slow (~1 day). But if you really want to train to have the SOTA result, please set `demo = False`. In order to make the data processing blocks execute in a more efficient way, we package them in the `load_translation_data` (`transform` etc.) function used as below. The function also returns the gold target sentences as well as the vocabularies.

```{.python .input  n=19}
demo = True
if demo:
    dataset = 'TOY'
else:
    dataset = 'WMT2014BPE'

data_train, data_val, data_test, val_tgt_sentences, test_tgt_sentences, src_vocab, tgt_vocab = \
    dataprocessor.load_translation_data(
        dataset=dataset, 
        src_lang=hparams.src_lang, 
        tgt_lang=hparams.tgt_lang)
    
data_train_lengths = dataprocessor.get_data_lengths(data_train)
data_val_lengths = dataprocessor.get_data_lengths(data_val)
data_test_lengths = dataprocessor.get_data_lengths(data_test)

data_train = data_train.transform(lambda src, tgt: (src, tgt, len(src), len(tgt)), lazy=False)
data_val = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                          for i, ele in enumerate(data_val)])
data_test = gluon.data.SimpleDataset([(ele[0], ele[1], len(ele[0]), len(ele[1]), i)
                           for i, ele in enumerate(data_test)])
```

```{.json .output n=19}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Loading dataset...\nLoading dataset...\nLoading dataset...\n"
 }
]
```

### Create Sampler and DataLoader for TOY Dataset

Now, we have obtained `data_train`, `data_val`, and `data_test`. The next step
is to construct sampler and DataLoader. The first step is to construct batchify
function, which pads and stacks sequences to form mini-batch.

```{.python .input  n=20}
train_batchify_fn = nlp.data.batchify.Tuple(
    nlp.data.batchify.Pad(), 
    nlp.data.batchify.Pad(),
    nlp.data.batchify.Stack(dtype='float32'), 
    nlp.data.batchify.Stack(dtype='float32'))
test_batchify_fn = nlp.data.batchify.Tuple(
    nlp.data.batchify.Pad(), 
    nlp.data.batchify.Pad(),
    nlp.data.batchify.Stack(dtype='float32'), 
    nlp.data.batchify.Stack(dtype='float32'),
    nlp.data.batchify.Stack())

target_val_lengths = list(map(lambda x: x[-1], data_val_lengths))
target_test_lengths = list(map(lambda x: x[-1], data_test_lengths))
```

We can then construct bucketing samplers, which generate batches by grouping
sequences with similar lengths.

```{.python .input  n=21}
bucket_scheme = nlp.data.ExpWidthBucket(bucket_len_step=1.2)
train_batch_sampler = nlp.data.FixedBucketSampler(lengths=data_train_lengths,
                                             batch_size=hparams.batch_size,
                                             num_buckets=hparams.num_buckets,
                                             ratio=0.0,
                                             shuffle=True,
                                             use_average_length=True,
                                             num_shards=1,
                                             bucket_scheme=bucket_scheme)
print('Train Batch Sampler:')
print(train_batch_sampler.stats())


val_batch_sampler = nlp.data.FixedBucketSampler(lengths=target_val_lengths,
                                       batch_size=hparams.test_batch_size,
                                       num_buckets=hparams.num_buckets,
                                       ratio=0.0,
                                       shuffle=False,
                                       use_average_length=True,
                                       bucket_scheme=bucket_scheme)
print('Validation Batch Sampler:')
print(val_batch_sampler.stats())

test_batch_sampler = nlp.data.FixedBucketSampler(lengths=target_test_lengths,
                                        batch_size=hparams.test_batch_size,
                                        num_buckets=hparams.num_buckets,
                                        ratio=0.0,
                                        shuffle=False,
                                        use_average_length=True,
                                        bucket_scheme=bucket_scheme)
print('Test Batch Sampler:')
print(test_batch_sampler.stats())
```

```{.json .output n=21}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Train Batch Sampler:\nFixedBucketSampler:\n  sample_num=30, batch_num=14\n  key=[(10, 11), (12, 12), (13, 13), (14, 15), (15, 16), (17, 18), (18, 20), (20, 22), (22, 24), (28, 31), (32, 36), (36, 41), (41, 47), (48, 55)]\n  cnt=[1, 1, 1, 1, 2, 2, 2, 4, 1, 2, 5, 1, 4, 3]\n  batch_size=[245, 225, 207, 180, 172, 155, 139, 136, 117, 89, 81, 72, 63, 53]\nValidation Batch Sampler:\nFixedBucketSampler:\n  sample_num=30, batch_num=14\n  key=[10, 11, 13, 15, 16, 18, 20, 22, 24, 31, 36, 41, 47, 55]\n  cnt=[1, 1, 1, 2, 3, 3, 2, 1, 1, 4, 3, 5, 2, 1]\n  batch_size=[25, 23, 19, 17, 16, 14, 13, 12, 11, 8, 7, 6, 5, 4]\nTest Batch Sampler:\nFixedBucketSampler:\n  sample_num=30, batch_num=14\n  key=[10, 11, 13, 15, 16, 18, 20, 22, 24, 31, 36, 41, 47, 55]\n  cnt=[1, 1, 1, 2, 3, 3, 2, 1, 1, 4, 3, 5, 2, 1]\n  batch_size=[25, 23, 19, 17, 16, 14, 13, 12, 11, 8, 7, 6, 5, 4]\n"
 }
]
```

Given the samplers, we can create DataLoader, which is iterable. Note that the data loader of validation and test dataset share the same batchifying function `test_batchify_fn`.

```{.python .input  n=22}
train_data_loader = nlp.data.ShardedDataLoader(data_train,
                                      batch_sampler=train_batch_sampler,
                                      batchify_fn=train_batchify_fn,
                                      num_workers=8)
print('Length of train_data_loader: %d' % len(train_data_loader))
val_data_loader = gluon.data.DataLoader(data_val,
                             batch_sampler=val_batch_sampler,
                             batchify_fn=test_batchify_fn,
                             num_workers=8)
print('Length of val_data_loader: %d' % len(val_data_loader))
test_data_loader = gluon.data.DataLoader(data_test,
                              batch_sampler=test_batch_sampler,
                              batchify_fn=test_batchify_fn,
                              num_workers=8)
print('Length of test_data_loader: %d' % len(test_data_loader))
```

```{.json .output n=22}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Length of train_data_loader: 14\nLength of val_data_loader: 14\nLength of test_data_loader: 14\n"
 }
]
```

### Define Transformer Model

After obtaining DataLoader, we then start to define the Transformer. The encoder and decoder of the Transformer
can be easily obtained by calling `get_transformer_encoder_decoder` function. Then, we
use the encoder and decoder in `NMTModel` to construct the Transformer model.
`model.hybridize` allows computation to be done using symbolic backend. We also use `label_smoothing`.

```{.python .input  n=23}
encoder, decoder = nmt.transformer.get_transformer_encoder_decoder(units=hparams.num_units,
                                                   hidden_size=hparams.hidden_size,
                                                   dropout=hparams.dropout,
                                                   num_layers=hparams.num_layers,
                                                   num_heads=hparams.num_heads,
                                                   max_src_length=530,
                                                   max_tgt_length=549,
                                                   scaled=hparams.scaled)
model = nmt.translation.NMTModel(src_vocab=src_vocab, tgt_vocab=tgt_vocab, encoder=encoder, decoder=decoder,
                 share_embed=True, embed_size=hparams.num_units, tie_weights=True,
                 embed_initializer=None, prefix='transformer_')
model.initialize(init=mx.init.Xavier(magnitude=3.0), ctx=ctx)
model.hybridize()

print(model)

label_smoothing = nmt.loss.LabelSmoothing(epsilon=hparams.epsilon, units=len(tgt_vocab))
label_smoothing.hybridize()

loss_function = nmt.loss.SoftmaxCEMaskedLoss(sparse_label=False)
loss_function.hybridize()

test_loss_function = nmt.loss.SoftmaxCEMaskedLoss()
test_loss_function.hybridize()

detokenizer = nlp.data.SacreMosesDetokenizer()
```

```{.json .output n=23}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "NMTModel(\n  (encoder): TransformerEncoder(\n    (dropout_layer): Dropout(p = 0.1, axes=())\n    (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n    (transformer_cells): HybridSequential(\n      (0): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (1): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (2): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (3): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (4): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (5): TransformerEncoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n    )\n  )\n  (decoder): TransformerDecoder(\n    (dropout_layer): Dropout(p = 0.1, axes=())\n    (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n    (transformer_cells): HybridSequential(\n      (0): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj_in): Dense(None -> 512, linear)\n        (proj_inter): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (1): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj_in): Dense(None -> 512, linear)\n        (proj_inter): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (2): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj_in): Dense(None -> 512, linear)\n        (proj_inter): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (3): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj_in): Dense(None -> 512, linear)\n        (proj_inter): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (4): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj_in): Dense(None -> 512, linear)\n        (proj_inter): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n      (5): TransformerDecoderCell(\n        (dropout_layer): Dropout(p = 0.1, axes=())\n        (attention_cell_in): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (attention_cell_inter): MultiHeadAttentionCell(\n          (_base_cell): DotProductAttentionCell(\n            (_dropout_layer): Dropout(p = 0.1, axes=())\n          )\n          (proj_query): Dense(None -> 512, linear)\n          (proj_key): Dense(None -> 512, linear)\n          (proj_value): Dense(None -> 512, linear)\n        )\n        (proj_in): Dense(None -> 512, linear)\n        (proj_inter): Dense(None -> 512, linear)\n        (ffn): PositionwiseFFN(\n          (ffn_1): Dense(None -> 2048, Activation(relu))\n          (ffn_2): Dense(None -> 512, linear)\n          (dropout_layer): Dropout(p = 0.1, axes=())\n          (layer_norm): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        )\n        (layer_norm_in): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n        (layer_norm_inter): LayerNorm(eps=1e-05, axis=-1, center=True, scale=True, in_channels=0)\n      )\n    )\n  )\n  (src_embed): HybridSequential(\n    (0): Embedding(358 -> 512, float32)\n    (1): Dropout(p = 0.0, axes=())\n  )\n  (tgt_embed): HybridSequential(\n    (0): Embedding(358 -> 512, float32)\n    (1): Dropout(p = 0.0, axes=())\n  )\n  (tgt_proj): Dense(None -> 381, linear)\n)\n"
 }
]
```

Here, we build the translator using the beam search

```{.python .input  n=24}
translator = nmt.translation.BeamSearchTranslator(model=model, 
                                                  beam_size=hparams.beam_size,
                                                  scorer=nlp.model.BeamSearchScorer(alpha=hparams.lp_alpha, 
                                                                                    K=hparams.lp_k),
                                                  max_length=200)
print('Use beam_size=%d, alpha=%.2f, K=%d' % (hparams.beam_size, hparams.lp_alpha, hparams.lp_k))
```

```{.json .output n=24}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Use beam_size=4, alpha=0.60, K=5\n"
 }
]
```

### Training Loop

Before conducting training, we need to create trainer for updating the
parameter. In the following example, we create a trainer that uses ADAM
optimzier.

```{.python .input  n=25}
trainer = gluon.Trainer(model.collect_params(), hparams.optimizer,
                        {'learning_rate': hparams.lr, 'beta2': 0.98, 'epsilon': 1e-9})
print('Use learning_rate=%.2f' 
      % (trainer.learning_rate))
```

```{.json .output n=25}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Use learning_rate=2.00\n"
 }
]
```

We can then write the training loop. During the training, we perform the evaluation on validation and testing dataset every epoch, and record the parameters that give the hightest BLEU score on validation dataset. Before performing forward and backward, we first use `as_in_context` function to copy the mini-batch to GPU. The statement `with mx.autograd.record()` will locate Gluon backend to compute the gradients for the part inside the block. For ease of observing the convergence of the update of the `Loss` in a quick fashion, we set the `epochs = 3`. Notice that, in order to obtain the best BLEU score, we will need more epochs and large warmup steps following the original paper as you can find the SOTA results in the first subsection. Besides, we use Averaging SGD [2] to update the parameters, since it is more robust for the machine translation task.

```{.python .input  n=26}
best_valid_loss = float('Inf')
step_num = 0
#We use warmup steps as introduced in [1].
warmup_steps = hparams.warmup_steps
grad_interval = hparams.num_accumulated
model.collect_params().setattr('grad_req', 'add')
#We use Averaging SGD [2] to update the parameters.
average_start = (len(train_data_loader) // grad_interval) * \
    (hparams.epochs - hparams.average_start)
average_param_dict = {k: mx.nd.array([0]) for k, v in
                                      model.collect_params().items()}
update_average_param_dict = True
model.collect_params().zero_grad()
for epoch_id in range(hparams.epochs):
    utils.train_one_epoch(epoch_id, model, train_data_loader, trainer, 
                          label_smoothing, loss_function, grad_interval, 
                          average_param_dict, update_average_param_dict, 
                          step_num, ctx)
    mx.nd.waitall()
    # We define evaluation function as follows. The `evaluate` function use beam search translator 
    # to generate outputs for the validation and testing datasets.
    valid_loss, _ = utils.evaluate(model, val_data_loader,
                                   test_loss_function, translator, 
                                   tgt_vocab, detokenizer, ctx)
    print('Epoch %d, valid Loss=%.4f, valid ppl=%.4f' 
          % (epoch_id, valid_loss, np.exp(valid_loss)))
    test_loss, _ = utils.evaluate(model, test_data_loader,
                                  test_loss_function, translator,
                                  tgt_vocab, detokenizer, ctx)
    print('Epoch %d, test Loss=%.4f, test ppl=%.4f' 
          % (epoch_id, test_loss, np.exp(test_loss)))
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        model.save_parameters('{}.{}'.format(hparams.save_dir, 'valid_best.params'))
    model.save_parameters('{}.epoch{:d}.params'.format(hparams.save_dir, epoch_id))
mx.nd.save('{}.{}'.format(hparams.save_dir, 'average.params'), average_param_dict)

if hparams.average_start > 0:
    for k, v in model.collect_params().items():
        v.set_data(average_param_dict[k])
else:
    model.load_parameters('{}.{}'.format(hparams.save_dir, 'valid_best.params'), ctx)
valid_loss, _ = utils.evaluate(model, val_data_loader, 
                               test_loss_function, translator, 
                               tgt_vocab, detokenizer, ctx)
print('Best model valid Loss=%.4f, valid ppl=%.4f' 
      % (valid_loss, np.exp(valid_loss)))
test_loss, _ = utils.evaluate(model, test_data_loader, 
                              test_loss_function, translator, 
                              tgt_vocab, detokenizer, ctx)
print('Best model test Loss=%.4f, test ppl=%.4f' 
      % (test_loss, np.exp(test_loss)))
```

```{.json .output n=26}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Epoch 0, valid Loss=15.2710, valid ppl=4286389.2761\nEpoch 0, test Loss=15.2710, test ppl=4286389.2761\nEpoch 1, valid Loss=11.3263, valid ppl=82971.9571\nEpoch 1, test Loss=11.3263, test ppl=82971.9571\nEpoch 2, valid Loss=6.7492, valid ppl=853.3337\nEpoch 2, test Loss=6.7492, test ppl=853.3337\nBest model valid Loss=8.7832, valid ppl=6523.6766\nBest model test Loss=8.7832, test ppl=6523.6766\n"
 }
]
```

## Conclusion

- Showcase with Transformer, we are able to support the deep neural networks for seq2seq task. We have already achieved SOTA results on the WMT 2014 English-German task.
- Gluon NLP Toolkit provides high-level APIs that could drastically simplify the development process of modeling for NLP tasks sharing the encoder-decoder structure.
- Low-level APIs in NLP Toolkit enables easy customization.

Documentation can be found at http://gluon-nlp.mxnet.io/index.html

Code is here https://github.com/dmlc/gluon-nlp

## References

[1] Vaswani, Ashish, et al. "Attention is all you need." Advances in Neural Information Processing Systems. 2017.

[2] Polyak, Boris T, and Anatoli B. Juditsky. "Acceleration of stochastic approximation by averaging." SIAM Journal on Control and Optimization. 1992.

