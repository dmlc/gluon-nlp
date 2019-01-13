# Fine-tuning Sentence Pair Classification with BERT

Pre-trained language
representations have been shown to improve many downstream NLP tasks such as
question answering, and natural language inference. To apply pre-trained
representations to these tasks, there are two strategies:

1. **feature-based** approach, which uses the pre-trained representations as additional
features to the downstream task.
2. **fine-tuning** based approach, which trains the downstream tasks by
fine-tuning pre-trained parameters.

While feature-based
approaches such as ELMo [3] (introduced in the previous tutorial) are effective
in improving many downstream tasks, they require task-specific architectures.
Devlin, Jacob, et al proposed BERT [1] (Bidirectional Encoder Representations
from Transformers), which **fine-tunes** deep bidirectional representations on a
wide range of tasks with minimal task-specific parameters, and obtained state-
of-the-art results.

In this tutorial, we will focus on fine-tuning with the
pre-trained BERT model to classify semantically equivalent sentence pairs.
Specifically, we will:

1. load the state-of-the-art pre-trained BERT model.
2.
process and transform sentence pair data to be used for fine-tuning.
3. fine-
tune BERT model for sentence classification.

## Preparation

To run this tutorial locally, please [install gluonnlp](http://gluon-nlp.mxnet.io/#installation)
and click the download button at the top of the tutorial page to get all related code.

Then we start with some usual preparation such as importing libraries
and setting the environment.

### Load MXNet and GluonNLP

```{.python .input}
import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
```

### Set Environment

```{.python .input}
np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
ctx = mx.gpu(0)
```

## Use the Pre-trained BERT model

The list of pre-trained BERT model available in GluonNLP can be found
[here](../../model_zoo/bert/index.rst).

In this tutorial, we will load the BERT
BASE model trained on uncased book corpus and English Wikipedia dataset in
GluonNLP model zoo.

### Get BERT

Let's first take a look at the BERT model
architecture for sentence pair classification below:

<div style="width:
500px;">![bert-sentence-pair](bert-sentence-pair.png)</div>

where the model takes a pair of
sequences and **pools** the representation of the first token in the sequence.
Note that the original BERT model was trained for masked language model and next
sentence prediction tasks, which includes layers for language model decoding and
classification and are not useful for sentence pair classification.

We load the
pre-trained BERT using the model API in GluonNLP, which returns the vocabulary
along with the model. To include the pooler layer of the pre-trained model,
`use_pooler` is set to `True`.

```{.python .input}
from bert import *

bert_base, vocabulary = nlp.model.get_model('bert_12_768_12', 
                                             dataset_name='book_corpus_wiki_en_uncased',
                                             pretrained=True, ctx=ctx, use_pooler=True,
                                             use_decoder=False, use_classifier=False)
print(bert_base)
```

### Model Definition for Sentence Pair Classification

Now that we have loaded
the BERT model, we only need to attach an additional layer for classification.
The `BERTClassifier` class uses a BERT base model to encode sentence
representation, followed by a `nn.Dense` layer for classification.

```{.python .input}
model = bert.BERTClassifier(bert_base, num_classes=2, dropout=0.1)
# only need to initialize the classifier layer.
model.classifier.initialize(init=mx.init.Normal(0.02), ctx=ctx)
model.hybridize(static_alloc=True)

# softmax cross entropy loss for classification
loss_function = gluon.loss.SoftmaxCELoss()
loss_function.hybridize(static_alloc=True)

metric = mx.metric.Accuracy()
```

## Data Preprocessing for BERT


### Dataset

In this tutorial, for demonstration we use the dev set of the
Microsoft Research Paraphrase Corpus dataset. Each example in the dataset
contains a pair of sentences, and a label indicating whether the two sentences
are semantically equivalent. 

Let's take a look at the 3rd example in the
dataset:

```{.python .input}
data_train = dataset.MRPCDataset('dev', root='.')
sample_id = 0
# sentence a
print(data_train[sample_id][0])
# sentence b
print(data_train[sample_id][1])
# 1 means equivalent, 0 means not equivalent
print(data_train[sample_id][2])
```

To use the pre-trained BERT model, we need to preprocess the data in the same
way it was trained. The following figure shows the input representation in BERT:
<div style="width: 500px;">![bert-embed](bert-embed.png)</div>

We will use
`ClassificationTransform` to perform the following transformations:
- tokenize
the input sequences
- insert [CLS], [SEP] as necessary
- generate segment ids to
indicate whether a token belongs to the first sequence or the second sequence.
-
generate valid length

```{.python .input}
# use the vocabulary from pre-trained model for tokenization
bert_tokenizer = tokenizer.FullTokenizer(vocabulary, do_lower_case=True)
# maximum sequence length
max_len = 128
all_labels = ["0", "1"]
transform = dataset.ClassificationTransform(bert_tokenizer, all_labels, max_len)
data_train = data_train.transform(transform)

print('token ids = \n%s'%data_train[sample_id][0])
print('valid length = \n%s'%data_train[sample_id][1])
print('segment ids = \n%s'%data_train[sample_id][2])
print('label = \n%s'%data_train[sample_id][3])
```

## Fine-tune BERT Model

Putting everything together, now we can fine-tune the
model with a few epochs. For demonstration, we use a fixed learning rate and
skip validation steps.

```{.python .input}
batch_size = 32
lr = 5e-6
train_sampler = nlp.data.FixedBucketSampler(lengths=[int(item[1]) for item in data_train],
                                            batch_size=batch_size,
                                            shuffle=True)
bert_dataloader = mx.gluon.data.DataLoader(data_train, batch_sampler=train_sampler)

trainer = gluon.Trainer(model.collect_params(), 'adam',
                        {'learning_rate': lr, 'epsilon': 1e-9})

# collect all differentiable parameters
# grad_req == 'null' indicates no gradients are calculated (e.g. constant parameters)
# the gradients for these params are clipped later
params = [p for p in model.collect_params().values() if p.grad_req != 'null']
grad_clip = 1

log_interval = 4
num_epochs = 3
for epoch_id in range(num_epochs):
    metric.reset()
    step_loss = 0
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(bert_dataloader):
        with mx.autograd.record():
            
            # load data to GPU
            token_ids = token_ids.as_in_context(ctx)
            valid_length = valid_length.as_in_context(ctx)
            segment_ids = segment_ids.as_in_context(ctx)
            label = label.as_in_context(ctx)
            
            # forward computation
            out = model(token_ids, segment_ids, valid_length.astype('float32'))
            ls = loss_function(out, label).mean()
            
        # backward computation
        ls.backward()
        
        # gradient clipping
        trainer.allreduce_grads()
        nlp.utils.clip_grad_global_norm(params, 1)
        trainer.update(1)

        step_loss += ls.asscalar()
        metric.update([label], [out])
        if (batch_id + 1) % (log_interval) == 0:
            print('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f}'
                         .format(epoch_id, batch_id + 1, len(bert_dataloader),
                                 step_loss / log_interval,
                                 trainer.learning_rate, metric.get()[1]))
            step_loss = 0
```

## Conclusion

In this tutorial, we show how to fine-tune a sentence pair
classification model with pre-trained BERT parameters. In GluonNLP, this can be
done with just a few simple steps: apply BERT-style data transformation to
preprocess the data, automatically download the pre-trained model, and feed the
transformed data into the model. For demonstration purpose, we skipped the warmup learning rate
schedule and validation on dev dataset used in the original implementation. Please visit
[here](../../model_zoo/bert/index.rst) for the complete fine-tuning scripts.

## References

[1] Devlin, Jacob, et al. "Bert: Pre-training of deep
bidirectional transformers for language understanding." arXiv preprint
arXiv:1810.04805 (2018).

[2] Dolan, William B., and Chris Brockett.
"Automatically constructing a corpus of sentential paraphrases." Proceedings of
the Third International Workshop on Paraphrasing (IWP2005). 2005.

[3] Peters,
Matthew E., et al. "Deep contextualized word representations." arXiv preprint
arXiv:1802.05365 (2018).
