# Extract Sentence Features with Pre-trained ELMo

While word-embeddings have been shown to capture syntactic and semantic information of words and have become a standard component in many state-of-the-art NLP architectures, their context-free nature limits their ability to represent context-dependent information.
Peters et. al. proposed a deep contextualized word representation method, called Embeddings from Language Models, or ELMo [1].
This model is pre-trained with a self-supervising task called bidirectional language model, and they show that the representation from this model is powerful and improves the state-of-the-art on many tasks such as question-answering, natural language inference, semantic role labeling, coreference resolution, named-entity recognition, and sentiment analysis.

In this notebook, we show how to use the model API in GluonNLP to automatically download pre-trained ELMo model, and generate sentence representation with this model.
We will focus on:

1) how to process and transform data to be used with pre-trained ELMo model, and
2) how to load the pretrained ELMo model, and use it to extract representation from preprocessed data.

## Preparation

We start with some usual preparation such as importing libraries and setting the environment.

### Load MXNet and GluonNLP

```{.python .input}
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import io

import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
```

## Preprocess Data

The goal of preprocessing the data is to numericalize the text using the preprocessing steps that are consistent with training ELMo model.
The exact same vocabulary needs to be used so that the indices in model embedding matches the pre-trained model.
We will proceed with the following steps:

1) Load a custom dataset.
2) Tokenize the dataset in the same way as training ELMo.
3) Numericalize the tokens on both words and characters using the provided vocab.

### Create Dataset

The first step is to create a dataset from existing data.
Here, we use a paragraph from [1] as our dataset, using the built-in [TextLineDataset](../../api/modules/data.rst#gluonnlp.data.TextLineDataset) class.
It's a dataset of 7 samples, each being a sentence.

```{.python .input}
elmo_intro = """
Extensive experiments demonstrate that ELMo representations work extremely well in practice.
We first show that they can be easily added to existing models for six diverse and challenging language understanding problems, including textual entailment, question answering and sentiment analysis.
The addition of ELMo representations alone significantly improves the state of the art in every case, including up to 20% relative error reductions.
For tasks where direct comparisons are possible, ELMo outperforms CoVe (McCann et al., 2017), which computes contextualized representations using a neural machine translation encoder.
Finally, an analysis of both ELMo and CoVe reveals that deep representations outperform those derived from just the top layer of an LSTM.
Our trained models and code are publicly available, and we expect that ELMo will provide similar gains for many other NLP problems.
"""

elmo_intro_file = 'elmo_intro.txt'
with io.open(elmo_intro_file, 'w', encoding='utf8') as f:
    f.write(elmo_intro)

dataset = nlp.data.TextLineDataset(elmo_intro_file, 'utf8')
print(len(dataset))
print(dataset[2]) # print a sentence
```

### Transform Dataset

Once we have the dataset that consists of sentences in raw text form, the next step is to transform
the dataset in the same way that ELMo model was trained.
In our case, transforming the dataset consists of tokenization and numericalization.

#### Tokenization

The ELMo pre-trained models are trained on Google 1-Billion Words dataset, which was tokenized with Moses Tokenizer.
In GluonNLP, using either [NLTKMosesTokenizer](../../api/modules/data.rst#gluonnlp.data.NLTKMosesTokenizer) or [SacreMosesTokenizer](../../api/modules/data.rst#gluonnlp.data.SacreMosesTokenizer) should do the trick.
Once tokenized, we can add markers for beginning and end of sentences.

```{.python .input}
tokenizer = nlp.data.NLTKMosesTokenizer()
dataset = dataset.transform(tokenizer)
dataset = dataset.transform(lambda x: ['<bos>'] + x + ['<eos>'])
print(dataset[2]) # print the same tokenized sentence
```


#### Using Vocab from Pre-trained ELMo

Numericalizing the dataset is as straightforward as using the ELMo-specific character-level
vocabulary as transformation. For details on ELMo's vocabulary, see
[ELMoCharVocab](../../api/modules/vocab.rst#gluonnlp.vocab.ELMoCharVocab).
We also calculate the length of each sentence in preparation for batching.

```{.python .input}
vocab = nlp.vocab.ELMoCharVocab()
dataset = dataset.transform(lambda x: (vocab[x], len(x)), lazy=False)
```

#### Create Dataloader

Now that the dataset is ready, loading it with dataloader is straightforward.
Here, we pad the first field to the maximum length, and stack the actual length numbers to form
batches.
The lengths will be used as mask.
For more advanced usage examples of dataloader, check out the
[Sentiment Analysis tutorial](../sentiment_analysis/sentiment_analysis.ipynb).

```{.python .input}
batch_size = 2
dataset_batchify_fn = nlp.data.batchify.Tuple(nlp.data.batchify.Pad(),
                                              nlp.data.batchify.Stack())
data_loader = gluon.data.DataLoader(dataset,
                                    batch_size=batch_size,
                                    batchify_fn=dataset_batchify_fn)
```

## Load Pretrained ELMo Model

Using the model API in GluonNLP, you can automatically download the pre-trained models simply by
calling get_model. Available options are:

1. elmo_2x1024_128_2048cnn_1xhighway
2. elmo_2x2048_256_2048cnn_1xhighway
3. elmo_2x4096_512_2048cnn_2xhighway

Note that the second field in get_model's return value is ELMo's vocabulary.
Since we already created an instance of it above, here we simply ignore this field.

```{.python .input}
elmo_bilm, _ = nlp.model.get_model('elmo_2x1024_128_2048cnn_1xhighway',
                                   dataset_name='gbw',
                                   pretrained=True,
                                   ctx=mx.cpu())
print(elmo_bilm)
```

## Putting everything together

Finally, now we feed the prepared data batch into the [ELMoBiLM](../../api/modules/model.rst#gluonnlp.model.ELMoBiLM) model.
```{.python .input}
def get_features(data, valid_lengths):
    length = data.shape[1]
    hidden_state = elmo_bilm.begin_state(mx.nd.zeros, batch_size=batch_size)
    mask = mx.nd.arange(length).expand_dims(0).broadcast_axes(axis=(0,), size=(batch_size,))
    mask = mask < valid_lengths.expand_dims(1).astype('float32')
    output, hidden_state = elmo_bilm(data, hidden_state, mask)
    return output

batch = next(iter(data_loader))
features = get_features(*batch)
print([x.shape for x in features])
```

## Conclusion

In this tutorial, we show how to generate sentence representation from ELMo model.
In GluonNLP, this can be done with just a few simple steps: reuse the data transformation from ELMo for preprocessing the data, automatically download the pre-trained model, and feed the transformed data into the model.
To see how to plug in the pre-trained models in your own model architecture and use fine-tuning to improve downstream tasks, check our [Sentiment Analysis tutorial](../sentiment_analysis/sentiment_analysis.ipynb).

## Reference
[1] Peters, Matthew E., et al. "Deep contextualized word representations." NAACL (2018).
