# Extract Sentence Features with Pre-trained ELMo

While word-embeddings have been shown to capture syntactic and semantic information of words and have become a standard component in many state-of-the-art NLP architectures, their context-free nature limits their ability to represent context-dependent information.
Peters et. al. proposed a deep contextualized word representation method, called Embeddings from Language Models, or ELMo.
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

import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
import numpy as np
import os
import json
```

## Preprocess Data

The goal of preprocessing the data is to numericalize the text using the preprocessing steps that are consistent with training ELMo model.
The exact same vocabulary needs to be used so that the indices in model embedding matches the pre-trained model.
We will proceed with the following steps:

1) Load a custom dataset.
2) Tokenize the dataset in the same way as training ELMo.
3) Numericalize the tokens on both words and characters using the provided vocab.

### Create Dataset

TODO load dataset (holmes?)

### Transform Dataset

TODO explain the steps of transformation.

#### Tokenization

TODO explain what tokenization was done when pre-training ELMo.

#### Using Vocab from Pre-trained ELMo

TODO explain char-level, word-level vocabs, and conversion to NDArray.

```{.python .input}
class DataListTransform(object):
    """Transform the sample dataset.

    Transform the sample dataset into char_ids and finally to a numpy array.
    """

    def __call__(self, sentence):
        sentence_chars = [np.array(self._convert_word_to_char_ids(token)) for token in sentence]
        sentence_chars_nd = np.stack(sentence_chars, axis=0)
        return sentence_chars_nd

    def _convert_word_to_char_ids(self, word):
        if word == nlp.model.ELMoCharacterVocab.bos_token:
            char_ids = nlp.model.ELMoCharacterVocab.beginning_of_sentence_characters
        elif word == nlp.model.ELMoCharacterVocab.eos_token:
            char_ids = nlp.model.ELMoCharacterVocab.end_of_sentence_characters
        else:
            word_encoded = word.encode('utf-8', 'ignore')[
                           :(nlp.model.ELMoCharacterVocab.max_word_length - 2)]
            char_ids = [nlp.model.ELMoCharacterVocab.padding_character] \
                       * nlp.model.ELMoCharacterVocab.max_word_length
            char_ids[0] = nlp.model.ELMoCharacterVocab.beginning_of_word_character
            for k, chr_id in enumerate(word_encoded, start=1):
                char_ids[k] = chr_id
            char_ids[len(word_encoded) + 1] = nlp.model.ELMoCharacterVocab.end_of_word_character

        # +1 one for masking
        return [c + 1 for c in char_ids]

sample_dataset = sample_dataset.transform(DataListTransform(), lazy=False)
sample_dataset = gluon.data.SimpleDataset([(ele, len(ele), i)
                      for i, ele in enumerate(sample_dataset)])
```

#### Create Dataloader

TODO create batches using dataloader. refer to API notes, sentiment analysis tutorial.

```{.python .input}
import gluonnlp.data.batchify as btf

#create batchify function
sample_dataset_batchify_fn = nlp.data.batchify.Tuple(btf.Pad(), btf.Stack(), btf.Stack())
```

TODO remove interval sampler
```{.python .input}
batch_size = 3
sample_data_loader = gluon.data.DataLoader(sample_dataset,
                                           batch_size=batch_size,
                                           sampler=mx.gluon.contrib.data.sampler.IntervalSampler(len(sample_dataset), interval=int(len(sample_dataset)/batch_size)),
                                           batchify_fn=sample_dataset_batchify_fn,
                                           num_workers=8)
```

## Load Pretrained ELMo Model

TODO explain get_model in model API.

```{.python .input}
elmo_bilm = nlp.model.get_model('elmo_2x1024_128_2048cnn_1xhighway',
                                 dataset_name='gbw',
                                 pretrained=True,
                                 ctx=mx.cpu())
print(elmo_bilm)
```

## Putting everything together


```{.python .input}

hidden_state = elmo_bilm.begin_state(mx.nd.zeros, batch_size=batch_size)

TODO explain removal of begin/end tokens
TODO define a function that returns the feature from the batches in dataloader
TODO print one output
def remove_sentence_boundaries(inputs, mask):
    """
    Remove begin/end of sentence embeddings from the batch of sentences.
    Given a batch of sentences with size ``(batch_size, timesteps, embedding_size)``
    this returns a tensor of shape ``(batch_size, timesteps - 2, embedding_size)`` after removing
    the beginning and end sentence markers.
    Returns both the new tensor and updated mask.

    Parameters
    ----------
    inputs : ``NDArray``
        with shape ``(batch_size, timesteps, embedding_size)``
    mask : ``NDArray``
        with shape ``(batch_size, timesteps)``
    Returns
    -------
    inputs_without_boundary_tokens : ``NDArray``
        with shape ``(batch_size, timesteps - 2, embedding_size)``
    new_mask : ``NDArray``
        The new mask with shape ``(batch_size, timesteps - 2)``.
    """
    sequence_lengths = mask.sum(axis=1).asnumpy()
    inputs_shape = list(inputs.shape)
    new_shape = list(inputs_shape)
    new_shape[1] = inputs_shape[1] - 2
    inputs_without_boundary_tokens = mx.nd.zeros(shape=new_shape)
    new_mask = mx.nd.zeros(shape=(new_shape[0], new_shape[1]))
    for i, j in enumerate(sequence_lengths):
        if j > 2:
            inputs_without_boundary_tokens[i, :int((j - 2)), :] = inputs[i, 1:int((j - 1)), :]
            new_mask[i, :int((j - 2))] = 1

    return inputs_without_boundary_tokens, new_mask


for i, batch in enumerate(sample_data_loader):
    print('batch id %d' % i)
    output, hidden_state, mask = elmo_bilm(batch[0], hidden_state)
    hidden_state = detach(hidden_state)
    top_layer_embeddings, mask = remove_sentence_boundaries(
        output[2],
        mask
    )
    # generate the mask lengths
    lengths = mask.asnumpy().sum(axis=1)
    for k in range(3):
        print('k %d' % k)
        print(top_layer_embeddings[k, :int(lengths[k]), :].asnumpy())
```

## Conclusion

In this tutorial, we show how to generate sentence representation from ELMo model.
In GluonNLP, this can be done with just a few simple steps: reuse the data transformation from ELMo for preprocessing the data, automatically download the pre-trained model, and feed the transformed data into the model.
To see how to plug in the pre-trained models in your own model architecture and use fine-tuning to improve downstream tasks, check our TODO link to sentiment analysis through finetuning notebook.

## Reference
[1] Peters, Matthew E., et al. "Deep contextualized word representations." NAACL (2018).
