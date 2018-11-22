# Using ELMo to Generate Sentence Representation

In this notebook, we will use GluonNLP to show how to generate ELMo sentence representation proposed in [1].

We will focus on showing how to 
* 1) process data, transform data and create the data loader;
* 2) load the pretrained ELMo model, and generate the sentence representation for the input data.

## Preparation

### Load MXNet and GluonNLP

```{.python .input  n=1}
import warnings
warnings.filterwarnings('ignore')

import mxnet as mx
from mxnet import gluon
import gluonnlp as nlp
import numpy as np
import os
import json
```

### Preprocess Data

```{.python .input}
def _load_sentences_embeddings(sentences_json_file):
    """
    Load the test sentences and the expected LM embeddings.

    These files loaded in this method were created with a batch-size of 3.
    The 30 sentences in sentences.json are split into 3 files in which
    the k-th sentence in each is from batch k.

    This method returns a sentences.
    """
    with open(sentences_json_file) as fin:
        sentences = json.load(fin)
    return sentences

sentences_json_file = os.path.join(os.path.dirname(os.path.abspath('elmo_sentence_representation.md')), 'sentences.json')
sentences = _load_sentences_embeddings(sentences_json_file)
```

### Create Dataset

```{.python .input}
class SampleDataset(gluon.data.SimpleDataset):
    """Common text dataset that reads a whole corpus in List[String] based on provided sample splitter
    and word tokenizer.

    Parameters
    ----------
    passages : list of str
        Path to the list data.
    encoding : str, default 'utf8'
        File encoding format.
    flatten : bool, default False
        Whether to return all samples as flattened tokens. If True, each sample is a token.
    skip_empty : bool, default True
        Whether to skip the empty samples produced from sample_splitters. If False, `bos` and `eos`
        will be added in empty samples.
    sample_splitter : function, default str.splitlines
        A function that splits the dataset string into samples.
    tokenizer : function or None, default str.split
        A function that splits each sample string into list of tokens. If None, raw samples are
        returned according to `sample_splitter`.
    bos : str or None, default None
        The token to add at the begining of each sequence. If None, or if tokenizer is not
        specified, then nothing is added.
    eos : str or None, default None
        The token to add at the end of each sequence. If None, or if tokenizer is not
        specified, then nothing is added.
    """

    def __init__(self, passages, encoding='utf8', flatten=False, skip_empty=True,
                 sample_splitter=nlp.data.line_splitter, tokenizer=nlp.data.whitespace_splitter,
                 bos=None, eos=None):

        self._passages = passages
        self._encoding = encoding
        self._flatten = flatten
        self._skip_empty = skip_empty
        self._sample_splitter = sample_splitter
        self._tokenizer = tokenizer
        self._bos = bos
        self._eos = eos
        super(SampleDataset, self).__init__(self._read())

    def _read(self):
        all_samples = []
        for passage in self._passages:
            samples = (s.strip() for s in passage)
            if self._tokenizer:
                samples = [
                    self._corpus_dataset_process(self._tokenizer(s), self._bos, self._eos)
                    for s in samples if s or not self._skip_empty
                ]
                if self._flatten:
                    samples = nlp.data.concat_sequence(samples)
            elif self._skip_empty:
                samples = [s for s in samples if s]

            all_samples += samples
        return all_samples

    def _corpus_dataset_process(self, s, bos, eos):
        tokens = [bos] if bos else []
        tokens.extend(s)
        if eos:
            tokens.append(eos)
        return tokens
    
sample_dataset = SampleDataset(sentences, flatten=False)
```

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

### Create Dataloader

```{.python .input}
import gluonnlp.data.batchify as btf

#create batchify function
sample_dataset_batchify_fn = nlp.data.batchify.Tuple(btf.Pad(), btf.Stack(), btf.Stack())
```

```{.python .input}
batch_size = 3
sample_data_loader = gluon.data.DataLoader(sample_dataset,
                                           batch_size=batch_size,
                                           sampler=mx.gluon.contrib.data.sampler.IntervalSampler(len(sample_dataset), interval=int(len(sample_dataset)/batch_size)),
                                           batchify_fn=sample_dataset_batchify_fn,
                                           num_workers=8)
```

## Load Pretrained ELMo Model

```{.python .input}
elmo_bilm = nlp.model.get_model('elmo_2x1024_128_2048cnn_1xhighway',
                                 dataset_name='gbw',
                                 pretrained=True,
                                 ctx=mx.cpu())
print(elmo_bilm)
hidden_state = elmo_bilm.begin_state(mx.nd.zeros, batch_size=batch_size)
```

## Generate Sentence Representation

```{.python .input}
def detach(hidden):
    """Transfer hidden states into new states, to detach them from the history.
    Parameters
    ----------
    hidden : NDArray
        The hidden states
    Returns
    ----------
    hidden: NDArray
        The detached hidden states
    """
    if isinstance(hidden, (tuple, list)):
        hidden = [detach(h) for h in hidden]
    else:
        hidden = hidden.detach()
    return hidden

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

In this section, we show that how to generate ELMo representation for sentences including 
* 1) how to process, transform data, and create the dataloader for the samples; 
* 2) how to use the pretrained ELMo model to generate the sentence representation.


## Reference
[1] Peters, Matthew E., et al. "Deep contextualized word representations." NAACL (2018).
