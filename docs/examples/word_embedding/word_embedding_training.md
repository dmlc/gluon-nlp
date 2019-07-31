# Word Embeddings Training and Evaluation

```{.python .input}
import warnings
warnings.filterwarnings('ignore')

import itertools
import time
import math
import logging
import random

import mxnet as mx
import gluonnlp as nlp
import numpy as np
from scipy import stats

# context = mx.cpu()  # Enable this to run on CPU
context = mx.gpu(0)  # Enable this to run on GPU
```

## Data
Here we use the Text8 corpus from the [Large Text Compression
Benchmark](http://mattmahoney.net/dc/textdata.html) which includes the first
100
MB of cleaned text from Wikipedia in English.

```{.python .input}
text8 = nlp.data.Text8()
print('# sentences:', len(text8))
for sentence in text8[:3]:
    print('# tokens:', len(sentence), sentence[:5])
```

Given the tokenized data, we first count all tokens and then construct a
vocabulary of all tokens that occur at least 5 times in the dataset. The
vocabulary contains a one-to-one mapping between tokens and integers (also
called indices or idx for short).

Furthermore, we can store the frequency count of each
token in the vocabulary as we will require this information later on for
sampling random negative (or noise) words. Finally, we replace all tokens with
their integer representation based on the vocabulary.

```{.python .input}
counter = nlp.data.count_tokens(itertools.chain.from_iterable(text8))
vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                  bos_token=None, eos_token=None, min_freq=5)
idx_to_counts = [counter[w] for w in vocab.idx_to_token]

def code(sentence):
    return [vocab[token] for token in sentence if token in vocab]

text8 = text8.transform(code, lazy=False)

print('# sentences:', len(text8))
for sentence in text8[:3]:
    print('# tokens:', len(sentence), sentence[:5])
```

Next we need to transform the coded Text8 dataset into batches that are more useful for
training an embedding model.

In this tutorial we train leveraging the SkipGram
objective made popular by the following: [1].

For SkipGram, we sample pairs of co-occurring
words from the corpus.
Two words are said to co-occur if they occur with
distance less than a specified *window* size.
The *window* size is usually
chosen around 5. Refer to the aforementioned paper for more details.

To obtain the samples from the corpus, we can shuffle the
sentences and then proceed linearly through each sentence, considering each word
as well as all the words in its window. In this case, we call the current word
in focus the center word, and the words in its window, the context words.
GluonNLP contains `gluonnlp.data.EmbeddingCenterContextBatchify` batchify
transformation, that takes a corpus, such as the coded Text8 we have here, and
returns a `DataStream` of batches of center and context words.

To obtain good
results, each sentence is further subsampled, meaning that words are deleted
with a probability proportional to their frequency.
[1] proposes to discard
individual occurrences of words from the dataset with probability

$$P(w_i) = 1 -
\sqrt{\frac{t}{f(w_i)}}$$

where $f(w_i)$ is the frequency with which a word is
observed in a dataset and $t$ is a subsampling constant typically chosen around
$10^{-5}$.
[1] has also shown that the final performance is improved if the
window size is chosen  uniformly random for each center words out of the range
[1, *window*].

For this notebook, we are interested in training a fastText
embedding model [2]. A fastText model not only associates an embedding vector with
each token in the vocabulary, but also with a pre-specified number of subwords.
Commonly 2 million subword vectors are obtained and each subword vector is
associated with zero, one, or multiple character-ngrams. The mapping between
character-ngrams and subwords is based on a hash function.
The *final* embedding
vector of a token is the mean of the vectors associated with the token and all
character-ngrams occurring in the string representation of the token. Thereby a
fastText embedding model can compute meaningful embedding vectors for tokens
that were not seen during training.

For this notebook, we have prepared a helper function `transform_data_fasttext`
which builds a series of transformations of the `text8 Dataset` created above,
applying the techniques we mention briefly above. It returns a `DataStream` over batches as
well as a `batchify_fn` function that applied to a batch looks up and includes the
fastText subwords associated with the center words. Additionally, it returns the subword
function which can be used to obtain the subwords of a given string
representation of a token. We will take a closer look at the subword function
farther on.

You can find the `transform_data_fasttext()` function in `data.py` in the
archive that can be downloaded via the `Download` button at the top of this page.

```{.python .input}
from data import transform_data_fasttext

batch_size=4096
data = nlp.data.SimpleDataStream([text8])  # input is a stream of datasets, here just 1. Allows scaling to larger corpora that don't fit in memory
data, batchify_fn, subword_function = transform_data_fasttext(
    data, vocab, idx_to_counts, cbow=False, ngrams=[3,4,5,6], ngram_buckets=100000, batch_size=batch_size, window_size=5)
```

```{.python .input}
batches = data.transform(batchify_fn)
```

Note that the number of subwords is potentially
different for every word. Therefore the batchify_fn represents a word with its
subwords as a row in a compressed sparse row (CSR) matrix. For more information on CSR matrices click here:
https://mxnet.incubator.apache.org/tutorials/sparse/csr.html

Separating the batchify_fn from the previous word-pair
sampling is useful, as it allows parallelization of the CSR matrix construction over
multiple CPU cores for separate batches.

## Subwords

`GluonNLP` provides the concept of a subword function which maps
words to a list of indices representing their subword.
Possible subword functions
include mapping a word to the sequence of it's characters/bytes or hashes of all
its ngrams.

FastText models use a hash function to map each ngram of a word to
a number in range `[0, num_subwords)`. We include the same hash function.
Above
`transform_data_fasttext` has also returned a `subword_function` object. Let's try it with
a few words:

```{.python .input}
idx_to_subwordidxs = subword_function(vocab.idx_to_token)
for word, subwords in zip(vocab.idx_to_token[:3], idx_to_subwordidxs[:3]):
    print('<'+word+'>', subwords, sep = '\t')
```

## Model

Here we define a SkipGram model for training fastText embeddings.
For
Skip-Gram, the model consists of two independent embedding networks.
One for the
center words, and one for the context words.
For center words, subwords are
taken into account while for context words only the token itself is taken into
account.

GluonNLP provides an `nlp.model.train.FasttextEmbeddingModel` block
which defines the fastText style embedding with subword support.
It can be used
for training, but also supports loading models trained with the original C++
fastText library from `.bin` files.
After training, vectors for arbitrary words
can be looked up via `embedding[['a', 'list', 'of', 'potentially', 'unknown',
'words']]` where `embedding` is an `nlp.model.train.FasttextEmbeddingModel`.

In
the `model.py` script we provide a definition for the fastText model for the
SkipGram objective.
The model definition is a Gluon HybridBlock, meaning that
the complete forward / backward pass are compiled and executed directly in the
MXNet backend. Not only does the block include the `FasttextEmbeddingModel` for
the center words and a simple embedding matrix for the context words, but it
also takes care of sampling a specified number of noise words for each center-
context pair. These noise words are called negatives, as the resulting center-
negative pair is unlikely to occur in the dataset. The model then must learn
which word-pairs are negatives and which ones are real. Thereby it obtains
meaningful word and subword vectors for all considered tokens. The negatives are
sampled from the smoothed unigram frequency distribution.

Let's instantiate and
initialize the model. We also create a trainer object for updating the
parameters with AdaGrad.
Finally we print a summary of the model.

```{.python .input}
from model import SG as SkipGramNet

emsize = 300
num_negatives = 5

negatives_weights = mx.nd.array(idx_to_counts)
embedding = SkipGramNet(
    vocab.token_to_idx, emsize, batch_size, negatives_weights, subword_function, num_negatives=5, smoothing=0.75)
embedding.initialize(ctx=context)
embedding.hybridize()
trainer = mx.gluon.Trainer(embedding.collect_params(), 'adagrad', dict(learning_rate=0.05))

print(embedding)
```

Let's take a look at the documentation of the forward pass.

```{.python .input}
print(SkipGramNet.hybrid_forward.__doc__)
```

Before we start training, let's examine the quality of our randomly initialized
embeddings:

```{.python .input}
def norm_vecs_by_row(x):
    return x / (mx.nd.sum(x * x, axis=1) + 1e-10).sqrt().reshape((-1, 1))


def get_k_closest_tokens(vocab, embedding, k, word):
    word_vec = norm_vecs_by_row(embedding[[word]])
    vocab_vecs = norm_vecs_by_row(embedding[vocab.idx_to_token])
    dot_prod = mx.nd.dot(vocab_vecs, word_vec.T)
    indices = mx.nd.topk(
        dot_prod.reshape((len(vocab.idx_to_token), )),
        k=k + 1,
        ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]
    result = [vocab.idx_to_token[i] for i in indices[1:]]
    print('closest tokens to "%s": %s' % (word, ", ".join(result)))
```

```{.python .input}
example_token = "vector"
get_k_closest_tokens(vocab, embedding, 10, example_token)
```

We can see that in the randomly initialized fastText model the closest tokens to
"vector" are based on overlapping ngrams.

## Training

Thanks to the Gluon data pipeline and the HybridBlock handling all
complexity, our training code is very simple.
We iterate over all batches, move
them to the appropriate context (GPU), do forward, backward, and parameter update
and finally include some helpful print statements for following the training
process.

```{.python .input}
log_interval = 500

def train_embedding(num_epochs):
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        l_avg = 0
        log_wc = 0

        print('Beginnign epoch %d and resampling data.' % epoch)
        for i, batch in enumerate(batches):
            batch = [array.as_in_context(context) for array in batch]
            with mx.autograd.record():
                l = embedding(*batch)
            l.backward()
            trainer.step(1)

            l_avg += l.mean()
            log_wc += l.shape[0]
            if i % log_interval == 0:
                mx.nd.waitall()
                wps = log_wc / (time.time() - start_time)
                l_avg = l_avg.asscalar() / log_interval
                print('epoch %d, iteration %d, loss %.2f, throughput=%.2fK wps'
                      % (epoch, i, l_avg, wps / 1000))
                start_time = time.time()
                log_wc = 0
                l_avg = 0

        get_k_closest_tokens(vocab, embedding, 10, example_token)
        print("")
```

```{.python .input}
train_embedding(num_epochs=1)
```

## Word Similarity and Relatedness Task

Word embeddings should capture the
relationship between words in natural language.
In the Word Similarity and
Relatedness Task, word embeddings are evaluated by comparing word similarity
scores computed from a pair of words with human labels for the similarity or
relatedness of the pair.

`GluonNLP` includes a number of common datasets for
the Word Similarity and Relatedness Task. The included datasets are listed in
the [API documentation](http://gluon-nlp.mxnet.io/api/data.html#word-embedding-evaluation-datasets). We use several of them in the evaluation example below.
We first show a few samples from the WordSim353 dataset, to get an overall
feeling of the Dataset structure.

## Evaluation

Thanks to the subword support of the `FasttextEmbeddingModel` we
can evaluate on all words in the evaluation dataset,
not only on the ones that we
observed during training.

We first compute a list of tokens in our evaluation
dataset and then create an embedding matrix for them based on the fastText model.

```{.python .input}
rw = nlp.data.RareWords()
rw_tokens  = list(set(itertools.chain.from_iterable((d[0], d[1]) for d in rw)))

rw_token_embedding = nlp.embedding.TokenEmbedding(unknown_token=None, allow_extend=True)
rw_token_embedding[rw_tokens]= embedding[rw_tokens]

print('There are', len(rw_tokens), 'unique tokens in the RareWords dataset. Examples are:')
for i in range(5):
    print('\t', rw[i])
print('The imputed TokenEmbedding has shape', rw_token_embedding.idx_to_vec.shape)
```

```{.python .input}
evaluator = nlp.embedding.evaluation.WordEmbeddingSimilarity(
    idx_to_vec=rw_token_embedding.idx_to_vec,
    similarity_function="CosineSimilarity")
evaluator.initialize(ctx=context)
evaluator.hybridize()
```

```{.python .input}
words1, words2, scores = zip(*([rw_token_embedding.token_to_idx[d[0]],
                                rw_token_embedding.token_to_idx[d[1]],
                                d[2]] for d in rw))
words1 = mx.nd.array(words1, ctx=context)
words2 = mx.nd.array(words2, ctx=context)
```

```{.python .input}
pred_similarity = evaluator(words1, words2)
sr = stats.spearmanr(pred_similarity.asnumpy(), np.array(scores))
print('Spearman rank correlation on {} pairs of {}: {}'.format(
    len(words1), rw.__class__.__name__, sr.correlation.round(3)))
```

## Further information

For further information and examples on training and
evaluating word embeddings with GluonNLP take a look at the Word Embedding
section on the Scripts / Model Zoo page. There you will find more thorough
evaluation techniques and other embedding models. In fact, the `data.py` and
`model.py` files used in this example are the same as the ones used in the
script.

## References

- [1] Mikolov, Tomas, et al. “Distributed representations of words and phrases
and their compositionally.”
   Advances in neural information processing
systems. 2013.


- [2] Bojanowski et al., "Enriching Word Vectors with Subword
Information" Transactions of the Association for Computational Linguistics 2017
