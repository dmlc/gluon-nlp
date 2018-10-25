# Word Embeddings Training and Evaluation

```python
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

### Data
Here we use the Text8 corpus from the [Large Text Compression
Benchmark](http://mattmahoney.net/dc/textdata.html) which includes the first
100
MB of cleaned text from the English Wikipedia.

```python
dataset = nlp.data.Text8()
print('# sentences:', len(dataset))
for sentence in dataset[:3]:
    print('# tokens:', len(sentence), sentence[:5])
```

We pass the dataset to the `prepare_batches` function from the `utils.py` of
this notebook.

The following actions are taken:
- subsampling
- negative
samples
- subwords in CSR

Some words such as "the", "a", and "in" are very
frequent.
One important trick applied when training word embeddings is to
subsample the dataset
according to the token frequencies. [1] proposes to
discard individual
occurences of words from the dataset with probability
$$P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}}$$

where $f(w_i)$ is the frequency with
which a word is
observed in a dataset and $t$ is a subsampling constant
typically chosen around
$10^{-5}$.

[1] Mikolov, Tomas, et al. “Distributed
representations of words and phrases and their compositionality.” Advances in
neural information processing systems. 2013.

```python
import utils
(batches, vocab, subword_function) = utils.prepare_batches(dataset, ngrams=[3,4,5,6],
                                                           num_subwords=10000, num_negatives=5, batch_size=16384, window=5)
```

## Training word embeddings with subword information

`gluonnlp` provides the
concept of a SubwordFunction which maps words to a list of indices representing
their subword.
Possible SubwordFunctions include mapping a word to the sequence
of it's characters/bytes or hashes of all its ngrams.

FastText models use a
hash function to map each ngram of a word to a number in range `[0,
num_subwords)`. We include the same hash function.

### Concept of a
SubwordFunction

```python
idx_to_subwordidxs = subword_function(vocab.idx_to_token)
for word, subwords in zip(vocab.idx_to_token[:3], idx_to_subwordidxs[:3]):
    print('<'+word+'>', subwords, sep = '\t')
```

### Model

`gluonnlp` provides model definitions for popular embedding models as
Gluon Blocks.
Here we show how to train them with the Skip-Gram objective, a
simple and popular embedding training objective  introduced
by "Mikolov et al.,
Efficient estimation of word representations in vector space. ICLR Workshop ,
2013."

The Skip-Gram objective trains word vectors such that the word vector of
a word
at some position in a sentence can best predict the surrounding words. We
call
these words *center* and *context* words.

Here we define a Skip-Gram model
for training fastText (Bojanowski et al., Enriching Word Vectors with Subword
Information)
embeddings. For Skip-Gram, the model consists of a fastText style
embedding with subword support which is used to embed
the center words and their
subword units as well as a standard embedding matrix used to embed the context
words and
negative samples.
After training, only the fastText style embedding
with subword support is used.

GluonNLP provides a
`nlp.model.train.FasttextEmbeddingModel` Block which defines the fastText style
embedding with subword support.
It can be used for training, but also supports
loading models trained with the original C++ fastText library from `.bin` files.
After training, vectors for arbitrary words can be looked up via
`embedding[['a', 'list', 'of', 'potentially', 'unknown', 'words']]` where
`embedding` is a `nlp.model.train.FasttextEmbeddingModel`.

```python
from utils import SkipGramNet
print(SkipGramNet.hybrid_forward.__doc__)
```

```python
emsize = 300
num_negatives = 5

embedding = SkipGramNet(emsize,  vocab, num_negatives, subword_function)
embedding.initialize(ctx=context)
embedding.hybridize()
trainer = mx.gluon.Trainer(embedding.collect_params(), 'adagrad', dict(learning_rate=0.05))

print(embedding)
```

Before we start training, let's examine the quality of our randomly initialized
embeddings:

```python
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

```python
example_token = "vector"
get_k_closest_tokens(vocab, embedding, 10, example_token)
```

We can see that in the randomly initialized fastText model the closest tokens to
"vector" are based on overlapping ngrams.

### Training

We train the model with negative sampling. That means, that for
every center-context word pair that occurs in our corpus, `num_negatives` random
words are sampled. The model is trained to distinguish for every center word if
a given other word is really co-occuring in the corpus or a random negative
sample.
Here we initialize `negatives_sampler` that allows us to sample
negatives based on the unigram token frequency.

We can use `EmbeddingCenterContextBatchify` to transform a corpus into batches
of center and context words.

```python
def train_embedding(num_epochs):
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_l_sum = 0
        num_samples = 0
        
        print('Beginnign epoch %d and resampling data.' % epoch)
        for i, batch in enumerate(batches):
            center, word_context, negatives, mask = batch
            center = center.as_in_context(context)
            word_context = word_context.as_in_context(context)
            negatives = negatives.as_in_context(context)
            mask = mask.as_in_context(context)
            
            with mx.autograd.record():
                l = embedding(center, word_context, negatives, mask)
            l.backward()
            trainer.step(1)
            
            train_l_sum += l.sum()
            num_samples += l.shape[0]
            if i % 50 == 0:
                mx.nd.waitall()
                wps = num_samples / (time.time() - start_time)
                print('epoch %d, time %.2fs, iteration %d, throughput=%.2fK wps'
                      % (epoch, time.time() - start_time, i, wps / 1000))

        print('epoch %d, time %.2fs, train loss %.2f'
              % (epoch, time.time() - start_time,
                 train_l_sum.asscalar() / num_samples))
        get_k_closest_tokens(vocab, embedding, 10, example_token)
        print("")
```

```python
train_embedding(num_epochs=1)
```

### Word Similarity and Relatedness Task

Word embeddings should capture the
relationsship between words in natural language.
In the Word Similarity and
Relatedness Task word embeddings are evaluated by comparing word similarity
scores computed from a pair of words with human labels for the similarity or
relatedness of the pair.

`gluonnlp` includes a number of common datasets for
the Word Similarity and Relatedness Task. The included datasets are listed in
the [API documentation](http://gluon-nlp.mxnet.io/api/data.html#word-embedding-
evaluation-datasets). We use several of them in the evaluation example below.
We first show a few samples from the WordSim353 dataset, to get an overall
feeling of the Dataset structur

### Evaluation

Thanks to the subword support of the `FasttextEmbeddingModel` we
can evaluate on all words in the evaluation dataset,
not only the ones that we
observed during training.

We first compute a list of tokens in our evaluation
dataset and then create a embedding matrix for them based on the fastText model.

```python
rw = nlp.data.RareWords()
rw_tokens  = list(set(itertools.chain.from_iterable((d[0], d[1]) for d in rw)))

rw_matrix = nlp.embedding.TokenEmbedding(unknown_token=None, allow_extend=True)
rw_matrix[rw_tokens]= embedding[rw_tokens]

print('There are', len(rw_tokens), 'unique tokens in the RareWords dataset. Examples are:')
for i in range(5):
    print('\t', rw[i])
print('The imputed TokenEmbedding has shape', token_embedding.idx_to_vec.shape)
```

```python
evaluator = nlp.embedding.evaluation.WordEmbeddingSimilarity(
    idx_to_vec=rw_matrix,
    similarity_function="CosineSimilarity")
evaluator.initialize(ctx=context)
evaluator.hybridize()
```

```python
words1, words2, scores = zip(*([token_embedding.token_to_idx[d[0]],
                                token_embedding.token_to_idx[d[1]],
                                d[2]] for d in rw))
words1 = mx.nd.array(words1, ctx=context)
words2 = mx.nd.array(words2, ctx=context)
```

```python
pred_similarity = evaluator(words1, words2)
sr = stats.spearmanr(pred_similarity.asnumpy(), np.array(scores))
print('Spearman rank correlation on {} pairs of {}: {}'.format(
    len(words1), rw.__class__.__name__, sr.correlation.round(3)))
```

# Further information

For further information and examples on training and
evaluating word embeddings with GluonNLP take a look at the Word Embedding
section on the Scripts / Model Zoo page. There you will find more thorough
evaluation techniques and other embedding models.
