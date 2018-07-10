# Using Pre-trained Word Embeddings

Here we introduce how to use pre-trained word embeddings, where each word is
represened by a vector. Two popular word embeddings are GloVe and fastText. The
used GloVe and fastText pre-trained word embeddings here are from the following
sources:

* GloVe project website：https://nlp.stanford.edu/projects/glove/
*
fastText project website：https://fasttext.cc/

Let us first import the following
packages used in this example.

```python
import warnings
warnings.filterwarnings('ignore')

from mxnet import gluon
from mxnet import nd
import gluonnlp as nlp
import re
```

## Creating Vocabulary with Word Embeddings

As a common use case, let us index
words, attach pre-trained word embeddings for them, and use such embeddings in
Gluon. We will assign a unique ID and word vector to each word in the vocabulary
in just a few lines of code.

### Creating Vocabulary from Data Sets

To begin
with, suppose that we have a simple text data set in the string format.

```python
text = " hello world \n hello nice world \n hi world \n"
```

We can use our defined tokenizer to count word frequency in the data set.

```python
def simple_tokenize(source_str, token_delim=' ', seq_delim='\n'):
    return filter(None, re.split(token_delim + '|' + seq_delim, source_str))
counter = nlp.data.count_tokens(simple_tokenize(text))
```

The obtained `counter` has key-value pairs whose keys are words and values are
word frequencies. This allows us to filter out infrequent words via `Vocab`
arguments such as `max_size` and `min_freq`. Suppose that we want to build
indices for all the keys in counter. We need a `Vocab` instance with counter as
its argument.

```python
vocab = nlp.Vocab(counter)
```

To attach word embedding to indexed words in `vocab`, let us go on to create a
fastText word embedding instance by specifying the embedding name `fasttext` and
the source name `wiki.simple`.

```python
fasttext_simple = nlp.embedding.create('fasttext', source='wiki.simple')
```

So we can attach word embedding `fasttext_simple` to indexed words in `vocab`.

```python
vocab.set_embedding(fasttext_simple)
```

To see other source names under the fastText word embedding, we can use
`text.embedding.list_sources`.

```python
nlp.embedding.list_sources('fasttext')[:5]
```

The created vocabulary `vocab` includes four different words and a special
unknown token. Let us check the size of `vocab`.

```python
len(vocab)
```

By default, the vector of any token that is unknown to `vocab` is a zero vector.
Its length is equal to the vector dimension of the fastText word embeddings:
300.

```python
vocab.embedding['beautiful'].shape
```

The first five elements of the vector of any unknown token are zeros.

```python
vocab.embedding['beautiful'][:5]
```

Let us check the shape of the embedding of words 'hello' and 'world' from
`vocab`.

```python
vocab.embedding['hello', 'world'].shape
```

We can access the first five elements of the embedding of 'hello' and 'world'.

```python
vocab.embedding['hello', 'world'][:, :5]
```

### Using Pre-trained Word Embeddings in Gluon

To demonstrate how to use pre-
trained word embeddings in Gluon, let us first obtain indices of the words
'hello' and 'world'.

```python
vocab['hello', 'world']
```

We can obtain the vectors for the words 'hello' and 'world' by specifying their
indices (2 and 1) and the weight matrix `vocab.embedding.idx_to_vec` in
`gluon.nn.Embedding`.

```python
input_dim, output_dim = vocab.embedding.idx_to_vec.shape
layer = gluon.nn.Embedding(input_dim, output_dim)
layer.initialize()
layer.weight.set_data(vocab.embedding.idx_to_vec)
layer(nd.array([5, 4]))[:, :5]
```

### Creating Vocabulary from Pre-trained Word Embeddings

We can also create
vocabulary by using vocabulary of pre-trained word embeddings, such as GloVe.
Below are a few pre-trained file names under the GloVe word embedding.

```python
nlp.embedding.list_sources('glove')[:5]
```

For simplicity of demonstration, we use a smaller word embedding file, such as
the 50-dimensional one.

```python
glove_6b50d = nlp.embedding.create('glove', source='glove.6B.50d')
```

Now we create vocabulary by using all the tokens from `glove_6b50d`.

```python
vocab = nlp.Vocab(nlp.data.Counter(glove_6b50d.idx_to_token))
vocab.set_embedding(glove_6b50d)
```

Below shows the size of `vocab` including a special unknown token.

```python
len(vocab.idx_to_token)
```

We can access attributes of `vocab`.

```python
print(vocab['beautiful'])
print(vocab.idx_to_token[71424])
```

## Applications of Word Embeddings

To apply word embeddings, we need to define
cosine similarity. It can compare similarity of two vectors.

```python
from mxnet import nd
def cos_sim(x, y):
    return nd.dot(x, y) / (nd.norm(x) * nd.norm(y))
```

The range of cosine similarity between two vectors is between -1 and 1. The
larger the value, the similarity between two vectors.

```python
x = nd.array([1, 2])
y = nd.array([10, 20])
z = nd.array([-1, -2])

print(cos_sim(x, y))
print(cos_sim(x, z))
```

### Word Similarity

Given an input word, we can find the nearest $k$ words from
the vocabulary (400,000 words excluding the unknown token) by similarity. The
similarity between any pair of words can be represented by the cosine similarity
of their vectors.

```python
def norm_vecs_by_row(x):
    return x / nd.sqrt(nd.sum(x * x, axis=1) + 1E-10).reshape((-1,1))

def get_knn(vocab, k, word):
    word_vec = vocab.embedding[word].reshape((-1, 1))
    vocab_vecs = norm_vecs_by_row(vocab.embedding.idx_to_vec)
    dot_prod = nd.dot(vocab_vecs, word_vec)
    indices = nd.topk(dot_prod.reshape((len(vocab), )), k=k+1, ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]
    # Remove unknown and input tokens.
    return vocab.to_tokens(indices[1:])
```

Let us find the 5 most similar words of 'baby' from the vocabulary (size:
400,000 words).

```python
get_knn(vocab, 5, 'baby')
```

We can verify the cosine similarity of vectors of 'baby' and 'babies'.

```python
cos_sim(vocab.embedding['baby'], vocab.embedding['babies'])
```

Let us find the 5 most similar words of 'computers' from the vocabulary.

```python
get_knn(vocab, 5, 'computers')
```

Let us find the 5 most similar words of 'run' from the vocabulary.

```python
get_knn(vocab, 5, 'run')
```

Let us find the 5 most similar words of 'beautiful' from the vocabulary.

```python
get_knn(vocab, 5, 'beautiful')
```

### Word Analogy

We can also apply pre-trained word embeddings to the word
analogy problem. For instance, "man : woman :: son : daughter" is an analogy.
The word analogy completion problem is defined as: for analogy 'a : b :: c : d',
given teh first three words 'a', 'b', 'c', find 'd'. The idea is to find the
most similar word vector for vec('c') + (vec('b')-vec('a')).

In this example,
we will find words by analogy from the 400,000 indexed words in `vocab`.

```python
def get_top_k_by_analogy(vocab, k, word1, word2, word3):
    word_vecs = vocab.embedding[word1, word2, word3]
    word_diff = (word_vecs[1] - word_vecs[0] + word_vecs[2]).reshape((-1, 1))
    vocab_vecs = norm_vecs_by_row(vocab.embedding.idx_to_vec)
    dot_prod = nd.dot(vocab_vecs, word_diff)
    indices = nd.topk(dot_prod.reshape((len(vocab), )), k=k, ret_typ='indices')
    indices = [int(i.asscalar()) for i in indices]
    return vocab.to_tokens(indices)
```

Complete word analogy 'man : woman :: son :'.

```python
get_top_k_by_analogy(vocab, 1, 'man', 'woman', 'son')
```

Let us verify the cosine similarity between vec('son')+vec('woman')-vec('man')
and vec('daughter')

```python
def cos_sim_word_analogy(vocab, word1, word2, word3, word4):
    words = [word1, word2, word3, word4]
    vecs = vocab.embedding[words]
    return cos_sim(vecs[1] - vecs[0] + vecs[2], vecs[3])

cos_sim_word_analogy(vocab, 'man', 'woman', 'son', 'daughter')
```

Complete word analogy 'beijing : china :: tokyo : '.

```python
get_top_k_by_analogy(vocab, 1, 'beijing', 'china', 'tokyo')
```

Complete word analogy 'bad : worst :: big : '.

```python
get_top_k_by_analogy(vocab, 1, 'bad', 'worst', 'big')
```

Complete word analogy 'do : did :: go :'.

```python
get_top_k_by_analogy(vocab, 1, 'do', 'did', 'go')
```
