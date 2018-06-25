# Using Pre-trained Word Embeddings

Here we introduce how to use pre-trained word embeddings, where each word is represened by a vector. Two popular word embeddings are GloVe and fastText. The used GloVe and fastText pre-trained word embeddings here are from the following sources:

* GloVe project website：https://nlp.stanford.edu/projects/glove/
* fastText project website：https://fasttext.cc/

Let us first import the following packages used in this example.

```{.python .input  n=1}
import warnings
warnings.filterwarnings('ignore')

from mxnet import gluon
from mxnet import nd
import gluonnlp as nlp
import re
```

## Creating Vocabulary with Word Embeddings

As a common use case, let us index words, attach pre-trained word embeddings for them, and use such embeddings in Gluon. We will assign a unique ID and word vector to each word in the vocabulary in just a few lines of code.

### Creating Vocabulary from Data Sets

To begin with, suppose that we have a simple text data set in the string format.

```{.python .input  n=2}
text = " hello world \n hello nice world \n hi world \n"
```

We can use our defined tokenizer to count word frequency in the data set.

```{.python .input  n=3}
def simple_tokenize(source_str, token_delim=' ', seq_delim='\n'):
    return filter(None, re.split(token_delim + '|' + seq_delim, source_str))
counter = nlp.data.count_tokens(simple_tokenize(text))
```

The obtained `counter` has key-value pairs whose keys are words and values are word frequencies. This allows us to filter out infrequent words via `Vocab` arguments such as `max_size` and `min_freq`. Suppose that we want to build indices for all the keys in counter. We need a `Vocab` instance with counter as its argument.

```{.python .input  n=4}
vocab = nlp.Vocab(counter)
```

To attach word embedding to indexed words in `vocab`, let us go on to create a fastText word embedding instance by specifying the embedding name `fasttext` and the source name `wiki.simple`.

```{.python .input  n=5}
fasttext_simple = nlp.embedding.create('fasttext', source='wiki.simple')
```

So we can attach word embedding `fasttext_simple` to indexed words in `vocab`.

```{.python .input  n=6}
vocab.set_embedding(fasttext_simple)
```

To see other source names under the fastText word embedding, we can use `text.embedding.list_sources`.

```{.python .input  n=7}
nlp.embedding.list_sources('fasttext')[:5]
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "['crawl-300d-2M', 'wiki.aa', 'wiki.ab', 'wiki.ace', 'wiki.ady']"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

The created vocabulary `vocab` includes four different words and a special unknown token. Let us check the size of `vocab`.

```{.python .input  n=8}
len(vocab)
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "8"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

By default, the vector of any token that is unknown to `vocab` is a zero vector. Its length is equal to the vector dimension of the fastText word embeddings: 300.

```{.python .input  n=9}
vocab.embedding['beautiful'].shape
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "(300,)"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

The first five elements of the vector of any unknown token are zeros.

```{.python .input  n=10}
vocab.embedding['beautiful'][:5]
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "\n[0. 0. 0. 0. 0.]\n<NDArray 5 @cpu(0)>"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Let us check the shape of the embedding of words 'hello' and 'world' from `vocab`.

```{.python .input  n=11}
vocab.embedding['hello', 'world'].shape
```

```{.json .output n=11}
[
 {
  "data": {
   "text/plain": "(2, 300)"
  },
  "execution_count": 11,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

We can access the first five elements of the embedding of 'hello' and 'world'.

```{.python .input  n=12}
vocab.embedding['hello', 'world'][:, :5]
```

```{.json .output n=12}
[
 {
  "data": {
   "text/plain": "\n[[ 0.39567   0.21454  -0.035389 -0.24299  -0.095645]\n [ 0.10444  -0.10858   0.27212   0.13299  -0.33165 ]]\n<NDArray 2x5 @cpu(0)>"
  },
  "execution_count": 12,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### Using Pre-trained Word Embeddings in Gluon

To demonstrate how to use pre-trained word embeddings in Gluon, let us first obtain indices of the words 'hello' and 'world'.

```{.python .input  n=13}
vocab['hello', 'world']
```

```{.json .output n=13}
[
 {
  "data": {
   "text/plain": "[5, 4]"
  },
  "execution_count": 13,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

We can obtain the vectors for the words 'hello' and 'world' by specifying their indices (2 and 1) and the weight matrix `vocab.embedding.idx_to_vec` in `gluon.nn.Embedding`.

```{.python .input  n=14}
input_dim, output_dim = vocab.embedding.idx_to_vec.shape
layer = gluon.nn.Embedding(input_dim, output_dim)
layer.initialize()
layer.weight.set_data(vocab.embedding.idx_to_vec)
layer(nd.array([5, 4]))[:, :5]
```

```{.json .output n=14}
[
 {
  "data": {
   "text/plain": "\n[[ 0.39567   0.21454  -0.035389 -0.24299  -0.095645]\n [ 0.10444  -0.10858   0.27212   0.13299  -0.33165 ]]\n<NDArray 2x5 @cpu(0)>"
  },
  "execution_count": 14,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### Creating Vocabulary from Pre-trained Word Embeddings

We can also create vocabulary by using vocabulary of pre-trained word embeddings, such as GloVe. Below are a few pre-trained file names under the GloVe word embedding.

```{.python .input  n=15}
nlp.embedding.list_sources('glove')[:5]
```

```{.json .output n=15}
[
 {
  "data": {
   "text/plain": "['glove.42B.300d',\n 'glove.6B.100d',\n 'glove.6B.200d',\n 'glove.6B.300d',\n 'glove.6B.50d']"
  },
  "execution_count": 15,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

For simplicity of demonstration, we use a smaller word embedding file, such as the 50-dimensional one.

```{.python .input  n=16}
glove_6b50d = nlp.embedding.create('glove', source='glove.6B.50d')
```

Now we create vocabulary by using all the tokens from `glove_6b50d`.

```{.python .input  n=17}
vocab = nlp.Vocab(nlp.data.Counter(glove_6b50d.idx_to_token))
vocab.set_embedding(glove_6b50d)
```

Below shows the size of `vocab` including a special unknown token.

```{.python .input  n=18}
len(vocab.idx_to_token)
```

```{.json .output n=18}
[
 {
  "data": {
   "text/plain": "400004"
  },
  "execution_count": 18,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

We can access attributes of `vocab`.

```{.python .input  n=19}
print(vocab['beautiful'])
print(vocab.idx_to_token[71424])
```

```{.json .output n=19}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "71424\nbeautiful\n"
 }
]
```

## Applications of Word Embeddings

To apply word embeddings, we need to define cosine similarity. It can compare similarity of two vectors.

```{.python .input  n=20}
from mxnet import nd
def cos_sim(x, y):
    return nd.dot(x, y) / (nd.norm(x) * nd.norm(y))
```

The range of cosine similarity between two vectors is between -1 and 1. The larger the value, the similarity between two vectors.

```{.python .input  n=21}
x = nd.array([1, 2])
y = nd.array([10, 20])
z = nd.array([-1, -2])

print(cos_sim(x, y))
print(cos_sim(x, z))
```

```{.json .output n=21}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[1.]\n<NDArray 1 @cpu(0)>\n\n[-1.]\n<NDArray 1 @cpu(0)>\n"
 }
]
```

### Word Similarity

Given an input word, we can find the nearest $k$ words from the vocabulary (400,000 words excluding the unknown token) by similarity. The similarity between any pair of words can be represented by the cosine similarity of their vectors.

```{.python .input  n=22}
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

Let us find the 5 most similar words of 'baby' from the vocabulary (size: 400,000 words).

```{.python .input  n=23}
get_knn(vocab, 5, 'baby')
```

```{.json .output n=23}
[
 {
  "data": {
   "text/plain": "['babies', 'boy', 'girl', 'newborn', 'pregnant']"
  },
  "execution_count": 23,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

We can verify the cosine similarity of vectors of 'baby' and 'babies'.

```{.python .input  n=24}
cos_sim(vocab.embedding['baby'], vocab.embedding['babies'])
```

```{.json .output n=24}
[
 {
  "data": {
   "text/plain": "\n[0.838713]\n<NDArray 1 @cpu(0)>"
  },
  "execution_count": 24,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Let us find the 5 most similar words of 'computers' from the vocabulary.

```{.python .input  n=25}
get_knn(vocab, 5, 'computers')
```

```{.json .output n=25}
[
 {
  "data": {
   "text/plain": "['computer', 'phones', 'pcs', 'machines', 'devices']"
  },
  "execution_count": 25,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Let us find the 5 most similar words of 'run' from the vocabulary.

```{.python .input  n=26}
get_knn(vocab, 5, 'run')
```

```{.json .output n=26}
[
 {
  "data": {
   "text/plain": "['running', 'runs', 'went', 'start', 'ran']"
  },
  "execution_count": 26,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Let us find the 5 most similar words of 'beautiful' from the vocabulary.

```{.python .input  n=27}
get_knn(vocab, 5, 'beautiful')
```

```{.json .output n=27}
[
 {
  "data": {
   "text/plain": "['lovely', 'gorgeous', 'wonderful', 'charming', 'beauty']"
  },
  "execution_count": 27,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

### Word Analogy

We can also apply pre-trained word embeddings to the word analogy problem. For instance, "man : woman :: son : daughter" is an analogy. The word analogy completion problem is defined as: for analogy 'a : b :: c : d', given teh first three words 'a', 'b', 'c', find 'd'. The idea is to find the most similar word vector for vec('c') + (vec('b')-vec('a')).

In this example, we will find words by analogy from the 400,000 indexed words in `vocab`.

```{.python .input  n=28}
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

```{.python .input  n=29}
get_top_k_by_analogy(vocab, 1, 'man', 'woman', 'son')
```

```{.json .output n=29}
[
 {
  "data": {
   "text/plain": "['daughter']"
  },
  "execution_count": 29,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Let us verify the cosine similarity between vec('son')+vec('woman')-vec('man') and vec('daughter')

```{.python .input  n=30}
def cos_sim_word_analogy(vocab, word1, word2, word3, word4):
    words = [word1, word2, word3, word4]
    vecs = vocab.embedding[words]
    return cos_sim(vecs[1] - vecs[0] + vecs[2], vecs[3])

cos_sim_word_analogy(vocab, 'man', 'woman', 'son', 'daughter')
```

```{.json .output n=30}
[
 {
  "data": {
   "text/plain": "\n[0.9658343]\n<NDArray 1 @cpu(0)>"
  },
  "execution_count": 30,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Complete word analogy 'beijing : china :: tokyo : '.

```{.python .input  n=31}
get_top_k_by_analogy(vocab, 1, 'beijing', 'china', 'tokyo')
```

```{.json .output n=31}
[
 {
  "data": {
   "text/plain": "['japan']"
  },
  "execution_count": 31,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Complete word analogy 'bad : worst :: big : '.

```{.python .input  n=32}
get_top_k_by_analogy(vocab, 1, 'bad', 'worst', 'big')
```

```{.json .output n=32}
[
 {
  "data": {
   "text/plain": "['biggest']"
  },
  "execution_count": 32,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Complete word analogy 'do : did :: go :'.

```{.python .input  n=33}
get_top_k_by_analogy(vocab, 1, 'do', 'did', 'go')
```

```{.json .output n=33}
[
 {
  "data": {
   "text/plain": "['went']"
  },
  "execution_count": 33,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```
