# Using Pre-trained Word Embeddings

In this notebook, we'll demonstrate how to use pre-trained word embeddings.
To see why word embeddings are useful, it's worth comparing them to the alternative.
Without word embeddings, we may represent each word with a one-hot vector `[0, ...,0, 1, 0, ... 0]`,
where at the index corresponding to the appropriate vocabulary word, `1` is placed,
and the value `0` occurs everywhere else.
The weight matrices connecting our word-level inputs to the network's hidden layers would each be $v \times h$,
where $v$ is the size of the vocabulary and $h$ is the size of the hidden layer.
With 100,000 words feeding into an LSTM layer with $1000$ nodes, the model would need to learn
$4$ different weight matrices (one for each of the LSTM gates), each with 100 million weights, and thus 400 million parameters in total.

Fortunately, it turns out that a number of efficient techniques
can quickly discover broadly useful word embeddings in an *unsupervised* manner.
These embeddings map each word onto a low-dimensional vector $w \in R^d$ with $d$ commonly chosen to be roughly $100$.
Intuitively, these embeddings are chosen based on the contexts in which words appear.
Words that appear in similar contexts, like "tennis" and "racquet," should have similar embeddings
while words that are not alike, like "rat" and "gourmet," should have dissimilar embeddings.

Practitioners of deep learning for NLP typically initialize their models
using *pre-trained* word embeddings, bringing in outside information, and reducing the number of parameters that a neural network needs to learn from scratch.

Two popular word embeddings are GloVe and fastText.

The following examples use pre-trained word embeddings drawn from the following sources:

* GloVe project website：https://nlp.stanford.edu/projects/glove/
* fastText project website：https://fasttext.cc/

To begin, let's first import a few packages that we'll need for this example:

```{.python .input}
import warnings
warnings.filterwarnings('ignore')

from mxnet import gluon, nd
import gluonnlp as nlp
import re
import collections
import numpy as np

```

## Creating Vocabulary with Word Embeddings

Now we'll demonstrate how to index words,
attach pre-trained word embeddings for them,
and use such embeddings in Gluon.
First, let's assign a unique ID and word vector to each word in the vocabulary
in just a few lines of code.


### Creating Vocabulary from Data Sets

To begin, suppose that we have a simple text data set consisting of newline-separated strings.

```{.python .input}
text = " hello world \n hello nice world \n hi world \n goodgod"
```

To start, let's implement a simple tokenizer to separate the words and then count the frequency of each word in the data set. We can use our defined tokenizer to count word frequency in the data set.

```{.python .input}
def simple_tokenize(source_str, token_delim=' ', seq_delim='\n'):
    return list(filter(None, re.split(token_delim + '|' + seq_delim, source_str)))
counter = collections.Counter(simple_tokenize(text))
```

The obtained `counter`'s key-value pairs consist of words and their frequencies, respectively.
We can then instantiate a `Vocab` object with a counter.
Because `counter` tracks word frequencies, we are able to specify arguments
such as `max_size` (maximum size) and `min_freq` (minimum frequency) to the `Vocab` constructor to restrict the size of the resulting vocabulary. 

Suppose that we want to build indices for all the keys in counter.
If we simply want to construct a  `Vocab` containing every word, then we can supply `counter`  the only argument.

```{.python .input}
vocab = nlp.data.Vocab(counter)
```

A `Vocab` object associates each word with an index. We can easily access words by their indices using the `vocab.all_tokens` attribute.

```{.python .input}
for word in vocab.all_tokens:
    print(word)
```

Contrarily, we can also grab an index given a token using `__getitem__` or `vocab.token_to_idx`.

```{.python .input}
print(vocab["<unk>"])
print(vocab.token_to_idx["world"])
```


### Load word embeddings

Our next step will be to load word embeddings for a given `vocab`.
In this example, we'll use *fastText* embeddings trained on the *wiki.simple* dataset.

```{.python .input}
matrix = nlp.embedding.load_embeddings(vocab, 'wiki.simple')
```

To see other available sources of pretrained word embeddings using the *fastText* algorithm,
we can call `nlp.embedding.list_sources`.

```{.python .input}
nlp.embedding.list_sources('fasttext')[:5]
```

The created vocabulary `vocab` includes five different words and a special
unknown token. Let us check the size of `vocab`.

```{.python .input}
len(vocab)
```

By default, the vector of any token that is unknown to `vocab` is the vector of `vocab.unk_token`.
Its length is equal to the vector dimensions of the fastText word embeddings:
(300,).

```{.python .input}
matrix[vocab['beautiful']].shape
```

Let us check the shape of the embedding of the words 'hello' from `vocab`.

```{.python .input}
matrix[vocab['hello']].shape
```

We can access the first five elements of the embedding of 'hello' and see that they are non-zero.

```{.python .input}
matrix[vocab['hello']][:5]
```

By default, the vector of any token that is in `vocab` but not in the pre-trained file
is a vector generated by by sampling from normal distribution 
with the same std and mean of the pre-trained embedding matrix.
Its length is equal to the vector dimensions of the fastText word embeddings:
(300,).

```{.python .input}
matrix[vocab['goodgod']].shape
```

We can access the first five elements of the embedding of 'goodgod'.

```{.python .input}
matrix[vocab['goodgod']][:5]
```

You can change the way to generate vectors for this kind of tokens by
specifying `unk_method` in `load_embeddings` function.
The `unk_method` is a function which receives `List[str]` 
and returns an embedding matrix(`numpy.ndarray`) for words not in the pre-trained file.
For example, 

```{.python .input}
def simple(words):
    return np.ones((len(words), 300))
matrix = nlp.embedding.load_embeddings(vocab, 'wiki.simple', unk_method=simple)
```

We can access the first five elements of the embedding of 'goodgod' and see that they are ones.

```{.python .input}
matrix[vocab['goodgod']][:5]
```

Sometimes we need to use `FastText` to compute vectors for Out-of-Vocabulary(OOV) words.
In this case, we provide `get_fasttext_model` to return a `FastText` model for you to use.

```{.python .input}
model = nlp.embedding.get_fasttext_model('wiki.en')
```

It will return a `fasttext.FastText._FastText` object, you can get more information 
about it from `fasttext.cc`.

Let us check the shape of the embedding of the OOV word 'goodgod'.

```{.python .input}
model['goodgod'].shape
```

We can access the first five elements of the embedding of 'goodgod'.

```{.python .input}
model['goodgod'][:5]
```

To see other available sources of the `FastText` model,
we can call `nlp.embedding.list_sources`.

```{.python .input}
nlp.embedding.list_sources('fasttext.bin')[:5]
```

### Using Pre-trained Word Embeddings in Gluon

To demonstrate how to use pre-trained word embeddings in Gluon, let us first obtain the indices of the words
'hello' and 'world'.

```{.python .input}
vocab['hello', 'world']
```

We can obtain the vectors for the words 'hello' and 'world' by specifying their
indices (5 and 4) and the weight or embedding matrix, which we get from 
`gluon.nn.Embedding`. We initialize a new layer and set the weights using the `layer.weight.set_data` method. Subsequently, we pull out the indices 5 and 4 from the weight vector and check their first five entries.

```{.python .input}
input_dim, output_dim = matrix.shape
layer = gluon.nn.Embedding(input_dim, output_dim)
layer.initialize()
layer.weight.set_data(matrix)
layer(nd.array([5, 4]))[:, :5]
```

### Creating Vocabulary from Pre-trained Word Embeddings

We can also create
vocabulary by using vocabulary of pre-trained word embeddings, such as GloVe.
Below are a few pre-trained file names under the GloVe word embedding.

```{.python .input}
nlp.embedding.list_sources('glove')[:5]
```

For simplicity of demonstration, we use a smaller word embedding file, such as
the 50-dimensional one. 
Now we create vocabulary by using all the tokens from `glove.6b.50d`.

```{.python .input}
matrix, vocab = nlp.embedding.load_embeddings(vocab=None, pretrained_name_or_dir='glove.6B.50d')
```

Below shows the size of `vocab` including a special unknown token.

```{.python .input}
len(vocab)
```

We can access attributes of `vocab`.

```{.python .input}
print(vocab['beautiful'])
print(vocab.all_tokens[71424])
```

## Applications of Word Embeddings

To apply word embeddings, we need to define
cosine similarity. Cosine similarity determines the similarity between two vectors.

```{.python .input}
import numpy as np
def cos_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
```

The range of cosine similarity between two vectors can be between -1 and 1. The
larger the value, the larger the similarity between the two vectors.

```{.python .input}
x = np.array([1, 2])
y = np.array([10, 20])
z = np.array([-1, -2])

print(cos_sim(x, y))
print(cos_sim(x, z))
```

### Word Similarity

Given an input word, we can find the nearest $k$ words from
the vocabulary (400,000 words excluding the unknown token) by similarity. The
similarity between any given pair of words can be represented by the cosine similarity
of their vectors.

We first must normalize each row, followed by taking the dot product of the entire
vocabulary embedding matrix and the single word embedding (`dot_prod`). 
We can then find the indices for which the dot product is greatest (`topk`), which happens to be the indices of the most similar words. 

```{.python .input}
def norm_vecs_by_row(x):
    return x / np.sqrt(np.sum(x * x, axis=1) + 1E-10).reshape((-1,1))

def topk(res, k):
    part = np.argpartition(res, k)
    return part[np.argsort(res[part])].tolist()

def get_knn(vocab, matrix, k, word):
    word_vec = matrix[vocab[word]].reshape((-1, 1))
    vocab_vecs = norm_vecs_by_row(matrix)
    dot_prod = np.dot(vocab_vecs, word_vec)
    indices = topk(dot_prod.reshape((len(vocab), )), k=k+1)
    # Remove unknown and input tokens.
    return vocab.to_tokens(indices[1:])
```

Let us find the 5 most similar words to 'baby' from the vocabulary (size:
400,000 words).

```{.python .input}
get_knn(vocab, matrix, 5, 'baby')
```

We can verify the cosine similarity of the vectors of 'baby' and 'babies'.

```{.python .input}
cos_sim(matrix[vocab['baby']], matrix[vocab['babies']])
```

Let us find the 5 most similar words to 'computers' from the vocabulary.

```{.python .input}
get_knn(vocab, matrix, 5, 'computers')
```

Let us find the 5 most similar words to 'run' from the given vocabulary.

```{.python .input}
get_knn(vocab, matrix, 5, 'run')
```

Let us find the 5 most similar words to 'beautiful' from the vocabulary.

```{.python .input}
get_knn(vocab, matrix, 5, 'beautiful')
```

### Word Analogy

We can also apply pre-trained word embeddings to the word
analogy problem. For example, "man : woman :: son : daughter" is an analogy.
This sentence can also be read as "A man is to a woman as a son is to a daughter."

The word analogy completion problem is defined concretely as: for analogy 'a : b :: c : d',
given the first three words 'a', 'b', 'c', find 'd'. The idea is to find the
most similar word vector for vec('c') + (vec('b')-vec('a')).

In this example,
we will find words that are analogous from the 400,000 indexed words in `vocab`.

```{.python .input}
def get_top_k_by_analogy(vocab, matrix, k, word1, word2, word3):
    word_vecs = [matrix[vocab[word]] for word in [word1, word2, word3]]
    word_diff = (word_vecs[1] - word_vecs[0] + word_vecs[2]).reshape((-1, 1))
    vocab_vecs = norm_vecs_by_row(matrix)
    dot_prod = np.dot(vocab_vecs, word_diff)
    indices = topk(dot_prod.reshape((len(vocab), )), k=k)
    return vocab.to_tokens(indices)
```

We leverage this method to find the word to complete the analogy 'man : woman :: son :'.

```{.python .input}
get_top_k_by_analogy(vocab, matrix, 1, 'man', 'woman', 'son')
```

Let us verify the cosine similarity between vec('son')+vec('woman')-vec('man')
and vec('daughter').

```{.python .input}
def cos_sim_word_analogy(vocab, matrix, word1, word2, word3, word4):
    words = [word1, word2, word3, word4]
    vecs = [matrix[vocab[word]] for word in words]
    return cos_sim(vecs[1] - vecs[0] + vecs[2], vecs[3])

cos_sim_word_analogy(vocab, matrix, 'man', 'woman', 'son', 'daughter')
```

And to perform some more tests, let's try the following analogy: 'beijing : china :: tokyo : '.

```{.python .input}
get_top_k_by_analogy(vocab, matrix, 1, 'beijing', 'china', 'tokyo')
```

And another word analogy: 'bad : worst :: big : '.

```{.python .input}
get_top_k_by_analogy(vocab, matrix, 1, 'bad', 'worst', 'big')
```

And the last analogy: 'do : did :: go :'.

```{.python .input}
get_top_k_by_analogy(vocab, matrix, 1, 'do', 'did', 'go')
```
