# Word Embeddings Training and Evaluation

## Evaluating Word Embeddings

The previous example has introduced how to load  pre-trained word embeddings from a set of sources included in the Gluon NLP toolkit. It was shown how make use of the word vectors to find the top most similar words of a given words or to solve the analogy task.

Besides manually investigating similar words or the predicted analogous words, we can facilitate word embedding evaluation datasets to quantify the evaluation.

Datasets for the *similarity* task come with a list of word pairs together with a human similarity judgement. The task is to recover the order of most-similar to least-similar pairs.

Datasets for the *analogy* tasks supply a set of analogy quadruples of the form  ‘a : b :: c : d’ and the task is to recover find the correct ‘d’ in as many cases as possible given just ‘a’, ‘b’, ‘c’. For instance, “man : woman :: son : daughter” is an analogy.

The Gluon NLP toolkit includes a set of popular *similarity* and *analogy* task datasets as well as helpers for computing the evaluation scores. Here we show how to make use of them.

```{.python .input  n=1}
# Workaround for https://github.com/apache/incubator-mxnet/issues/11314
%env MXNET_FORCE_ADDTAKEGRAD = 1
```

```{.json .output n=1}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "env: MXNET_FORCE_ADDTAKEGRAD=1\n"
 }
]
```

```{.python .input  n=2}
import time
import warnings
import logging
import random
warnings.filterwarnings('ignore')

import mxnet as mx
import gluonnlp as nlp
import numpy as np
from scipy import stats

# context = mx.cpu()  # Enable this to run on CPU
context = mx.gpu(0)  # Enable this to run on GPU
```

We first load pretrained FastText word embeddings.

```{.python .input  n=3}
embedding = nlp.embedding.create('fasttext', source='crawl-300d-2M')

vocab = nlp.Vocab(nlp.data.Counter(embedding.idx_to_token))
vocab.set_embedding(embedding)
```

### Word Similarity and Relatedness Task

Word embeddings should capture the relationsship between words in natural language.
In the Word Similarity and Relatedness Task word embeddings are evaluated by comparing word similarity scores computed from a pair of words with human labels for the similarity or relatedness of the pair.

`gluonnlp` includes a number of common datasets for the Word Similarity and Relatedness Task. The included datasets are listed in the [API documentation](http://gluon-nlp.mxnet.io/api/data.html#word-embedding-evaluation-datasets). We use several of them in the evaluation example below.

We first show a few samples from the WordSim353 dataset, to get an overall feeling of the Dataset structur

```{.python .input  n=4}
wordsim353 = nlp.data.WordSim353()
for i in range(15):
    print(*wordsim353[i], sep=', ')
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "computer, keyboard, 7.62\nJerusalem, Israel, 8.46\nplanet, galaxy, 8.11\ncanyon, landscape, 7.53\nOPEC, country, 5.63\nday, summer, 3.94\nday, dawn, 7.53\ncountry, citizen, 7.31\nplanet, people, 5.75\nenvironment, ecology, 8.81\nMaradona, football, 8.62\nOPEC, oil, 8.59\nmoney, bank, 8.5\ncomputer, software, 8.5\nlaw, lawyer, 8.38\n"
 }
]
```

### Similarity evaluator

The Gluon NLP toolkit includes a `WordEmbeddingSimilarity`  block, which predicts similarity score between word pairs given an embedding matrix.

```{.python .input  n=5}
evaluator = nlp.embedding.evaluation.WordEmbeddingSimilarity(
    idx_to_vec=vocab.embedding.idx_to_vec,
    similarity_function="CosineSimilarity")
evaluator.initialize(ctx=context)
evaluator.hybridize()
```

#### Evaluation: Running the task

```{.python .input  n=6}
words1, words2, scores = zip(*([vocab[d[0]], vocab[d[1]], d[2]] for d in wordsim353))
words1 = mx.nd.array(words1, ctx=context)
words2 = mx.nd.array(words2, ctx=context)
```

The similarities can be predicted by passing the two arrays of words through the evaluator. Thereby the *ith* word in `words1` will be compared with the *ith* word in `words2`.

```{.python .input  n=7}
pred_similarity = evaluator(words1, words2)
print(pred_similarity[:5])
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[0.4934404  0.69630307 0.5902223  0.31201977 0.16985895]\n<NDArray 5 @gpu(0)>\n"
 }
]
```

We can evaluate the predicted similarities, and thereby the word embeddings, by computing the Spearman Rank Correlation between the predicted similarities and the groundtruth, human, similarity scores from the dataset:

```{.python .input  n=8}
sr = stats.spearmanr(pred_similarity.asnumpy(), np.array(scores))
print('Spearman rank correlation on {}: {}'.format(wordsim353.__class__.__name__,
                                                   sr.correlation.round(3)))
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Spearman rank correlation on WordSim353: 0.792\n"
 }
]
```

### Word Analogy Task

In the Word Analogy Task word embeddings are evaluated by inferring an analogous word `D`, which is related to a given word `C` in the same way as a given pair of words `A, B` are related.

`gluonnlp` includes a number of common datasets for the Word Analogy Task. The included datasets are listed in the [API documentation](http://gluon-nlp.mxnet.io/api/data.html#word-embedding-evaluation-datasets). In this notebook we use the GoogleAnalogyTestSet dataset.


```{.python .input  n=9}
google_analogy = nlp.data.GoogleAnalogyTestSet()
```

We first demonstrate the structure of the dataset by printing a few examples

```{.python .input  n=10}
sample = []
print(('Printing every 1000st analogy question '
       'from the {} questions'
        'in the Google Analogy Test Set:').format(len(google_analogy)))
print('')
for i in range(0, 19544, 1000):
    print(*google_analogy[i])
    sample.append(google_analogy[i])
```

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Printing every 1000st analogy question from the 19544 questionsin the Google Analogy Test Set:\n\nathens greece baghdad iraq\nbaku azerbaijan dushanbe tajikistan\ndublin ireland kathmandu nepal\nlusaka zambia tehran iran\nrome italy windhoek namibia\nzagreb croatia astana kazakhstan\nphiladelphia pennsylvania tampa florida\nwichita kansas shreveport louisiana\nshreveport louisiana oxnard california\ncomplete completely lucky luckily\ncomfortable uncomfortable clear unclear\ngood better high higher\nyoung younger tight tighter\nweak weakest bright brightest\nslow slowing describe describing\nireland irish greece greek\nfeeding fed sitting sat\nslowing slowed decreasing decreased\nfinger fingers onion onions\nplay plays sing sings\n"
 }
]
```

```{.python .input  n=11}
words1, words2, words3, words4 = list(zip(*sample))
```

We restrict ourselves here to the first (most frequent) 300000 words of the pretrained embedding as well as all tokens that occur in the evaluation datasets as possible answers to the analogy questions.

```{.python .input  n=12}
import itertools

most_freq = 300000
counter = nlp.data.utils.Counter(embedding.idx_to_token[:most_freq])
google_analogy_tokens = set(itertools.chain.from_iterable((d[0], d[1], d[2], d[3]) for d in google_analogy))
counter.update(t for t in google_analogy_tokens if t in embedding)

vocab = nlp.vocab.Vocab(counter)
vocab.set_embedding(embedding)

print("Using most frequent {} + {} extra words".format(most_freq, len(vocab) - most_freq))


google_analogy_subset = [
    d for i, d in enumerate(google_analogy) if
    d[0] in vocab and d[1] in vocab and d[2] in vocab and d[3] in vocab
]
print('Dropped {} pairs from {} as they were OOV.'.format(
    len(google_analogy) - len(google_analogy_subset),
    len(google_analogy)))

google_analogy_coded = [[vocab[d[0]], vocab[d[1]], vocab[d[2]], vocab[d[3]]]
                 for d in google_analogy_subset]
google_analogy_coded_batched = mx.gluon.data.DataLoader(
    google_analogy_coded, batch_size=256)
```

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Using most frequent 300000 + 96 extra words\nDropped 1781 pairs from 19544 as they were OOV.\n"
 }
]
```

```{.python .input  n=13}
evaluator = nlp.embedding.evaluation.WordEmbeddingAnalogy(
    idx_to_vec=vocab.embedding.idx_to_vec,
    exclude_question_words=True,
    analogy_function="ThreeCosMul")
evaluator.initialize(ctx=context)
evaluator.hybridize()
```

To show a visual progressbar, make sure the `tqdm` package is installed.

```{.python .input  n=14}
# ! pip install  tqdm
import sys
# workaround for deep learning AMI on EC2
sys.path.append('/home/ubuntu/anaconda3/envs/python3/lib/python3.6/site-packages')
```

```{.python .input  n=15}
try:
    import tqdm
except:
    tqdm = None

acc = mx.metric.Accuracy()

if tqdm is not None:
    google_analogy_coded_batched = tqdm.tqdm(google_analogy_coded_batched)
for batch in google_analogy_coded_batched:
    batch = batch.as_in_context(context)
    words1, words2, words3, words4 = (batch[:, 0], batch[:, 1],
                                      batch[:, 2], batch[:, 3])
    pred_idxs = evaluator(words1, words2, words3)
    acc.update(pred_idxs[:, 0], words4.astype(np.float32))

print('Accuracy on %s: %s'% (google_analogy.__class__.__name__, acc.get()[1].round(3)))
```

```{.json .output n=15}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 70/70 [00:34<00:00,  2.02it/s]"
 },
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Accuracy on GoogleAnalogyTestSet: 0.772\n"
 },
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "\n"
 }
]
```

## Training word embeddings

Besides loading pre-trained word embeddings, the toolkit also facilitates training word embedding models with your own datasets. `gluonnlp` provides trainable Blocks for a simple word-level embedding model and the popular FastText embedding model.

### Loading the training data

We can load a word embedding training dataset from the datasets provided by the `gluonnlp` toolkit.

Word embedding training datasets are structured as a nested list. The outer list represents sentences in the corpus. The inner lists represents the words in each sentence.

We then build a vocabulary of all the tokens in the dataset that occur more than 5 times and code the dataset, ie. replace the words with their indices.

```{.python .input  n=16}
frequent_token_subsampling = 1E-4

import itertools
dataset = nlp.data.Text8(segment='train')

counter = nlp.data.count_tokens(itertools.chain.from_iterable(dataset))
vocab = nlp.Vocab(counter, unknown_token=None, padding_token=None,
                  bos_token=None, eos_token=None, min_freq=5)
idx_to_counts = np.array([counter[w] for w in vocab.idx_to_token])
f = idx_to_counts / np.sum(idx_to_counts)
idx_to_pdiscard = 1 - np.sqrt(frequent_token_subsampling / f)

coded_dataset = [[vocab[token] for token in sentence
                  if token in vocab
                  and random.uniform(0, 1) > idx_to_pdiscard[vocab[token]]] for sentence in dataset]
```

### Trainable embedding model

A word embedding model associates words with word vectors. Each word is represented by it's vocabulary index and the embedding model associates these indices with vectors.

`gluonnlp` provides Blocks for simple embedding models as well as models that take into account subword information (covered later). A variety of loss functions exist to train word embedding models. The Skip-Gram objective is a simple and popular objective which we use in this notebook.
It was introduced by "Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. ICLR Workshop , 2013."

The Skip-Gram objective trains word vectors such that the word vector of a word at some position in a sentence can best predict the surrounding words. We call these words *center* and *context* words.

![Skip-Gram model](http://blog.aylien.com/wp-content/uploads/2016/10/skip-gram.png)

Skip-Gram and picture from "Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. ICLR Workshop , 2013."


For the Skip-Gram objective, we initialize two embedding models: `embedding` and `embedding_out`. `embedding` is used to look up embeddings for the *center* words. `embedding_out` is used for the *context* words.

The weights of `embedding` are the final word embedding weights.

```{.python .input  n=17}
emsize = 300
embedding = nlp.model.train.SimpleEmbeddingModel(
    token_to_idx=vocab.token_to_idx,
    embedding_size=emsize,
    weight_initializer=mx.init.Uniform(scale=1 / emsize))
embedding_out = nlp.model.train.SimpleEmbeddingModel(
    token_to_idx=vocab.token_to_idx,
    embedding_size=emsize,
    weight_initializer=mx.init.Uniform(scale=1 / emsize))

embedding.initialize(ctx=context)
embedding_out.initialize(ctx=context)
embedding.hybridize(static_alloc=True)
embedding_out.hybridize(static_alloc=True)

params = list(embedding.collect_params().values()) + \
    list(embedding_out.collect_params().values())
trainer = mx.gluon.Trainer(params, 'adagrad', dict(learning_rate=0.05))
```

### Training objective

#### Naive objective

To naively maximize the Skip-Gram objective, if we sample a center word we need to compute a prediction for every other word in the vocabulary if it occurs in the context of the center word or not. We can then backpropagate and update the parameters to make the prediction of the correct *context* words more likely and of all other words less likely.


However, this naive method is computationally very expensive as it requires computing a Softmax function over all words in the vocabulary. Instead, "Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. ICLR Workshop , 2013." introduced *Negative Sampling*.

#### Negative sampling

*Negative Sampling* means that instead of using a small number of *correct* (or *positive*) *context* and all other (*negative*) words to compute the loss and update the parameters we may choose a small, constant number of *negative* words at random. Negative words are choosen randomly based on their frequency in the training corpus. It is recommend to smoothen the frequency distribution by the factor `0.75`.

`gluonnlp` includes a `ContextSampler` and `NegativeSampler`. Once initialized, we can iterate over them to get batches of *center* and *context* words from the `ContextSampler` as well as batches of *negatives* from the `NegativeSampler`.

The `ContextSampler` can be initialized with the word embedding training dataset, a batch size and the window size specifying the number of words before and after the *center* word to consider as part of the context. (It is recommended to shuffle the sentences in the dataset before initializing the ContextSampler.)

`NegativeSampler` takes a vocabulary with counts, the batch size, the number of samples to consider as well as a smoothing constant.

```{.python .input  n=18}
context_sampler = nlp.data.ContextSampler(coded=coded_dataset, batch_size=2048, window=5)

negatives_weights = mx.nd.array([counter[w] for w in vocab.idx_to_token])**0.75
negatives_sampler = nlp.data.UnigramCandidateSampler(negatives_weights)
```

To train a model with the *center*, *context* and *negative* batches, we use a `SigmoidBinaryCrossEntropyLoss`.

```{.python .input  n=19}
loss_function = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()
```

```{.python .input  n=20}
# The context sampler exposes the number of batches
# in the training dataset as it's length
num_batches = len(context_sampler)
num_negatives = 5

# Logging variables
log_interval = 500
log_wc = 0
log_start_time = time.time()
log_avg_loss = 0

# We iterate over all batches in the context_sampler
for i, batch in enumerate(context_sampler):
    # Each batch from the context_sampler includes
    # a batch of center words, their contexts as well
    # as a mask as the contexts can be of varying lengths
    (center, word_context, word_context_mask) = batch

    negatives_shape = (word_context.shape[0],
                       word_context.shape[1] * num_negatives)
    negatives, negatives_mask = negatives_sampler(
        negatives_shape, word_context, word_context_mask)

    # We copy all data to the GPU
    center = center.as_in_context(context)
    word_context = word_context.as_in_context(context)
    word_context_mask = word_context_mask.as_in_context(context)
    negatives = negatives.as_in_context(context)
    negatives_mask = negatives_mask.as_in_context(context)


    # We concatenate the positive context words and negatives
    # to a single ndarray
    word_context_negatives = mx.nd.concat(word_context, negatives, dim=1)
    word_context_negatives_mask = mx.nd.concat(word_context_mask, negatives_mask, dim=1)

    # We record the gradient of one forward pass
    with mx.autograd.record():
        # 1. Compute the embedding of the center words
        emb_in = embedding(center)

        # 2. Compute the context embedding
        emb_out = embedding_out(word_context_negatives,
                                word_context_negatives_mask)

        # 3. Compute the prediction
        # To predict if a context work is likely or not, the dot product
        # between the word vector of the center word and the output weights
        # of the context / negative words is computed and passed through a
        # Sigmoid function
        pred = mx.nd.batch_dot(emb_in, emb_out.swapaxes(1, 2))
        pred = pred.squeeze() * word_context_negatives_mask
        label = mx.nd.concat(word_context_mask, mx.nd.zeros_like(negatives), dim=1)

        # 4. Compute the Loss function (SigmoidBinaryCrossEntropyLoss)
        loss = loss_function(pred, label)

    # Compute the gradient
    loss.backward()

    # Update the parameters
    trainer.step(batch_size=1)

    # Logging
    log_wc += loss.shape[0]
    log_avg_loss += loss.mean()
    if (i + 1) % log_interval == 0:
        wps = log_wc / (time.time() - log_start_time)
        # Forces waiting for computation by computing loss value
        log_avg_loss = log_avg_loss.asscalar() / log_interval
        print('[Batch {}/{}] loss={:.4f}, '
                     'throughput={:.2f}K wps, wc={:.2f}K'.format(
                         i + 1, num_batches, log_avg_loss,
                         wps / 1000, log_wc / 1000))
        log_start_time = time.time()
        log_avg_loss = 0
        log_wc = 0
```

```{.json .output n=20}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[Batch 500/4118] loss=0.3939, throughput=125.90K wps, wc=1024.00K\n[Batch 1000/4118] loss=0.3641, throughput=181.37K wps, wc=1024.00K\n[Batch 1500/4118] loss=0.3553, throughput=189.75K wps, wc=1024.00K\n[Batch 2000/4118] loss=0.3509, throughput=186.19K wps, wc=1024.00K\n[Batch 2500/4118] loss=0.3464, throughput=180.96K wps, wc=1024.00K\n[Batch 3000/4118] loss=0.3440, throughput=187.66K wps, wc=1024.00K\n[Batch 3500/4118] loss=0.3426, throughput=181.33K wps, wc=1024.00K\n[Batch 4000/4118] loss=0.3393, throughput=176.82K wps, wc=1024.00K\n"
 }
]
```

### Evaluation of trained embedding

As we have only obtained word vectors for words that occured in the training corpus,
we filter the evaluation dataset and exclude out of vocabulary words.

```{.python .input  n=21}
words1, words2, scores = zip(*([vocab[d[0]], vocab[d[1]], d[2]]
    for d in wordsim353  if d[0] in vocab and d[1] in vocab))
words1 = mx.nd.array(words1, ctx=context)
words2 = mx.nd.array(words2, ctx=context)
```

We create a new `TokenEmbedding` object and set the embedding vectors for the words we care about for evaluation.

```{.python .input  n=22}
token_embedding = nlp.embedding.TokenEmbedding(unknown_token=None)
token_embedding[vocab.idx_to_token] = embedding[vocab.idx_to_token]

evaluator = nlp.embedding.evaluation.WordEmbeddingSimilarity(
    idx_to_vec=token_embedding.idx_to_vec,
    similarity_function="CosineSimilarity")
evaluator.initialize(ctx=context)
evaluator.hybridize()
```

```{.python .input  n=23}
pred_similarity = evaluator(words1, words2)
sr = stats.spearmanr(pred_similarity.asnumpy(), np.array(scores))
print('Spearman rank correlation on {} pairs of {} (total {}): {}'.format(
    len(words1), wordsim353.__class__.__name__, len(wordsim353), sr.correlation.round(3)))
```

```{.json .output n=23}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Spearman rank correlation on 432 pairs of WordSim353 (total 455): 0.546\n"
 }
]
```

## Unknown token handling and subword information

Sometimes we may run into a word for which the embedding does not include a word vector. While the `vocab` object is happy to replace it with a special index for unknown tokens.


```{.python .input  n=24}
print('Is "hello" known? ', 'hello' in vocab)
print('Is "likelyunknown" known? ', 'likelyunknown' in vocab)
```

```{.json .output n=24}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Is \"hello\" known?  True\nIs \"likelyunknown\" known?  False\n"
 }
]
```

Some embedding models such as the FastText model support computing word vectors for unknown words by taking into account their subword units.



- Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. ICLR Workshop , 2013.

## Training word embeddings with subword information

`gluonnlp` provides the concept of a SubwordFunction which maps words to a list of indices representing their subword.
Possible SubwordFunctions include mapping a word to the sequence of it's characters/bytes or hashes of all its ngrams.

FastText models use a hash function to map each ngram of a word to a number in range `[0, num_subwords)`. We include the same hash function.

### Concept of a SubwordFunction

```{.python .input  n=25}
subword_function = nlp.vocab.create_subword_function(
    'NGramHashes', ngrams=[3, 4, 5, 6], num_subwords=500000)

idx_to_subwordidxs = subword_function(vocab.idx_to_token)
for word, subwords in zip(vocab.idx_to_token[:3], idx_to_subwordidxs[:3]):
    print('<'+word+'>', subwords, sep = '\t')
```

```{.json .output n=25}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "<the>\t\n[151151. 361980. 316280. 409726.  60934. 148960.]\n<NDArray 6 @cpu(0)>\n<of>\t\n[497102. 228930. 164528.]\n<NDArray 3 @cpu(0)>\n<and>\t\n[378080. 395046. 125443. 235020. 119624.  30390.]\n<NDArray 6 @cpu(0)>\n"
 }
]
```

As words are of varying length, we have to pad the lists of subwords to obtain a batch. To distinguish padded values from valid subword indices we use a mask.
We first pad the subword arrays with `-1`, compute the mask and change the `-1` entries to some valid subword index (here `0`).

```{.python .input  n=26}
subword_padding = nlp.data.batchify.Pad(pad_val=-1)

subwords = subword_padding(idx_to_subwordidxs[:3])
subwords_mask = subwords != -1
subwords += subwords == -1  # -1 is invalid. Change to 0
print(subwords)
print(subwords_mask)
```

```{.json .output n=26}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "\n[[151151. 361980. 316280. 409726.  60934. 148960.]\n [497102. 228930. 164528.      0.      0.      0.]\n [378080. 395046. 125443. 235020. 119624.  30390.]]\n<NDArray 3x6 @cpu_shared(0)>\n\n[[1. 1. 1. 1. 1. 1.]\n [1. 1. 1. 0. 0. 0.]\n [1. 1. 1. 1. 1. 1.]]\n<NDArray 3x6 @cpu(0)>\n"
 }
]
```

To enable fast training, we precompute the mapping from the  words in  our training corpus to  the  subword indices.

```{.python .input  n=27}
# Precompute a idx to subwordidxs mapping to support fast lookup
idx_to_subwordidxs = list(subword_function(vocab.idx_to_token))
max_subwordidxs_len = max(len(s) for s in idx_to_subwordidxs)

# Padded max_subwordidxs_len + 1 so each row contains at least one -1
# element which can be found by np.argmax below.
idx_to_subwordidxs = np.stack(
    np.pad(b.asnumpy(), (0, max_subwordidxs_len - len(b) + 1), \
           constant_values=-1, mode='constant')
    for b in idx_to_subwordidxs).astype(np.float32)
idx_to_subwordidxs = mx.nd.array(idx_to_subwordidxs)

def indices_to_subwordindices_mask(indices, idx_to_subwordidxs):
    """Return array of subwordindices for indices.

    A padded numpy array and a mask is returned. The mask is used as
    indices map to varying length subwords.

    Parameters
    ----------
    indices : list of int, numpy array or mxnet NDArray
        Token indices that should be mapped to subword indices.

    Returns
    -------
    Array of subword indices.

    """
    if not isinstance(indices, mx.nd.NDArray):
        indices = mx.nd.array(indices)
    subwords = idx_to_subwordidxs[indices]
    mask = mx.nd.zeros_like(subwords)
    mask += subwords != -1
    lengths = mx.nd.argmax(subwords == -1, axis=1)
    subwords += subwords == -1

    new_length = int(max(mx.nd.max(lengths).asscalar(), 1))
    subwords = subwords[:, :new_length]
    mask = mask[:, :new_length]

    return subwords, mask
```

### The model

Instead of the `SimpleEmbeddingModel` we now train a `FasttextEmbeddingModel` Block which can combine the word and subword information.

```{.python .input  n=28}
emsize = 300
embedding = nlp.model.train.FasttextEmbeddingModel(
    token_to_idx=vocab.token_to_idx,
    subword_function=subword_function,
    embedding_size=emsize,
    weight_initializer=mx.init.Uniform(scale=1 / emsize))
embedding_out = nlp.model.train.SimpleEmbeddingModel(
    token_to_idx=vocab.token_to_idx,
    embedding_size=emsize,
    weight_initializer=mx.init.Uniform(scale=1 / emsize))
loss_function = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()

embedding.initialize(ctx=context)
embedding_out.initialize(ctx=context)
embedding.hybridize(static_alloc=True)
embedding_out.hybridize(static_alloc=True)

params = list(embedding.collect_params().values()) + \
    list(embedding_out.collect_params().values())
trainer = mx.gluon.Trainer(params, 'adagrad', dict(learning_rate=0.05))
```

### Training

Compared to training the `SimpleEmbeddingModel`, we now also look up the subwords of each center word in the batch and pass the subword infor

```{.python .input  n=29}
num_batches = len(context_sampler)
num_negatives = 5

# Logging variables
log_interval = 500
log_wc = 0
log_start_time = time.time()
log_avg_loss = 0

# We iterate over all batches in the context_sampler
for i, batch in enumerate(context_sampler):
    (center, word_context, word_context_mask) = batch

    negatives_shape = (word_context.shape[0],
                       word_context.shape[1] * num_negatives)
    negatives, negatives_mask = negatives_sampler(
        negatives_shape, word_context, word_context_mask)


    # Get subwords for all unique words in the batch
    unique, inverse_unique_indices = np.unique(
        center.asnumpy(), return_inverse=True)
    unique = mx.nd.array(unique)
    inverse_unique_indices = mx.nd.array(
        inverse_unique_indices, ctx=context)
    subwords, subwords_mask = indices_to_subwordindices_mask(unique, idx_to_subwordidxs)

    # To GPU
    center = center.as_in_context(context)
    subwords = subwords.as_in_context(context)
    subwords_mask = subwords_mask.as_in_context(context)
    word_context_negatives = mx.nd.concat(word_context, negatives, dim=1).as_in_context(context)
    word_context_negatives_mask = mx.nd.concat(word_context_mask, negatives_mask, dim=1).as_in_context(context)
    word_context_mask = word_context_mask.as_in_context(context)

    with mx.autograd.record():
        emb_in = embedding(center, subwords, subwordsmask=subwords_mask,
                           words_to_unique_subwords_indices=inverse_unique_indices)
        emb_out = embedding_out(word_context_negatives, word_context_negatives_mask)

        # Compute loss
        pred = mx.nd.batch_dot(emb_in, emb_out.swapaxes(1, 2))
        pred = pred.squeeze() * word_context_negatives_mask
        label = mx.nd.concat(word_context_mask, mx.nd.zeros(negatives.shape, ctx=context), dim=1)

        loss = loss_function(pred, label)

    loss.backward()
    trainer.step(batch_size=1)

    # Logging
    log_wc += loss.shape[0]
    log_avg_loss += loss.mean()
    if (i + 1) % log_interval == 0:
        wps = log_wc / (time.time() - log_start_time)
        # Forces waiting for computation by computing loss value
        log_avg_loss = log_avg_loss.asscalar() / log_interval
        print('[Batch {}/{}] loss={:.4f}, '
                     'throughput={:.2f}K wps, wc={:.2f}K'.format(
                         i + 1, num_batches, log_avg_loss,
                         wps / 1000, log_wc / 1000))
        log_start_time = time.time()
        log_avg_loss = 0
        log_wc = 0
```

```{.json .output n=29}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[Batch 500/4118] loss=0.3667, throughput=89.84K wps, wc=1024.00K\n[Batch 1000/4118] loss=0.3515, throughput=91.00K wps, wc=1024.00K\n[Batch 1500/4118] loss=0.3461, throughput=113.12K wps, wc=1024.00K\n[Batch 2000/4118] loss=0.3439, throughput=115.73K wps, wc=1024.00K\n[Batch 2500/4118] loss=0.3413, throughput=121.04K wps, wc=1024.00K\n[Batch 3000/4118] loss=0.3394, throughput=120.04K wps, wc=1024.00K\n[Batch 3500/4118] loss=0.3383, throughput=120.54K wps, wc=1024.00K\n[Batch 4000/4118] loss=0.3359, throughput=127.55K wps, wc=1024.00K\n"
 }
]
```

### Evaluation

Thanks to the subword support of the `FasttextEmbeddingModel` we can now evaluate on all words in the evaluation dataset, not only the ones that we observed during training (the `SimpleEmbeddingModel` only provides vectors for words observed at training).

We first find the all tokens in the evaluation dataset and then convert the `FasttextEmbeddingModel` to a `TokenEmbedding` with exactly those tokens.

```{.python .input  n=30}
wordsim353_tokens  = list(set(itertools.chain.from_iterable((d[0], d[1]) for d in wordsim353)))
token_embedding = nlp.embedding.TokenEmbedding(unknown_token=None)
token_embedding[wordsim353_tokens] = embedding[wordsim353_tokens]

print('There are', len(wordsim353_tokens), 'unique tokens in WordSim353')
print('The imputed TokenEmbedding has shape', token_embedding.idx_to_vec.shape)
```

```{.json .output n=30}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "There are 437 unique tokens in WordSim353\nThe imputed TokenEmbedding has shape (437, 300)\n"
 }
]
```

```{.python .input  n=31}
evaluator = nlp.embedding.evaluation.WordEmbeddingSimilarity(
    idx_to_vec=token_embedding.idx_to_vec,
    similarity_function="CosineSimilarity")
evaluator.initialize(ctx=context)
evaluator.hybridize()
```

```{.python .input  n=32}
words1, words2, scores = zip(*([token_embedding.token_to_idx[d[0]],
                                token_embedding.token_to_idx[d[1]],
                                d[2]] for d in wordsim353))
words1 = mx.nd.array(words1, ctx=context)
words2 = mx.nd.array(words2, ctx=context)
```

```{.python .input  n=33}
pred_similarity = evaluator(words1, words2)
sr = stats.spearmanr(pred_similarity.asnumpy(), np.array(scores))
print('Spearman rank correlation on {} pairs of {}: {}'.format(
    len(words1), wordsim353.__class__.__name__, sr.correlation.round(3)))
```

```{.json .output n=33}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Spearman rank correlation on 455 pairs of WordSim353: 0.472\n"
 }
]
```

## Loading pretrained FastText models with subword information

As the `FasttextEmbeddingModel` in `gluonnlp` uses the same structure as the models provided by `facebookresearch/fasttext` it is possible to load models trained by `facebookresearch/fasttext` into the `FasttextEmbeddingModel`.

```{.python .input  n=34}
embedding = nlp.model.train.FasttextEmbeddingModel.load_fasttext_format('/home/ubuntu/skipgram-text8.bin')
token_embedding = nlp.embedding.TokenEmbedding(unknown_token=None)
token_embedding[wordsim353_tokens] = embedding[wordsim353_tokens]

evaluator = nlp.embedding.evaluation.WordEmbeddingSimilarity(
    idx_to_vec=token_embedding.idx_to_vec,
    similarity_function="CosineSimilarity")
evaluator.initialize(ctx=context)
evaluator.hybridize()

pred_similarity = evaluator(words1, words2)
sr = stats.spearmanr(pred_similarity.asnumpy(), np.array(scores))
print('Spearman rank correlation on {} pairs of {}: {}'.format(
    len(words1), wordsim353.__class__.__name__, sr.correlation.round(3)))
```

```{.json .output n=34}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Spearman rank correlation on 455 pairs of WordSim353: 0.584\n"
 }
]
```
