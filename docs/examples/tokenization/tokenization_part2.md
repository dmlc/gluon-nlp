# Part2: Learn Subword Models with GluonNLP

In this tutorial, you will learn the following:

- The basic idea about [Byte-pair Encoding (BPE)](https://arxiv.org/pdf/1508.07909.pdf), [SentencePiece](https://www.aclweb.org/anthology/D18-2012.pdf), and [Byte-level BPE](https://github.com/openai/gpt-2)
- Try out different subword models with `nlp_process learn_subword` and `nlp_process apply_subword`.

## Download Dataset

We will use the official evaluation set of WMT20 EN-DE and WMT20 EN-ZH tasks to illustration different algorithms. We choose to also add Chinese dataset because it uses **non-Latin alphabets** while English and German use Latin alphabets.


```{.shell .input}
!sacrebleu -t wmt20 -l en-de --echo src > test.en
!sacrebleu -t wmt20 -l en-de --echo ref > test.de
!sacrebleu -t wmt20 -l en-zh --echo ref > test.zh
```

Let's take a look at these three corpus:


```{.shell .input}
!head -n 2 test.de
```


```{.shell .input}
!head -n 2 test.en
```


```{.shell .input}
!head -n 2 test.zh
```

Chinese language does not have white space characters. We will first use the [jieba](https://github.com/fxsjy/jieba) tokenizer to tokenize it into words, which is the common practice for such scenarios.


```{.python .input}
from gluonnlp.data.tokenizers import JiebaTokenizer

tokenizer = JiebaTokenizer()
with open('test.zh.tok', 'w', encoding='utf-8') as out_f:
    with open('test.zh', 'r', encoding='utf-8') as in_f:
        for line in in_f:
            out_f.write(' '.join(tokenizer.encode(line)))
```

Let's view the tokenized Chinese corpus. You may notice that there are additional spaces between "前" and "保镖".


```{.shell .input}
!head -n 2 test.zh.tok
```

## Subword Learning Algorithms

### Byte-Pair Encoding (BPE)

All modern pretrained models use subword tokenizers. The advantage is that subword models can help balance the vocabulary size and the length of the encoded sequence. For example, let's first consider the extreme scenario, in which we represent the word as a sequence of characters:


```{.python .input}
s = "Sunnyvale"
characters = [ele for ele in s]
print(s)
print(characters)
```

Since there are far less number of unicode characters than the words in a corpus, we can greatly reduce the vocabulary size. The BPE algorithm improves upon this very basic idea. BPE tracks the most frequent bi-grams and merges this bigram into a new token. For example, assume the merge rule is:

'e', 'r' --> 'er'
'er', '</w>' --> 'er</w>'
'a', 't' --> 'at'

The word "later", will be converted to

['l', 'a', 't', 'e', 'r', '</w>']
['l', 'a', 't', 'er', '</w>']
['l', 'a', 't', 'er</w>']
['l', 'at', 'er</w>']


Let's train the BPE with the GluonNLP CLI. Internally, it is using the [subword-nmt](https://github.com/rsennrich/subword-nmt).


```{.shell .input}
!nlp_process learn_subword --model subword_nmt \
                           --corpus test.en test.de \
                           --vocab-size 5000 --save-dir subword_models
```

In addition, you can use `nlp_process apply_subword` to directly apply the learned BPE tokenizer to tokenize a corpus.


```{.shell .input}
!nlp_process apply_subword --model subword_nmt \
                           --model-path subword_models/subword_nmt.model \
                           --vocab-path subword_models/subword_nmt.vocab \
                           --corpus test.en \
                           --save-path test.en.subword_nmt
```


```{.shell .input}
!head -n 1 test.en.subword_nmt
```


```{.python .input}
from gluonnlp.data.tokenizers import SubwordNMTTokenizer

tokenizer = SubwordNMTTokenizer('subword_models/subword_nmt.model',
                                vocab='subword_models/subword_nmt.vocab')
```

Also, we can repeat the process on the Chinese corpus.


```{.shell .input}
!nlp_process learn_subword --model subword_nmt \
                           --corpus test.zh.tok \
                           --vocab-size 5000 --save-dir subword_models_zh
!nlp_process apply_subword --model subword_nmt \
                           --model-path subword_models_zh/subword_nmt.model \
                           --vocab-path subword_models_zh/subword_nmt.vocab \
                           --corpus test.zh.tok \
                           --save-path test.zh.subword_nmt

!head -n 1 test.zh.subword_nmt
```

### SentencePiece

Different from BPE, [SentencePiece](https://www.aclweb.org/anthology/D18-2012.pdf) learns the subwords based on the [Unigram language model](https://arxiv.org/pdf/1804.10959.pdf). SentencePiece offers a way for lossless tokenization. 

- "Hello World." --> ["Hello", "World", "."] --> ?

The model replaces the space to a special character `'_'` (U+2581)
- "Hello World." --> ["Hello", "\_World", "."] --> "Hello World."


```{.shell .input}
!nlp_process learn_subword --model spm \
                           --corpus test.en test.de \
                           --vocab-size 5000 \
                           --save-dir subword_models
!nlp_process apply_subword --model spm \
                           --model-path subword_models/spm.model \
                           --vocab-path subword_models/spm.vocab \
                           --corpus test.en \
                           --save-path test.en.spm
```


```{.shell .input}
!head -n 2 test.en.spm
```

### Byte-level BPE

The Byte-level BPE model was proposed in the [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf). The idea is to run BPE in the byte-level. Byte-BPE will **not have unknow token**. It can even tokenize the emojis '😀 😃 😄 😁 😆 😅'!

The general idea is to convert the input string into **bytes** rather than **unicodes**, and then learn BPE in the Byte-level. This is also suitable for dealing with web text because there are lots of different unicode characters in the web. In theory, there are more than 130,000 unicode characters. Let's see how to learn the Byte-level BPE with the `nlp_process learn_subword`. Internally, it calls the [huggingface/tokenizers](https://github.com/huggingface/tokenizers) package to achieve the goal.


```{.shell .input}
!nlp_process learn_subword --model hf_bytebpe \
                           --corpus test.en test.de \
                           --vocab-size 5000 \
                           --save-dir subword_models
```


```{.python .input}
from gluonnlp.data.tokenizers import HuggingFaceTokenizer, \
                                     SentencepieceTokenizer, \
                                     SubwordNMTTokenizer
tokenizer = HuggingFaceTokenizer('subword_models/hf_bytebpe.model',
                                 'subword_models/hf_bytebpe.vocab')
print(tokenizer.vocab)
print('Byte-BPE:', tokenizer.decode(tokenizer.encode('😃', int)))

bpe_tokenizer = SubwordNMTTokenizer('subword_models/subword_nmt.model',
                                    'subword_models/subword_nmt.vocab')
spm_tokenizer = SentencepieceTokenizer('subword_models/spm.model',
                                       'subword_models/spm.vocab')
print('BPE:', bpe_tokenizer.decode(bpe_tokenizer.encode('😃', int)))
print('SentencePiece:', spm_tokenizer.decode(spm_tokenizer.encode('😃', int)))
```


```{.shell .input}
!nlp_process apply_subword --model hf_bytebpe \
                           --model-path subword_models/hf_bytebpe.model \
                           --vocab-path subword_models/hf_bytebpe.vocab \
                           --corpus test.en \
                           --save-path test.en.hf_bytebpe
```


```{.shell .input}
!head -n 2 test.en.hf_bytebpe
```

Don't be afraid of 'Ġ'! It's a shifted version of space. 


```{.python .input}
chr(ord(' ') + 256)
```

### FAQ

You can find more details about how to use the `nlp_process` to clean up the data, learn subword model, and train a machine learning model based on Transformer in https://github.com/dmlc/gluon-nlp/tree/master/scripts/machine_translation.

Also, feel free to ask questions by submitting issues in GluonNLP!
