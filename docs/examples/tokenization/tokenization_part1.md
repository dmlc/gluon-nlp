# Part1: Basic Usage of Tokenizer and Vocabulary

In this tutorial, we will understand the basic usage of the tokenizer and vocabulary class in GluonNLP. These two components are quite essential in the text processing workflow:

raw text => normalized (cleaned) text => tokens => network


```{.python .input}
import gluonnlp
from gluonnlp.models import get_backbone
```

## Tokenizer Basics - Encode

Tokenization converts the raw sentence into a series of tokens. For example, let's consider two basic tokenizers, the `WhitespaceTokenizer` and the `MosesTokenizer`. We can simply call `tokenizer.encode()` to encode the sequence to a list of tokens.


```{.python .input}
from gluonnlp.data.tokenizers import WhitespaceTokenizer
from gluonnlp.data.tokenizers import MosesTokenizer
whitespace_tokenizer = WhitespaceTokenizer()
moses_tokenizer = MosesTokenizer('en')

sentence = '"Take CalTrain to Sunnyvale."'
print('Original Sentence:')
print(sentence)
print('Output of WhitespaceTokenizer:')
print(whitespace_tokenizer.encode(sentence))
print('Output of MosesTokenizer:')
print(moses_tokenizer.encode(sentence))
```

## Tokenizer Basics - Decode

To merge back (detokenize) a list of tokens to the original sentence, we can use `tokenizer.decode()`.


```{.python .input}
recovered_sentence = moses_tokenizer.decode(moses_tokenizer.encode(sentence))
print('Decoded Sentence=', recovered_sentence)
```

## Subword Tokenization

The idea of **Subword Tokenization** is widely adopted in state-of-the-art pretrained models. For example, BERT used the [WordPiece](https://arxiv.org/pdf/1609.08144.pdf) subword tokenization algorithm.

Let's load the subword-tokenizer in the BERT-cased model and see the output.


```{.python .input}
_, _, tokenizer, _, _ = get_backbone('google_en_cased_bert_base')
tokenizer.encode(sentence, str)
```

We can also access to the vocabulary of the WordPiece tokenizer used in BERT:


```{.python .input}
tokenizer.vocab
```

Here, the **[CLS]**, **[SEP]** are special tokens. We can fetch the id and value of these tokens via `vocab.cls_token`, `vocab.cls_id`, and `vocab.sep_token`, `vocab.sep_id`. Also, there is the unknow token.


```{.python .input}
print('cls_token = ', tokenizer.vocab.cls_token, ', cls_id = ', tokenizer.vocab.cls_id)
print('sep_token = ', tokenizer.vocab.sep_token, ', sep_id = ', tokenizer.vocab.sep_id)
print(tokenizer.encode('😁 means smile'))
```

## Encode With Offsets

In GluonNLP, to better facilitate span extraction applications, the tokenizers support the `encode_with_offset` functionality, which also returns the character-level offsets of the input sentence. This has been used in the QA tutorial.


```{.python .input}
encoded_tokens, offsets = tokenizer.encode_with_offsets(sentence, str)
print(encoded_tokens, offsets)
```


```{.python .input}
for token, offset in zip(encoded_tokens, offsets):
    print('token = {}\t sentence[{}:{}] = {}'.format(token, offset[0], offset[1], sentence[offset[0]:offset[1]]))
```

We also support to directly map the sentence to a list of integers by using `encode_with_offsets(sentence, int)`


```{.python .input}
tokenizer.encode_with_offsets(sentence, int)
```
