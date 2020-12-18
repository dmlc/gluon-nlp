# Part3: Download Data from Wikipedia and Learn Subword

In this tutorial, we will download the Wikipedia classical Chinese dataset with `nlp_data` and learn a customized sentencepiece vocabulary.

## Download Data

```{.shell .input}
!nlp_data prepare_wikipedia --mode download+format --lang zh-classical --date latest --quiet -o wiki_zh_classical
```

To save time, we will use the first 10000 sentences for training the subword model.


```{.shell .input}
!head -10000 wiki_zh_classical/prepared_wikipedia/wikipedia-prepared-0000.txt > train_corpus.txt
```

```{.shell .input}
!nlp_data prepare_wikipedia --mode download+format --lang zh-classical --date latest --quiet -o wikicorpus_zh_classical
```


```{.shell .input}
!nlp_process learn_subword --model spm --corpus train_corpus.txt --vocab-size 10000 \
                           --disable-bos --disable-eos \
                           --custom-special-tokens "cls_token=<cls>" "sep_token=<sep>"
```

The model are saved in "spm" folder.

```{.shell .input}
!ls spm
!less spm/spm.vocab
```

## Build the Tokenizer with the Saved Model


```{.python .input}
import gluonnlp
import json
from gluonnlp.data.tokenizers import SentencepieceTokenizer
tokenizer = SentencepieceTokenizer(model_path='spm/spm.model', vocab="spm/spm.vocab")
print(tokenizer)
print()
print('The first 10 tokens in the vocabulary:')
print('--------------------------------------')
print(tokenizer.vocab.all_tokens[:10])
```

You can use the tokenizer direclty.


```{.python .input}
tokenizer.encode('賈夫人仙逝揚州城 ·')
```


```{.python .input}
tokenizer.encode_with_offsets('賈夫人仙逝揚州城 ·')
```

## Explore More Options

To explore more options, you may check the README.


```{.shell .input}
!nlp_process learn_subword --help
```
