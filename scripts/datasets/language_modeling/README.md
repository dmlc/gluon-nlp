# Language Modeling Benchmark

Prepare the language modeling benchmarking datasets. 
In order to help reproduce the papers, we use 
the tokenized corpus as the training/validation/testing dataset.

```bash
# WikiText-2
nlp_data prepare_lm --dataset wikitext2

# WikiText-103
nlp_data prepare_lm --dataset wikitext103

# enwik8
nlp_data prepare_lm --dataset enwik8

# Text-8
nlp_data prepare_lm --dataset text8

# Google One-Billion-Word
nlp_data prepare_lm --dataset gbw
```

Happy language modeling :)
