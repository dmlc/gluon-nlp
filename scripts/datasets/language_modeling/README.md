# Language Modeling Benchmark

Prepare the language modeling benchmarking datasets. 
In order to help reproduce the papers, we use 
the tokenized corpus as the training/validation/testing dataset.

- WikiText-2

```bash
nlp_data prepare_lm --dataset wikitext2
```

- WikiText-103
```bash
nlp_data prepare_lm --dataset wikitext103
```

- enwik8
```bash
nlp_data prepare_lm --dataset enwik8
```

- text8
```bash
nlp_data prepare_lm --dataset text8
```

- One-Billion-Word
```bash
nlp_data prepare_lm --dataset gbw
```

Happy language modeling :)
