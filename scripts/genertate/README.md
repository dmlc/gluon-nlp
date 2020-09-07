Use the following command to generate gpt2 samples
```bash
python generate_unconditional_gpt2_samples.py \
    --model_name gpt2_774M
    --nsamples 5000 > samples
```

Some metrics for the unconditional generated text

|   GPT2 774M   |  Perplexity  |   Self-BLEU4   |Zipf Coefficient|  Repetition %  |
|---------------|--------------|----------------|----------------|----------------|
| openai        |              |                | -              |  -             |
| gluon         |              |  -             | -              |  -             |
|               |              |  -             | -              |  -             |

