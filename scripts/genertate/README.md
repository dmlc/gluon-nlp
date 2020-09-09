
Some of the examples below may include Unicode text characters. Set the environment variable:
```bash
export PYTHONIOENCODING=UTF-8
```

Use the following command to generate gpt2 samples
```bash
python3 generate_unconditional_gpt2_samples.py \
    --model_name gpt2_774M \
    --nsamples 5000 > samples
```

```bash
python3 interactive_conditional_gpt2_samples.py \
    --model_name gpt2_774M \
    --nsamples 1
```

```bash
python3 calculate_metrics.py samples
```

# 贴点sample

Some metrics for the unconditional generated text

|   GPT2 774M   |  Perplexity  |   Self-BLEU4   |Zipf Coefficient|  Repetition %  |
|---------------|--------------|----------------|----------------|----------------|
| openai        |              |                | -              |  -             |
| gluon         |              |  -             | -              |  -             |
|               |              |  -             | -              |  -             |

