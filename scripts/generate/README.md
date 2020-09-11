
Some of the examples below may include Unicode text characters. Set the environment variable:
```bash
export PYTHONIOENCODING=UTF-8
```

Use the following command to generate gpt2 unconditional samples
```bash
python3 generate_unconditional_gpt2_samples.py \
    --model_name gpt2_774M \
    --gpu 0 \
    --temperature 0.7 \
    --top_k 40 \
    --nsamples 1000 > samples
```


Interactive generate gpt2 conditioanl samples
```bash
python3 interactive_conditional_gpt2_samples.py \
    --model_name gpt2_774M \
    --nsamples 1
```

Calculate some metrics in https://arxiv.org/pdf/1904.09751.pdf. 
These metrics are just heuristics and there is no guarantee that they correlates well with human evaluation.
```bash
python3 calculate_metrics.py \
    --file samples
```


Some metrics for the unconditional generated text

|   GPT2 774M   |   Self-BLEU4   |Zipf Coefficient|  Repetition %  |
|---------------|----------------|----------------|----------------|
| pure sampling |                |                | -              |
| original gpt2 |                |  -             | -              |
| t=0.9         |                |  -             | -              |
| topk=40       |                |  -             | -              |
| topk=640      |                |  -             | -              |
| topk=40 t=0.7 |                |  -             | -              |


Some interesting generated unconditional samples
# TODO