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


Interactively generate gpt2 conditioanl samples
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
| pure sampling | 0.2701         | 0.9522         | 0.0            |
| original gpt2 | 0.2750         | 0.9512         | 0.0            |
| t=0.9         | 0.3683         | 0.9619         | 0.1            |
| topk=40       | 0.4291         | 0.9666         | 0.0            |
| topk=640      | 0.3384         | 0.9623         | 0.0            |
| topk=40 t=0.7 | 0.4621         | 0.9586         | 1.1            |


Part of some interesting generated unconditional example


A story
```
Looking back, Dil shook his head. The doll market was growing at an extraordinary rate; in his own opinion, it was unwarranted since his brother was sold to an abandoned bank. He was aware of what he had to do and was sure where his family was going; the thoughts worried him.

Although his brother had already grown an incredibly bulky gig paperback, he had never read a novel with an arguably more sinister turn. The intellectual gift of a child was reserved for reciting worked examples. As usual, exploiting loopholes, smart brother had practiced the art of overacting. Those tricks that remained medicinal classes grew weaker and smaller; in the end, one could not predict the fruition of those fighting skills.

Although he knew of a possible method of dealing with the right-winger, although he did not get his brother's hands on it, Regulus had already leaked his intentions in searching for Dil. He had already rushed passengers directly including that stupid bull. Due to the numerous setback, while Dil had luckily survived, he still suffered a decrease in his power.

He was reminded of the real reason why keeping secrets was not worth nothing; one must develop ones latent talents; in order to reverse one's stage of development all one had to do was give lessons to an opposite-type STUDENT that had similar abilities to those those that were bestowed by the parents; it was thus necessary to sift through the cat and mouse game over the years for those that had true deficiencies.
```

Code with comments
```
struct Read <T> {

index: usize ,

size: usize ,

}

extern crate glob;

/// A function indexed by some unique index (The &uniqueID Thing will become the

/// @value@).

struct Parse <T: Read, U: Read, D: Read, Tore: Read> {

index: usize ,

index64: usize ,

}

```
