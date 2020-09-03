# Conversion Scripts

In GluonNLP, we provide shared scripts to convert the model checkpoints in other repositories to GluonNLP.  

At this stage, the model needs to be downloaded locally, and the converting scripts accepts only the file directory as the argument,
without the support of accepting the url. In addition, both the tensorflow fine-tuned models that
can be loaded in TF1 Hub modules and TF2 SavedModels are accepted, although the parameters of mask
language model are not provided in TF2 SavedModels in most cases, and
the differences of these parameters are not required to be tested after converting.

The testing step mentioned above are controlled by the flag `--test`, in which the maximum
tolerance of 1e-3 between gluon model with converted weights and original tensorflow model.
In addition, we can use GPU in all converting scripts by adding `--gpu 0`.

For RoBERTa XLM-R and BART model, we rely on the master version of [fairseq](https://github.com/pytorch/fairseq#requirements-and-installation) package locally as `pip install git+https://github.com/pytorch/fairseq.git@master`.

## Convert all models

``bash
bash convert_all.sh
``

### BERT
Convert model from [BERT LIST](https://tfhub.dev/google/collections/bert/1).

You can use the script provided in [convert_bert.sh](convert_bert.sh).
The following command give you a rough idea about the code.

```bash
bash convert_bert.sh
```

In the process, we downloaded the config file from the [official repo](https://github.com/google-research/bert#pre-trained-models), download the configuration file `bert_config.json`,
and move it into `${case}_bert_${model}/assets/`.

### ALBERT
You can use the command described in
```bash
bash convert_albert.sh
```

### ELECTRA
The TF Hub is not available for ELECTRA model currently.
Thus, you will need to clone the [electra repository](https://github.com/ZheyuYe/electra)
and download the checkpoint. The parameters are converted from local checkpoints.
By running the following command, you can convert + verify the ELECTRA model with both the discriminator and the generator.

Notice: please set up the `--electra_path` with the cloned path if you'd like to directly use `convert_electra.py`.

```bash
bash convert_electra.sh
```

### MobileBert
```bash
bash convert_mobilebert.sh
```

### RoBERTa
```bash
bash convert_roberta.sh
```

### XLM-R
```bash
bash convert_xlmr.sh
```

### BART
```bash
bash convert_bart.sh
```

### GPT-2
```bash
bash convert_gpt2.sh
```
