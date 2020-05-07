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

## BERT
Convert model from [BERT LIST](https://tfhub.dev/google/collections/bert/1).

You can use the script provided in [convert_bert_from_tf_hub.sh](convert_bert_from_tf_hub.sh).
The following command give you a rough idea about the code.

```bash
bash convert_bert_from_tf_hub.sh
```

In the process, we downloaded the config file from the [official repo](https://github.com/google-research/bert#pre-trained-models), download the configuration file `bert_config.json`,
and move it into `${case}_bert_${model}/assets/`.

## ALBERT

```bash
for model in base large xlarge xxlarge
do
    mkdir albert_${model}_v2
    wget "https://tfhub.dev/google/albert_${model}/3?tf-hub-format=compressed" -O "albert_${model}_v3.tar.gz"
    tar -xvf albert_${model}_v3.tar.gz --directory albert_${model}_v2
    python convert_tf_hub_model.py --tf_hub_model_path albert_${model}_v2 --model_type albert --test
done
```
## RoBERTa

TBA

## ELECTRA
The TF Hub is not available for ELECTRA model currently.
Thus, you will need to clone the [electra repository](https://github.com/ZheyuYe/electra)
and download the checkpoint. The parameters are converted from local checkpoints.
By running the following command, you can convert + verify the ELECTRA model with both the discriminator and the generator.

Notice: pleas set up the `--electra_path` with the cloned path or get this electra repository packaged by `pip install -e .`.

```bash
# Need to use TF 1.13.2 to use contrib layer
pip uninstall tensorflow
pip install tensorflow==1.13.2

# Actual conversion
bash convert_electra.sh
```

## T5

TBA
