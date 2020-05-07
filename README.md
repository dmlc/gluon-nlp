# GluonNLP + Numpy

Implementing NLP algorithms using the new numpy-like interface of MXNet. It's also a testbed for the next-generation release of GluonNLP.

This is a work-in-progress.


# Features

- Data Pipeline for NLP
- AutoML support
- Pretrained Model Zoo
- Fast Deployment
    - [TVM](https://tvm.apache.org/) (TODO)
- AWS Integration


# Installation
First of all, install the latest MXNet. You may use the following commands:

```bash

# Install the version with CUDA 10.1
pip install -U --pre mxnet-cu101 -f https://dist.mxnet.io/python

# Install the cpu-only version
pip install -U --pre mxnet -f https://dist.mxnet.io/python

# In case you do not have the permission, try the following
pip install -U --pre mxnet-cu101 -f https://dist.mxnet.io/python --user
```


To install, use

```bash
pip install -U -e .

# Also, it's highly recommended to install all the extra requirements via
pip install -U -e .[extras]
```

If you find that you do not have the permission, you can also install to the user folder:

```bash
pip install -U -e . --user
```

For Windows users, we recommend to use the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/about). 


# Access the Command-line Toolkits

To facilitate the researcher and the engineers, we provide command-line-toolkits for 
downloading and preprocessing the NLP datasets. For more details, you may refer to
 [GluonNLP Datasets](./scripts/datasets) and [GluonNLP Preprocessing Tools](./scripts/preprocess).

```bash
# CLI for downloading / preparing the dataset
nlp_data help

# CLI for accessing some common data preprocessing scripts
nlp_preprocess help

# Also, you can use `python -m` to access the toolkits
python -m gluonnlp.cli.data help
python -m gluonnlp.cli.preprocess help

```
# Run Unittests
You may go to [tests](tests) to see all the unittests.
