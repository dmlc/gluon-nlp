<h2 align="center">
<a href="https://github.com/dmlc/gluon-nlp"><img src="https://raw.githubusercontent.com/dmlc/gluon-nlp/master/docs/_static/gluon-logo.svg" alt="GluonNLP Logo" width="500"/></a>
</h2>

<h2 align="center">
GluonNLP: Your Choice of Deep Learning for NLP
</h2>

<p align="center">
    <a href="https://github.com/dmlc/gluon-nlp/actions"><img src="https://github.com/dmlc/gluon-nlp/workflows/continuous%20build/badge.svg"></a>
    <a href="https://github.com/dmlc/gluon-nlp/actions"><img src="https://github.com/dmlc/gluon-nlp/workflows/continuous%20build%20-%20gpu/badge.svg"></a>
    <a href="https://codecov.io/gh/dmlc/gluon-nlp"><img src="https://codecov.io/gh/dmlc/gluon-nlp/branch/master/graph/badge.svg"></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
    <a href="https://github.com/dmlc/gluonnlp/actions"><img src="https://img.shields.io/badge/python-3.6%2C3.7%2C3.8-blue.svg"></a>
    <a href="https://pypi.org/project/gluonnlp/#history"><img src="https://img.shields.io/pypi/v/gluonnlp.svg"></a>
</p>

GluonNLP is a toolkit that helps you solve NLP problems. It provides easy-to-use tools that helps you load the text data, 
process the text data, and train models.

See our documents at https://nlp.gluon.ai/master/index.html.

# Features

- Easy-to-use Text Processing Tools and Modular APIs
- Pretrained Model Zoo
- Write Models with Numpy-like API
- Fast Inference via [Apache TVM (incubating)](https://tvm.apache.org/) (Experimental)
- AWS Integration via [SageMaker](https://aws.amazon.com/sagemaker/)


# Installation
First of all, install the MXNet 2 release such as MXNet 2 Alpha. You may use the
following commands:

```bash
# Install the version with CUDA 10.2
python3 -m pip install -U --pre "mxnet-cu102>=2.0.0a"

# Install the version with CUDA 11
python3 -m pip install -U --pre "mxnet-cu110>=2.0.0a"

# Install the cpu-only version
python3 -m pip install -U --pre "mxnet>=2.0.0a"
```


To install GluonNLP, use

```bash
python3 -m pip install -U -e .

# Also, you may install all the extra requirements via
python3 -m pip install -U -e ."[extras]"
```

If you find that you do not have the permission, you can also install to the user folder:

```bash
python3 -m pip install -U -e . --user
```

For Windows users, we recommend to use the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/about).


# Access the Command-line Toolkits

To facilitate both the engineers and researchers, we provide command-line-toolkits for
downloading and processing the NLP datasets. For more details, you may refer to
 [GluonNLP Datasets](./scripts/datasets) and [GluonNLP Data Processing Tools](./scripts/processing).

```bash
# CLI for downloading / preparing the dataset
nlp_data help

# CLI for accessing some common data processing scripts
nlp_process help

# Also, you can use `python -m` to access the toolkits
python3 -m gluonnlp.cli.data help
python3 -m gluonnlp.cli.process help

```

# Run Unittests
You may go to [tests](tests) to see how to run the unittests.


# Use Docker
You can use Docker to launch a JupyterLab development environment with GluonNLP installed.

```
# GPU Instance
docker pull gluonai/gluon-nlp:gpu-latest
docker run --gpus all --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 --shm-size=2g gluonai/gluon-nlp:gpu-latest

# CPU Instance
docker pull gluonai/gluon-nlp:cpu-latest
docker run --rm -it -p 8888:8888 -p 8787:8787 -p 8786:8786 --shm-size=2g gluonai/gluon-nlp:cpu-latest
``` 

For more details, you can refer to the guidance in [tools/docker](tools/docker).
