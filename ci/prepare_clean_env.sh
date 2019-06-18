#!/bin/bash
env_name=$1

export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export CUDA_VISIBLE_DEVICES=$EXECUTOR_NUMBER
export CONDA_ENVS_PATH=$PWD/conda
export CONDA_PKGS_DIRS=$PWD/conda/pkgs
export MXNET_HOME=$PWD/tests/data
export HOROVOD_WITHOUT_TENSORFLOW=1
export HOROVOD_WITHOUT_PYTORCH=1
export HOROVOD_WITH_MXNET=1

make clean
conda env update --prune -p conda/${env_name} -f env/${env_name}.yml
conda activate ./conda/${env_name}
conda list
printenv

pip install -v -e .
python -m spacy download en
python -m spacy download de
python -m nltk.downloader all
python -m pip install horovod==0.16.2 --no-cache-dir -U
