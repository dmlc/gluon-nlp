#!/bin/bash
set -ex
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
source ci/prepare_clean_env.sh doc
ci/install_dep.sh
export CUDA_VISIBLE_DEVICES=$EXECUTOR_NUMBER
make clean_doc
dev=$1
if [[ $dev == true ]]; then
    make dev_docs
else
    make docs
    enforce_linkcheck=$2
    if [[ $enforce_linkcheck == true ]]; then
        make -C docs linkcheck SPHINXOPTS=-W
    else
        make -C docs linkcheck
    fi;
fi;
set +ex
