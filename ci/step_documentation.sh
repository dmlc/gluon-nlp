#!/bin/bash
set -ex
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
source ci/prepare_clean_env.sh doc
ci/install_dep.sh
export CUDA_VISIBLE_DEVICES=$EXECUTOR_NUMBER
make -C docs clean
make docs

enforce_linkcheck=$1
if [[ $enforce_linkcheck == true ]]; then
    make -C docs linkcheck SPHINXOPTS=-W
else
    set +ex
    make -C docs linkcheck
fi;
set +ex
