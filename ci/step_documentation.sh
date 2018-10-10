#!/bin/bash
set -ex
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
source ci/prepare_clean_env.sh doc
ci/install_dep.sh
make clean_doc
make docs
set +ex
