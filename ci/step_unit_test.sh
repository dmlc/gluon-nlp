#!/bin/bash
lang=$1
branch=$2
capture_flag=$3
set -ex
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
source ci/prepare_clean_env.sh ${lang}
ci/install_dep.sh
export CUDA_VISIBLE_DEVICES=$EXECUTOR_NUMBER
pytest -v ${capture_flag} -n 4 -m "not serial" --durations=50 tests/unittest --cov gluonnlp
coverage xml
ci/codecov.sh -c -F ${branch},${lang},notserial -n unittests -f coverage.xml
pytest -v ${capture_flag} -n 0 -m "serial" --durations=50 tests/unittest --cov gluonnlp
coverage xml
ci/codecov.sh -c -F ${branch},${lang},serial -n unittests -f coverage.xml
pytest -v ${capture_flag} -n 4 -m "not serial" --durations=50 scripts --cov scripts
coverage xml
ci/codecov.sh -c -F ${branch},${lang},notserial -n integration -f coverage.xml
pytest -v ${capture_flag} -n 0 -m "serial" --durations=50 scripts --cov scripts
coverage xml
ci/codecov.sh -c -F ${branch},${lang},serial -n integration -f coverage.xml
set +ex
