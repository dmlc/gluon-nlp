#!/bin/bash
lang=$1
branch=$2
capture_flag=$3
set -ex
source ci/prepare_clean_env.sh ${lang}
ci/install_dep.sh
py.test -v ${capture_flag} -n 4 -m "not serial" --durations=50 --cov=gluonnlp tests/unittest
py.test -v ${capture_flag} -n 0 -m "serial" --durations=50 --cov=gluonnlp tests/unittest
ci/codecov.sh -c -F ${branch}-${lang} -n unittests
py.test -v ${capture_flag} -n 4 -m "not serial" --durations=50 --cov=gluonnlp --cov=scripts scripts
py.test -v ${capture_flag} -n 0 -m "serial" --durations=50 --cov=gluonnlp --cov=scripts scripts
ci/codecov.sh -c -F ${branch}-${lang} -n integration
set +ex
