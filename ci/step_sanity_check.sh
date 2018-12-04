#!/bin/bash
source ci/prepare_clean_env.sh pylint
make lint
