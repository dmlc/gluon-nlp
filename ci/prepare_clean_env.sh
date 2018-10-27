#!/bin/bash
env_name=$1
make clean
conda env update --prune -p conda/${env_name} -f env/${env_name}.yml
conda activate ./conda/${env_name}
conda list
printenv
