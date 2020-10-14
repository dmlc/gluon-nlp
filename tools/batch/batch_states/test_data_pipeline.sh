#!/bin/bash
# Shell script for testing the data preprocessing on AWS Batch

set -ex
echo $PWD

for MODEL in spm yttm
do
  bash ../../../scripts/datasets/machine_translation/wmt2014_ende.sh ${MODEL}
done
for MODEL in spm yttm
do
  bash ../../../scripts/datasets/machine_translation/wmt2017_zhen.sh ${MODEL}
done
