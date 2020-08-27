set -ex

python3 -m pip install --pre "mxnet-cu102>=2.0.0b20200802" -f https://dist.mxnet.io/python
cd ../../scripts/datasets/machine_translation
for MODEL in spm subword_nmt yttm hf_bytebpe hf_wordpiece hf_bpe
  bash wmt2014_ende.sh ${MODEL}
for MODEL in spm subword_nmt yttm hf_bytebpe hf_wordpiece hf_bpe
  bash wmt2017_zhen.sh ${MODEL}
