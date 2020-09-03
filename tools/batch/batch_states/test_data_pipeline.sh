set -ex
echo $PWD

python3 -m pip install --pre "mxnet>=2.0.0b20200802" -f https://dist.mxnet.io/python

for MODEL in spm yttm
do
  bash ../../../scripts/datasets/machine_translation/wmt2014_ende.sh ${MODEL}
done
for MODEL in spm yttm
do
  bash ../../../scripts/datasets/machine_translation/wmt2017_zhen.sh ${MODEL}
done
