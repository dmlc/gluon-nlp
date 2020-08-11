python3 -m pip install git+https://github.com/pytorch/fairseq.git@master --upgrade --user
for model in base large
do
    mkdir bart_${model}
    wget  "https://dl.fbaipublicfiles.com/fairseq/models/bart.${model}.tar.gz"
    tar zxf bart.${model}.tar.gz --directory bart_${model}
    python3 convert_fairseq_bart.py --fairseq_model_path bart_${model}/bart.${model} --test
done
