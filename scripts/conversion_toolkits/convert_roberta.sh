python3 -m pip install git+https://github.com/pytorch/fairseq.git@master --upgrade --user
for model in base large
do
    mkdir roberta_${model}
    wget "https://dl.fbaipublicfiles.com/fairseq/models/roberta.${model}.tar.gz"
    tar zxf roberta.${model}.tar.gz --directory roberta_${model}
    python3 convert_fairseq_roberta.py --fairseq_model_path roberta_${model}/roberta.${model} --test
done
