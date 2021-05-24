python3 -m pip install fairseq==0.10.1 --upgrade --user
for model in base large
do
    mkdir xlmr_${model}
    wget "https://dl.fbaipublicfiles.com/fairseq/models/xlmr.${model}.tar.gz"
    tar zxf xlmr.${model}.tar.gz --directory xlmr_${model}
    python3 convert_fairseq_xlmr.py --fairseq_model_path xlmr_${model}/xlmr.${model} --model_size ${model} --test
done
