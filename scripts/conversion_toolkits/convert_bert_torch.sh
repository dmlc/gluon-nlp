set -ex

python3 -m pip install 'tensorflow<3' --upgrade --user
python3 -m pip install tensorflow_hub --upgrade --user
export TF_FORCE_GPU_ALLOW_GROWTH="true"

# Conversion for English Models
for model in base large
do
    for case in cased uncased
        do
            hub_directory="google_en_${case}_bert_${model}"
            mkdir -p ${hub_directory}
            if [ ${model} == base ];then
                url="https://tfhub.dev/google/bert_${case}_L-12_H-768_A-12/1?tf-hub-format=compressed"
            else
                url="https://tfhub.dev/google/bert_${case}_L-24_H-1024_A-16/1?tf-hub-format=compressed"
            fi
            wget ${url} -O "${hub_directory}.tar.gz"
            tar -xvf ${hub_directory}.tar.gz --directory ${hub_directory}
            cp bert_${model}_config.json ${hub_directory}/assets/
            python3 convert_tf_hub_model.py --tf_hub_model_path ${hub_directory} --model_type bert --test --torch
        done
    done
