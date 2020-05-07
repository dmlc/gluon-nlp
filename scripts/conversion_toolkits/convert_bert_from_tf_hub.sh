export TF_FORCE_GPU_ALLOW_GROWTH="true"

# Conversion for English Models
for model in base large
do
    for case in cased uncased
    do
        hub_directory="google_en_${case}_bert_${model}"
        mkdir ${hub_directory}
        if [ ${model} == base ];then
            url="https://tfhub.dev/google/bert_${case}_L-12_H-768_A-12/1?tf-hub-format=compressed"
        else
            url="https://tfhub.dev/google/bert_${case}_L-24_H-1024_A-16/1?tf-hub-format=compressed"
        fi
        wget ${url} -O "${hub_directory}.tar.gz"
        tar -xvf ${hub_directory}.tar.gz --directory ${hub_directory}
        cp bert_${model}_config.json ${hub_directory}/assets/
        python convert_tf_hub_model.py --tf_hub_model_path ${hub_directory} --model_type bert --test
    done
done

# Conversion for Chinese Models
url="https://tfhub.dev/google/bert_zh_L-12_H-768_A-12/2?tf-hub-format=compressed"
hub_directory="google_zh_bert_base"
mkdir ${hub_directory}
wget ${url} -O "${hub_directory}.tar.gz"
tar -xvf ${hub_directory}.tar.gz --directory ${hub_directory}
cp bert_base_config.json ${hub_directory}/assets/
python convert_tf_hub_model.py --tf_hub_model_path ${hub_directory} --model_type bert --test

# Conversion for Multi-lingual Models
url="https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/2?tf-hub-format=compressed"
hub_directory="google_multi_cased_bert_base"
mkdir ${hub_directory}
wget ${url} -O "${hub_directory}.tar.gz"
tar -xvf ${hub_directory}.tar.gz --directory ${hub_directory}
cp bert_base_config.json ${hub_directory}/assets/
python convert_tf_hub_model.py --tf_hub_model_path ${hub_directory} --model_type bert --test

# Conversion for Whole-word-masking Models
for case in cased uncased
do
    hub_directory="google_en_${case}_bert_wwm_large"
    mkdir ${hub_directory}
    url="https://tfhub.dev/google/bert_en_wwm_${case}_L-24_H-1024_A-16/2?tf-hub-format=compressed"
    wget ${url} -O ${hub_directory}
    tar -xvf ${hub_directory}.tar.gz --directory ${hub_directory}
    cp bert_${model}_config.json ${hub_directory}/assets/
    python convert_tf_hub_model.py --tf_hub_model_path ${hub_directory} --model_type bert --test
done
