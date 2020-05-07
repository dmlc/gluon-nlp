export TF_FORCE_GPU_ALLOW_GROWTH="true"
git clone https://github.com/ZheyuYe/electra.git
for model in small base large
do
    wget https://storage.googleapis.com/electra-data/electra_${model}.zip
    unzip electra_${model}.zip
    python convert_electra.py --tf_model_path electra_${model} --electra_path electra --model_size ${model} --test
done
