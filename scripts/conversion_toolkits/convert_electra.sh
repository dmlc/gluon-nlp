python3 -m pip install tensorflow==1.15 --upgrade --user
export TF_FORCE_GPU_ALLOW_GROWTH="true"
git clone https://github.com/ZheyuYe/electra.git
cd electra
git checkout 923179410471f9e1820b3f0771c239e1752e4e18
cd ..
for model in small base large
do
    wget https://storage.googleapis.com/electra-data/electra_${model}.zip
    unzip electra_${model}.zip
    python3 convert_electra.py --tf_model_path electra_${model} --electra_path electra --model_size ${model} --test
done
