set -ex

python3 -m pip install tensorflow==1.15 --upgrade --user
python3 -m pip install tensorflow_hub --upgrade --user
export TF_FORCE_GPU_ALLOW_GROWTH="true"
for model in base large xlarge xxlarge
do
    hub_directory="google_albert_${model}_v2"
    mkdir -p ${hub_directory}
    wget "https://tfhub.dev/google/albert_${model}/3?tf-hub-format=compressed" -O "${hub_directory}.tar.gz"
    tar -xvf ${hub_directory}.tar.gz --directory ${hub_directory}
    python3 convert_tf_hub_model.py --tf_hub_model_path ${hub_directory} --model_type albert --test
done
