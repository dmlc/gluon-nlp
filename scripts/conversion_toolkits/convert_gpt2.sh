python3 -m pip install tensorflow==1.15 --upgrade --user
git clone https://github.com/openai/gpt-2.git gpt_2
for model in 124M 355M 774M 1558M
do
    python3 gpt_2/download_model.py ${model}
    mkdir gpt2_${model}
    CUDA_VISIBLE_DEVICES="" python3 convert_gpt2.py --tf_model_path models/${model} --save_dir gpt2_${model} --test
done
