python3 -m pip install git+https://github.com/huggingface/transformers.git --upgrade
for model in small base large 3B 11B
do
    dest_dir="google_t5_${model}"
    mkdir ${dest_dir}
    python3 convert_t5.py "t5-${model,,}" ${dest_dir} --test
done