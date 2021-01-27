python3 -m pip install git+https://github.com/huggingface/transformers.git --upgrade
for model in small base large xl xxl
do
    dest_dir="google_mt5_${model}"
    mkdir ${dest_dir}
    python3 convert_mt5.py "google/mt5-${model}" ${dest_dir} --test
done
