python3 -m pip install tensorflow==1.15 --upgrade --user
export TF_FORCE_GPU_ALLOW_GROWTH="true"
svn checkout https://github.com/google-research/google-research/trunk/mobilebert

mkdir mobilebert_model
url='https://storage.googleapis.com/cloud-tpu-checkpoints/mobilebert/uncased_L-24_H-128_B-512_A-4_F-4_OPT.tar.gz'
wget ${url} -O "mobilebert.tar.gz"
tar -xvf mobilebert.tar.gz --directory mobilebert_model
python3 convert_mobilebert.py --tf_model_path mobilebert_model/mobilebert --mobilebert_dir mobilebert --test
