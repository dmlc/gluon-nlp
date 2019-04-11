python --version
python -c "import mxnet; print('successfully imported mxnet', mxnet); import gluonnlp; print('successfully imported gluonnlp', gluonnlp)"

DATA='/home/ec2-user/generated-book-feb-uncased-py3-128/train/part-0/part-000.npz'
#DATA='/home/ec2-user/generated-enwiki-feb-uncased-py3-512/train/part-0/part-000.npz'

PYTHONPATH=~/gluon-nlp/src python run_pretraining.py --gpus 0 --batch_size 64 --accumulate 1 --dummy_data_len 128 --lr 2e-5 --data $DATA --warmup_ratio 0.5 --num_steps 1000000 --log_interval=50 --data_eval 'out/*.npz' --batch_size_eval 8 --ckpt_dir ckpt --num_buckets 10 --dtype float16
