python --version
python -c "import mxnet; print('successfully imported mxnet', mxnet); import gluonnlp; print('successfully imported gluonnlp', gluonnlp)"

DATA='/home/ec2-user/generated-book-feb-uncased-py3-128/train/part-0/part-000.npz'
#DATA='/home/ec2-user/generated-enwiki-feb-uncased-py3-512/train/part-0/part-000.npz'

horovodrun -np 8 -H localhost:8 python run_pretraining_hvd.py --batch_size 32 --accumulate 1 --dummy_data_len 128 --lr 1e-4 --data $DATA --warmup_ratio 0.01 --num_steps 1000000 --log_interval=50 --ckpt_dir './ckpt' --ckpt_interval 25000 --num_buckets 10 --dtype float16 #--verbose --profile gpu_0_hvd.json
