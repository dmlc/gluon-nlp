python --version
python -c "import mxnet; print('successfully imported mxnet', mxnet); import gluonnlp; print('successfully imported gluonnlp', gluonnlp)"
horovodrun -np 8 -H localhost:8 python run_pretraining_hvd.py --batch_size 4096 --lr 1e-4 --data './out/*.npz' --warmup_ratio 0.01 --num_steps 1000000 --log_interval=50 --ckpt_dir './ckpt' --ckpt_interval 25000 --accumulate 1 --num_buckets 10 --dtype float16 --use_avg_len
