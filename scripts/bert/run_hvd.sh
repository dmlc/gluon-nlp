python --version
python -c "import mxnet; print('successfully imported mxnet', mxnet); import gluonnlp; print('successfully imported gluonnlp', gluonnlp)"

DATA='/home/ec2-user/generated-enwiki-feb-uncased-py3-512/train/part-0/part-000.npz'

horovodrun -np 8 -H localhost:8 python run_pretraining_hvd.py --batch_size 8 --accumulate 4 --max_len 512 --large --lr 1e-4 --data $DATA --warmup_ratio 0.01 --num_steps 1000000 --log_interval=50 --ckpt_dir './ckpt' --ckpt_interval 25000 --num_buckets 10 --dtype float16 #--verbose --profile gpu_0_hvd_large.json
#mpirun -np 8 -H localhost:8 --bind-to none --map-by slot -mca pml ob1 -mca btl ^openib -x NCCL_MIN_NRINGS=2 -x MXNET_USE_OPERATOR_TUNING=0 -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 python run_pretraining_hvd.py --batch_size 32 --lr 1e-4 --data $DATA --warmup_ratio 0.01 --num_steps 301 --log_interval=50 --ckpt_dir './ckpt' --ckpt_interval 25000 --accumulate 1 --num_buckets 10 --dtype float16 2>&1 | tee -a ~/hvd.log #--profile gpu_0_hvd.json
