# Pre-training Scripts

This folder contains two preliminary scripts for multi-GPU pre-training for BERT. `run_pretraining.py` uses the parameter server approach, while `run_pretraining_hvd.py` uses horovod all-reduce for multi-GPU/multi-machine training. Horovod is recommended on p3.16xlarge instances.

## Horovd

### Pre-requisite

To use GluonNLP with Horovd, the following two ways are supported:
1. Install pre-built MXNet binary on a machine with gcc-4 (e.g. Amazon Linux 1). Then install horovod.
```
# install mxnet
pip install mxnet-cu90==1.5.0b20190407 --user
# install horovod
git clone --recursive https://github.com/uber/horovod horovod;
cd horovd; pip install . --user;
# install gluonnlp
cd gluonnlp; python setup.py install --user
```
2. Install MXNet and horovod from source. This does not require gcc-4 on the machine.
```
# install mxnet
git clone --recursive https://github.com/apache/incubator-mxnet mxnet;
cd mxnet;
cp make/config.mk .;
echo "ADD_CFLAGS += -I/usr/include/openblas" >>config.mk;
echo "USE_CUDA=1" >>config.mk;
echo "USE_CUDA_PATH=/usr/local/cuda" >>config.mk;
echo "USE_CUDNN=1" >>config.mk;
echo "USE_DIST_KVSTORE=1" >>config.mk;
make -j64;
# install horovod
git clone --recursive https://github.com/uber/horovod horovod;
cd horovd; pip install . --user;
# install gluonnlp
cd gluonnlp; python setup.py install --user
```

### Usage

#### Test on sample dataset
```
# training sample creation
python create_pretraining_data.py --input_file sample_text.txt --output_dir out --vocab book_corpus_wiki_en_uncased --max_seq_length 128 --max_predictions_per_seq 20 --dupe_factor 5 --masked_lm_prob 0.15 --short_seq_prob 0.1
# training with 1 GPU
horovodrun -np 1 -H localhost:1 python run_pretraining_hvd.py --batch_size 32 --accumulate 1 --lr 2e-5 --data 'out/*.npz' --warmup_ratio 0.5 --num_steps 20 --pretrained --log_interval=2 --data_eval 'out/*.npz' --batch_size_eval 8 --ckpt_dir ckpt --verbose
# training with 8 GPUs with fp16 (requires at least 8 input npz files)
horovodrun -np 8 -H localhost:8 python run_pretraining_hvd.py --batch_size 32 --accumulate 1 --lr 2e-5 --dtype float16 --data 'out/*.npz' --warmup_ratio 0.5 --num_steps 20 --pretrained --log_interval=2 --ckpt_dir './ckpt'
```

#### Full pre-training

```
horovodrun run_pretraining_hvd.py --data='/feb/generated-*/train/*/*.npz' --num_steps 1000000 --log_interval 250 --lr 1e-4 --batch_size 4096 --warmup_ratio 0.01 --ckpt_dir ./ckpt --ckpt_interval 25000 --accumulate 4 --num_buckets 10 --dtype float16 --seed 0 --use_avg_len
```

#### Profiling training speed
```
export DATA1='/home/ec2-user/generated-book-feb-uncased-py3-128/train/part-0/part-000.npz'
export DATA2='/home/ec2-user/generated-enwiki-feb-uncased-py3-512/train/part-0/part-000.npz'
# training with 8 GPUs
horovodrun -np 8 -H localhost:8 python run_pretraining_hvd.py --batch_size 32 --accumulate 1 --dummy_data_len 128 --lr 1e-4 --data $DATA1 --warmup_ratio 0.01 --num_steps 1000000 --log_interval=50 --ckpt_dir './ckpt' --ckpt_interval 25000 --num_buckets 10 --dtype float16 --profile profile_result.json --verbose
# training with 32 GPUs
mpirun -np 32 -H localhost:8,ip-172-31-11-207:8,ip-172-31-11-168:8,ip-172-31-9-146:8 --bind-to none --map-by slot -mca pml ob1 -mca btl ^openib -mca btl_tcp_if_exclude docker0,lo -x NCCL_MIN_NRINGS=4 -x MXNET_USE_OPERATOR_TUNING=0 -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 python run_pretraining_hvd.py --batch_size 32 --accumulate 1 --dummy_data_len 512 --lr 1e-4 --data $DATA2 --warmup_ratio 0.01 --num_steps 30000 --log_interval=50 --ckpt_dir './ckpt' --ckpt_interval 25000 --num_buckets 10 --dtype float16
# related env vars: HOROVOD_FUSION_THRESHOLD, HOROVOD_TIMELINE_MARK_CYCLES, HOROVOD_TIMELINE, HOROVOD_CYCLE_TIME
```

## Parameter Server

### Pre-requiresite
```
# install mxnet
pip install mxnet-cu90==1.5.0b20190407 --user
# install gluonnlp
cd gluonnlp; python setup.py install --user
```

### Usage

#### Test on sample dataset
```
# training sample creation
python create_pretraining_data.py --input_file sample_text.txt --output_dir out --vocab book_corpus_wiki_en_uncased --max_seq_length 128 --max_predictions_per_seq 20 --dupe_factor 5 --masked_lm_prob 0.15 --short_seq_prob 0.1
# training with 1 GPU
python run_pretraining.py --gpus 0 --batch_size 32 --lr 2e-5 --data 'out/*.npz' --warmup_ratio 0.5 --num_steps 20 --pretrained --log_interval=2 --data_eval 'out/*.npz' --batch_size_eval 8 --ckpt_dir ckpt
```

#### Full pre-training

```
run_pretraining.py --data='/feb/generated-*/train/*/*.npz' --num_steps 1000000 --log_interval 250 --lr 1e-4 --batch_size 4096 --warmup_ratio 0.01 --gpus 0,1,2,3,4,5,6,7 --ckpt_dir ./ckpt --ckpt_interval 25000 --accumulate 4 --num_buckets 10 --kvstore device_sync --dtype float16 --seed 0 --use_avg_len
```

#### Profiling training speed
```
export DATA1='/home/ec2-user/generated-book-feb-uncased-py3-128/train/part-0/part-000.npz'
export DATA2='/home/ec2-user/generated-enwiki-feb-uncased-py3-512/train/part-0/part-000.npz'
python run_pretraining.py --gpus 0,1,2,3,4,5,6,7 --batch_size 64 --accumulate 1 --dummy_data_len 128 --lr 2e-5 --data $DATA1 --warmup_ratio 0.5 --num_steps 1000000 --log_interval=50 --data_eval 'out/*.npz' --batch_size_eval 8 --ckpt_dir ckpt --num_buckets 10 --dtype float16
```
