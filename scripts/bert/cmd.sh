#hudl -v -h ~/hosts 'pkill python'
pkill python

#mpirun -np 32 --hostfile ~/hosts -mca pml ob1 -mca btl ^openib \
mpirun -np 8 --hostfile ~/hosts -mca pml ob1 -mca btl ^openib \
       -mca btl_tcp_if_exclude docker0,lo --map-by ppr:4:socket \
       --mca plm_rsh_agent 'ssh -q -o StrictHostKeyChecking=no' \
       -x GLUON_REAL_DATA=0 -x MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=15 -x HOROVOD_NORMAL=1 -x HVD_SORT=0 -x HVD_REVERSE=1 -x GLUON_ORDERED=1 \
       -x NCCL_MIN_NRINGS=16 -x NCCL_DEBUG=VERSION -x HOROVOD_HIERARCHICAL_ALLREDUCE=0 -x HOROVOD_CYCLE_TIME=50 \
       -x HOROVOD_TIMELINE_MARK_CYCLES=0 \
       -x MXNET_SAFE_ACCUMULATION=1 --tag-output python run_pretraining.py \
       --data_eval='~/mxnet-data/bert-pretraining/datasets/*/*/*.dev.5K,' \
       --num_steps 25 --log_interval 10 --ckpt_interval 20 \
       --lr 2e-4 --batch_size 32 --accumulate 1 --raw --short_seq_prob 0 --accumulate 1 --model bert_12_768_12 --batch_size_eval 12 \
       --warmup 0.04 \
       --comm_backend horovod \
       --data='~/mxnet-data/bert-pretraining/datasets/*/*/wiki.dev.5K,~/mxnet-data/bert-pretraining/datasets/*/*/wiki.dev.5K,~/mxnet-data/bert-pretraining/datasets/*/*/wiki.dev.5K,~/mxnet-data/bert-pretraining/datasets/*/*/wiki.dev.5K,~/mxnet-data/bert-pretraining/datasets/*/*/wiki.dev.5K,~/mxnet-data/bert-pretraining/datasets/*/*/wiki.dev.5K,~/mxnet-data/bert-pretraining/datasets/*/*/wiki.dev.5K,~/mxnet-data/bert-pretraining/datasets/*/*/wiki.dev.5K,' 2>&1 | tee train.log #--profile p.json

       # --data='~/mxnet-data/bert-pretraining/datasets/*/*/*.train,' \
       # --data_eval='~/mxnet-data/bert-pretraining/datasets/*/*/*.dev,' \
       #-x HOROVOD_TIMELINE=time \
       #-x HOROVOD_FUSION_THRESHOLD=67108864 \
# 67108864
# GLUON_REAL_DATA=1 MXNET_SAFE_ACCUMULATION=1 python run_pretraining_hvd.py --data='~/mxnet-data/bert-pretraining/datasets/*/*/*.train,'   --data_eval='~/mxnet-data/bert-pretraining/datasets/*/*/*.dev,' --num_steps 1000000              --lr 1e-4 --batch_size 4096 --accumulate 1 --raw --short_seq_prob 0 --log_interval 10 --accumulate 1 --model bert_24_1024_16 --batch_size_eval 12
