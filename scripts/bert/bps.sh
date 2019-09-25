export NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DMLC_WORKER_ID=0
export DMLC_NUM_WORKER=1
export DMLC_ROLE=worker

export EVAL_TYPE=benchmark
python /workspace/byteps/launcher/launch.py \
       python run_pretraining.py \
	    --data='/data/book-corpus/book-corpus-large-split/*.train,/data/enwiki/enwiki-feb-doc-split/*.train' \
	    --data_eval='/data/book-corpus/book-corpus-large-split/*.test,/data/enwiki/enwiki-feb-doc-split/*.test' \
	    --num_steps 900000 \
	    --dtype float32 \
	    --lr 1e-4 \
	    --total_batch_size 256 \
	    --accumulate 4 \
	    --model bert_24_1024_16 \
	    --max_seq_length 128 \
	    --max_predictions_per_seq 20 \
	    --num_data_workers 4 \
	    --no_compute_acc --raw \
	    --synthetic_data --eval_use_npz \
	    --comm_backend byteps --log_interval 10

       #/workspace/byteps/example/mxnet/start_mxnet_byteps.sh \
       #/workspace/byteps/example/mxnet/start_mxnet_byteps.sh \

