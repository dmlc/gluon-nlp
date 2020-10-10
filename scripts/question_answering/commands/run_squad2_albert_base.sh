USE_HOROVOD=${1:-0}  # Horovod flag. Do not use horovod by default

VERSION=2.0  # Either 2.0 or 1.1
MODEL_NAME=google_albert_base_v2

# Prepare the Data
nlp_data prepare_squad --version ${VERSION}

# Run the script

if [ ${USE_HOROVOD} -eq 0 ];
then
  RUN_COMMAND="python3 run_squad.py --gpus 0,1,2,3"
else
  RUN_COMMAND="horovodrun -np 4 -H localhost:4 python3 run_squad.py --comm_backend horovod"
fi

${RUN_COMMAND} --model_name ${MODEL_NAME} \
    --data_dir squad \
    --output_dir fintune_${MODEL_NAME}_squad_${VERSION} \
    --version ${VERSION} \
    --do_eval \
    --do_train \
    --batch_size 4 \
    --num_accumulated 3 \
    --epochs 3 \
    --lr 2e-5 \
    --warmup_ratio 0.1 \
    --wd 0.01 \
    --max_seq_length 512 \
    --max_grad_norm 0.1 \
    --overwrite_cache
