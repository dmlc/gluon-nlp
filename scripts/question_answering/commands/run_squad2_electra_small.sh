USE_HOROVOD=${1:-0}  # Horovod flag. Do not use horovod by default
VERSION=${2:-2.0}   # Version
MODEL_NAME=google_electra_small
BATCH_SIZE=8
NUM_ACCUMULATED=1
EPOCHS=2
LR=0.0003
WARMUP_RATIO=0.1
WD=0
MAX_SEQ_LENGTH=512
MAX_GRAD_NORM=0.1
LAYERWISE_DECAY=0.8

# Prepare the Data
nlp_data prepare_squad --version ${VERSION}

# Run the script
if [ ${USE_HOROVOD} -eq 0 ];
then
  RUN_COMMAND="python3 run_squad.py --gpus 0,1,2,3"
else
  RUN_COMMAND="horovodrun -np 4 -H localhost:4 python3 run_squad.py --comm_backend horovod"
fi
python3 run_squad.py \
    --model_name ${MODEL_NAME} \
    --data_dir squad \
    --output_dir fintune_${MODEL_NAME}_squad_${VERSION} \
    --version ${VERSION} \
    --do_eval \
    --do_train \
    --batch_size ${BATCH_SIZE} \
    --num_accumulated ${NUM_ACCUMULATED} \
    --layerwise_decay ${LAYERWISE_DECAY} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --warmup_ratio ${WARMUP_RATIO} \
    --wd ${WD} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --overwrite_cache
