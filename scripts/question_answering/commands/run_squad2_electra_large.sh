VERSION=2.0  # Either 2.0 or 1.1
MODEL_NAME=google_electra_large

# Prepare the Data
nlp_data prepare_squad --version ${VERSION}

# Run the script

python3 run_squad.py \
    --model_name ${MODEL_NAME} \
    --data_dir squad \
    --output_dir fintune_${MODEL_NAME}_squad_${VERSION} \
    --version ${VERSION} \
    --do_eval \
    --do_train \
    --batch_size 2 \
    --num_accumulated 4 \
    --gpus 0,1,2,3 \
    --epochs 2 \
    --lr 5e-5 \
    --layerwise_decay 0.9 \
    --warmup_ratio 0.1 \
    --wd 0 \
    --max_seq_length 512 \
    --max_grad_norm 0.1 \
