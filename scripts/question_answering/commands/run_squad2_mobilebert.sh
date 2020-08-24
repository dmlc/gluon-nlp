VERSION=2.0  # Either 2.0 or 1.1
MODEL_NAME=google_uncased_mobilebert

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
    --batch_size 8 \
    --num_accumulated 1 \
    --gpus 0,1,2,3 \
    --epochs  5 \
    --lr 4e-5 \
    --warmup_steps 1400 \
    --wd 0.0 \
    --max_seq_length 384 \
    --max_grad_norm 0.1 \
    --overwrite_cache \
