# Datasets
## OpenWebTextCorpus
Following the instruction of [Prepare OpenWebTextCorpus](../datasets/pretrain_corpus#openwebtext), download and prepare the dataset, obtaining a total of 20610 text files in the folder `prepared_owt`.

```bash
python preprocesse_owt.py --input prepared_owt --output preprocessed_owt --shuffle
```
The above command allows us to generate the preprocessed Numpy features saved in `.npz`.
# Pretrain Model
## ELECTRA

```bash
horovodrun -np 8 -H localhost:8 python -m run_electra \
    --model_name google_electra_small \
    --data `preprocessed_owt/*.npz` \
    --gpus 0,1,2,3,4,5,6,7 \
    --do_train \
    --do_eval \
    --output_dir ${OUTPUT} \
    --num_accumulated ${ACCMULATE} \
    --batch_size ${BS} \
    --lr ${LR} \
    --wd ${WD} \
    --max_seq_len ${MSL} \
    --max_grad_norm 1 \
    --warmup_steps 10000 \
    --num_train_steps 1000000 \
    --log_interval 50 \
    --save_interval 10000 \
    --mask_prob 0.15 \
    --comm_backend horovod \
```

Or we could preprocessing the features on the fly based on the `.txt` files like
```bash
horovodrun -np 8 -H localhost:8 python -m run_electra \
    --model_name google_electra_small \
    --data `prepared_owt/*.txt` \
    --from_raw \
    --gpus 0,1,2,3,4,5,6,7 \
    --do_train \
    --do_eval \
    --output_dir ${OUTPUT} \
    --num_accumulated ${ACCMULATE} \
    --batch_size ${BS} \
    --lr ${LR} \
    --wd ${WD} \
    --max_seq_len ${MSL} \
    --max_grad_norm 1 \
    --warmup_steps 10000 \
    --num_train_steps 1000000 \
    --log_interval 50 \
    --save_interval 10000 \
    --mask_prob 0.15 \
    --comm_backend horovod \
```
