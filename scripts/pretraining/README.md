# Datasets
## OpenWebTextCorpus
Following the instruction of [Prepare OpenWebTextCorpus](../datasets/pretrain_corpus#openwebtext), download and prepare the dataset, obtaining a total of 20610 text files in the folder `prepared_owt`.

```bash
python3 data_preprocessing.py --input prepared_owt --output preprocessed_owt --max_seq_length 128 --shuffle
```
The above command allows us to generate the preprocessed Numpy features saved in `.npz`.
# Pretrain Model
## ELECTRA
Following [Official Quickstart](https://github.com/google-research/electra#quickstart-pre-train-a-small-electra-model), pretrain a small model using OpenWebText as pretraining corpus. Note that [horovod](https://github.com/horovod/horovod) needs to be installed in advance, if `comm_backend` is set to `horovod`.

```bash
horovodrun -np 2 -H localhost:2 python3 -m run_electra \
    --model_name google_electra_small \
    --data 'preprocessed_owt/*.npz' \
    --generator_units_scale 0.25 \
    --gpus 0,1 \
    --do_train \
    --do_eval \
    --output_dir ${OUTPUT} \
    --num_accumulated 1 \
    --batch_size 64 \
    --lr 5e-4 \
    --wd 0.01 \
    --max_seq_len 128 \
    --max_grad_norm 1 \
    --warmup_steps 10000 \
    --num_train_steps 1000000 \
    --log_interval 50 \
    --save_interval 10000 \
    --mask_prob 0.15 \
    --comm_backend horovod \
```

Alternatively, we could preprocessing the features on the fly and train this model with raw text directly like
```bash
horovodrun -np 2 -H localhost:2 python3 -m run_electra \
    --model_name google_electra_small \
    --generator_units_scale 0.25 \
    --data 'prepared_owt/*.txt' \
    --from_raw \
    --gpus 0,1 \
    --do_train \
    --do_eval \
    --output_dir ${OUTPUT} \
    --num_accumulated 1 \
    --batch_size 64 \
    --lr 5e-4 \
    --wd 0.01 \
    --max_seq_len 128 \
    --max_grad_norm 1 \
    --warmup_steps 10000 \
    --num_train_steps 1000000 \
    --log_interval 50 \
    --save_interval 10000 \
    --mask_prob 0.15 \
    --comm_backend horovod \
```

For the convenience of verification, the pretrained small model trained on OpenWebText named `gluon_electra_small_owt` is released and uploaded to S3 with directory structure as

```
gluon_electra_small_owt
├── vocab-{short_hash}.json    
├── model-{short_hash}.params
├── model-{short_hash}.yml    
├── gen_model-{short_hash}.params   
├── disc_model-{short_hash}.params
```

After pretraining, several downstream NLP tasks such as Question Answering are available to fine-tune. Here is an example of fine-tuning a local pretrained model on [SQuAD 1.1/2.0](../question_answering#squad).

```bash
python3 run_squad.py \
    --model_name google_electra_small \
    --data_dir squad \
    --backbone_path ${OUTPUT}/model-{short_hash}.params \
    --output_dir ${FINE-TUNE_OUTPUT} \
    --version ${VERSION} \
    --do_eval \
    --do_train \
    --batch_size 32 \
    --num_accumulated 1 \
    --gpus 0 \
    --epochs 2 \
    --lr 3e-4 \
    --layerwise_decay 0.8 \
    --warmup_ratio 0.1 \
    --max_saved_ckpt 6 \
    --all_evaluate \
    --wd 0 \
    --max_seq_length 128 \
    --max_grad_norm 0.1 \
```

Resulting in the following output

| Model Name    | SQuAD1.1 dev  | SQuAD2.0 dev |
|--------------------------|---------------|--------------|
|gluon_electra_small_owt   | 69.40/76.98   | 67.63/69.89  |
