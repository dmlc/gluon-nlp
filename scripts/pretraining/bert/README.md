# BERT

1. Prepare the dataset, bookcorpus_wiki, in the folder `data`.

2. Phase 1 training with sequence length 128

```bash
python run_pretraining.py \
  --model_name google_en_cased_bert_base \
  --optimizer adamw \
  --lr 1e-4 \
  --wd 0.01 \
  --data data/*.train \
  --raw \
  --batch_size 32 \
  --num_dataset_workers 2 \
  --num_batch_workers 2 \
  --num_steps 900000 \
  --max_seq_length 128 \
  --gpus 0,1,2,3,4,5,6,7
```

3. Phase 2 training with sequence length 512

```bash
python run_pretraining.py \
  --model_name google_en_cased_bert_base \
  --optimizer adamw \
  --lr 1e-4 \
  --wd 0.01 \
  --data data/*.train \
  --raw \
  --batch_size 8 \
  --num_accumulated 4 \
  --num_dataset_workers 2 \
  --num_batch_workers 2 \
  --num_steps 100000 \
  --phase2 \
  --start_step 900000 \
  --max_seq_length 512 \
  --gpus 0,1,2,3,4,5,6,7
```

Finally we obtain a folder of structure as followed,

```
gluon_en_cased_bert_base
├── vocab-{short_hash}.json    
├── model-{short_hash}.params
├── model-{short_hash}.yml    
```

The pretrained model has been finetuned on SQUAD 2.0, obtaining 77.89/74.72.





