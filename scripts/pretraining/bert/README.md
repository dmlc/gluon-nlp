# BERT

1. Prepare the dataset, use nlp_data prepare_bookcorpus to download raw txt file.

2. Phase 1 training with sequence length 128 

```bash
python3 run_pretraining.py \
  --model_name google_en_cased_bert_base \
  --optimizer adamw \
  --lr 1e-4 \
  --wd 0.01 \
  --data_dir BookCorpus/books1/epubtxt \
  --raw \
  --batch_size 32 \
  --num_dataset_workers 2 \
  --num_batch_workers 2 \
  --num_steps 900000 \
  --max_seq_length 128 \
  --gpus 0,1,2,3,4,5,6,7
```
and can use amp and horovod to accelerate:
```bash 
SUBWORD_ALGO=yttm
SRC=en
TGT=de
nohup horovodrun -np 8 -H localhost:8 python run_pretraining.py \
  --comm_backend horovod \
  --model_name google_en_cased_bert_base \
  --data_dir BookCorpus/books1/epubtxt \
  --optimizer adamw \
  --lr 1e-4 \
  --wd 0.01 \
  --data books1/epubtxt/crouton-a-love-story.epub.txt,books1/epubtxt/gathered-words-from-an-island.epub.txt \
  --raw \
  --use_amp \
  --batch_size 32 \
  --num_dataset_workers 2 \
  --num_batch_workers 2 \
  --num_steps 900000 \
  --max_seq_length 128 \
  --gpus 0,1,2,3,4,5,6,7 &
```

3. Phase 2 training with sequence length 512

```bash
python3 run_pretraining.py \
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

The pretrained model has been finetuned on SQUAD 2.0, obtaining F1/EM as 78.50/75.57. See also [question_answering](../../question_answering) for more details on the finetuning results.
