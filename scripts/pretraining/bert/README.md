# BERT


1. Prepare the dataset, use
```bash
nlp_data prepare_bookcorpus --segment_sentences --segment_num_worker 16
nlp_data prepare_wikipedia --mode download_prepared --segment_sentences --segment_num_worker 16
```
to prepare data, after that, we recommend to convert raw text file to npz data by using
```bash
python create_pretraining_data.py \
      --input_dir BookCorpus/one_sentence_per_line/,wikicorpus/one_sentence_per_line/  \
      --output_dir ./npz_data128 \
      --shard_num  64 \
      --current_shard 7 \
      --max_seq_length 128 \

python create_pretraining_data.py \
      --input_dir BookCorpus/one_sentence_per_line/,wikicorpus/one_sentence_per_line/  \
      --output_dir ./npz_data512 \
      --shard_num  64 \
      --current_shard 7 \
      --max_seq_length 512 \
```
Since it needs too much memory to convert whole dataset, you can use shard number to divide it to different shards.

2. Phase 1 training with sequence length 128 

```bash
python3 run_pretraining.py \
  --model_name google_en_cased_bert_base \
  --optimizer adamw \
  --lr 1e-4 \
  --wd 0.01 \
  --data npz_data128 \
  --batch_size 32 \
  --num_dataset_workers 2 \
  --num_batch_workers 2 \
  --num_steps 900000 \
  --max_seq_length 128 \
  --gpus 0,1,2,3,4,5,6,7
```

also use AMP to accelerate:
```bash
python3 run_pretraining.py \
  --model_name google_en_cased_bert_base \
  --optimizer adamw \
  --lr 1e-4 \
  --wd 0.01 \
  --data npz_data128 \
  --batch_size 32 \
  --num_dataset_workers 2 \
  --num_batch_workers 2 \
  --num_steps 900000 \
  --use_amp \
  --max_seq_length 128 \
  --gpus 0,1,2,3,4,5,6,7
```

3. Phase 2 training with sequence length 512

```bash
python3 run_pretraining.py \
  --model_name google_en_cased_bert_base \
  --optimizer adamw \
  --lr 1e-4 \
  --wd 0.01 \
  --data npz_data512 \
  --batch_size 8 \
  --num_accumulated 4 \
  --num_dataset_workers 2 \
  --num_batch_workers 2 \
  --num_steps 100000 \
  --phase2 \
  --start_step 900000 \
  --max_seq_length 512 \
  --phase1_num_steps 900000 \
  --gpus 0,1,2,3,4,5,6,7
```

```bash
python3 run_pretraining.py \
  --model_name google_en_cased_bert_base \
  --optimizer adamw \
  --lr 1e-4 \
  --wd 0.01 \
  --data npz_data512 \
  --batch_size 8 \
  --num_accumulated 4 \
  --num_dataset_workers 2 \
  --num_batch_workers 2 \
  --num_steps 100000 \
  --phase2 \
  --use_amp \
  --start_step 900000 \
  --max_seq_length 512 \
  --phase1_num_steps 900000 \
  --gpus 0,1,2,3,4,5,6,7
```

Finally we obtain a folder of structure as followed,

```
gluon_en_cased_bert_base
├── vocab-{short_hash}.json    
├── model-{short_hash}.params
├── model-{short_hash}.yml    
```

The pretrained model has been finetuned on SQUAD 2.0, obtaining F1/EM as 77.09/74.38. See also [question_answering](../../question_answering) for more details on the finetuning results.
And when using AMP, we can refer this sheet https://docs.google.com/spreadsheets/d/1c5QPLBpJEAVDUk_BmcpKphp8nN3Dk9q2bKYqa3AIUz0/edit?usp=sharing to see more training details in phase1 and we get F1/EM 76.87/74.20.
