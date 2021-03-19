NOTE: GluonNLP uses `/dev/shm/gluonnlp` shared memory filesystem to share
datasets among multi-process workloads. At this time, `/dev/shm/gluonnlp` is not
cleaned up automatically after the workload completes and manual deletion is
needed to free up memory. Sometimes you may not want to delete
`/dev/shm/gluonnlp` after running a workload, as you intend to run a workload
based on same dataset later and it's useful to keep the dataset in shared
memory.

# BERT

-1. p4 instance preparation

```bash
sudo mkfs.btrfs /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1 /dev/nvme5n1 /dev/nvme6n1 /dev/nvme7n1 /dev/nvme8n1
sudo mount /dev/nvme1n1 /mnt
sudo chown ubuntu:ubuntu /mnt/
```

1. Get the dataset

```bash
nlp_data prepare_bookcorpus --segment_sentences --segment_num_worker 16
nlp_data prepare_wikipedia --mode download_prepared --segment_sentences --segment_num_worker 16
find wikicorpus/one_sentence_per_line BookCorpus/one_sentence_per_line -type f > input_reference
```

2. Prepare batches

```bash
python3 prepare_quickthought.py \
    --input-reference input_reference
    --output /mnt/out_quickthought_128 \
    --model-name google_en_cased_bert_base \
    --max-seq-length 128
```


1. Phase 1 training with sequence length 128

```bash
python3 -m torch.distributed.launch --nproc_per_node=8 run_pretraining.py \
  --model_name google_en_cased_bert_base \
  --lr 0.005 \
  --batch_size 128 \
  --num_accumulated 96 \
  --num_dataloader_workers 4 \
  --num_steps 3870 \
  --input-files /mnt/out_quickthought_128/*feather \
  --mmap-folder /mnt/gluonnlp_mmap \
  --ckpt_dir /mnt/ckpt_dir \
  --ckpt_interval 1000 2>&1| tee train.log;
```

3. Phase 2 training with sequence length 512

TBD

Finally we obtain a folder of structure as followed,

```
coder_base
├── vocab-{short_hash}.json
├── model-{short_hash}.params
├── model-{short_hash}.yml
```
