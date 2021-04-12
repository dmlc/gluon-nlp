# finetune classification
## prepare datasets
use nlp_data to prepare data at first.
```bash
nlp_data prepare_glue --benchmark glue -t sst
```
##finetine scripts
Then run the scripts to finetune:
```bash
python train_classification.py \
  --model_name google_en_uncased_bert_base \
  --task_name cola \
  --lr 2e-5\
  --model_name google_en_cased_bert_base \
  --batch_size 32 \
  --do_train \
  --do_eval \
  --seed 7800 \
  --epochs 10 \
  --optimizer adamw \
  --train_dir glue/cola/train.parquet \
  --eval_dir glue/cola/dev.parquet \
  --gpus 0
```
alternatively, because some task are slow(like MNLI), you can use horovod to accelerate,
```bash
horovodrun -np 4 -H localhost:4  python train_classification.py \
  --comm_backend horovod \
  --model_name google_en_uncased_bert_base \
  --task_name mnli \
  --lr 2e-4\
  --batch_size 32 \
  --do_train \
  --do_eval \
  --epochs 5 \
  --log_interval 500 \
  --warmup_ratio 0.1 \
  --optimizer adamw \
  --train_dir glue/mnli/train.parquet \
  --eval_dir glue/mnli/dev_matched.parquet \
  --gpus 0,1,2,3
```

## some result
here are some results with their hyperparameters

| task Name    | metirc | learning rate  | batch size | seed | epoch | result | tensorboard dev |
|-----------|-------------|---------------|--------------|---------|-------|------|-----|
|    SST    | Accuracy |  2e-5       | 32    | 7800 |  5 |  93.23 | https://tensorboard.dev/experiment/eKVI0DC6SEWBbHzS8ZphNg/|
|    STS    |  Pearson Corr. | 2e-5       | 32    | 24 |  10 |  89.26 |  https://tensorboard.dev/experiment/kPOnlNeiQ4W5EmFlkqjC6A/|
|    CoLA    | Matthew Corr.  | 2e-5       | 32    | 7800 |  10 |  59.23 |  https://tensorboard.dev/experiment/33euRGh9SrW3p15JWgILnw/ |
|    RTE    |  Accuracy | 2e-5       | 32    | 1800 |  10 |  69.67 |  https://tensorboard.dev/experiment/XjTxr5anRrC1LMukLJJQ3g/|
|    MRPC    | Accuracy/F1  | 3e-5       | 32    | 7800 |  5 |  85.38/87.31 |  https://tensorboard.dev/experiment/jEJFq2XXQ8SvCxt6eKIjwg/ |
|    MNLI    |  Accuracy(m/mm) | 2e-5       | 48    | 7800 |  4 |  84.90/85.10 |  https://tensorboard.dev/experiment/CZQlOBedRQeTZwn5o5fbKQ/ |


## different method
We also offer different finetune method to save time and space. So now we offer two different method:
bias-finetune() and adapter-finetune. To use them, you can directly add an augment "method" like:
```bash
python train_classification.py \
  --model_name google_en_uncased_bert_base \
  --method adapter \
  --task_name mrpc \
  --lr 4.5e-4\
  --model_name google_en_cased_bert_base \
  --batch_size 32 \
  --do_train \
  --do_eval \
  --seed 7800 \
  --epochs 10 \
  --optimizer adamw \
  --train_dir glue/mrpc/train.parquet \
  --eval_dir glue/mrpc/dev.parquet \
  --gpus 1
```
And here are some result of different method(the blank means we can't find proper hyperparameter until now)

| task Name    | metirc | full | bias-finetune | adapter |
|-----------|-------------|-------------|-------------|-------------|
|    SST    | Accuracy |  93.23 |  | 93.46 |
|    STS    |  Pearson Corr. | 89.26 | 89.30 | 89.70 |
|    CoLA    | Matthew Corr.  | 59.23 |  | 61.20 |
|    RTE    |  Accuracy | 69.67 | 69.31 | 70.75 |
|    MRPC    | Accuracy/F1  | 85.38/87.31 | 85.29/88.63 | 87.74/91.39|
|    MNLI    |  Accuracy(m/mm) |  84.90/85.10 |