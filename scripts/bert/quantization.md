# Bert Quantization for MRPC and SQuAD

Tested on EC2 c5.12xlarge.

1. Install MXNet and GluonNLP

```bash
pip install mxnet-mkl --pre [--user]
pip install gluonnlp --pre [--user]
```

2. Clone BERT scripts to local

BERT scripts are provided in the GluonNLP repository.

```bash
git clone https://github.com/dmlc/gluon-nlp.git gluon-nlp
cd gluon-nlp
cd scripts/bert
```

## Sentence Classification (MRPC)

1. Fine-tune the MRPC task

```bash
python finetune_classifier.py --task_name MRPC --batch_size 32 --optimizer bertadam --epochs 3 --lr 2e-5
```

2. Run calibration

Use 1 core on 1 socket for calibration.

```bash
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=1
numactl --physcpubind=0 --membind=0 python finetune_classifier.py --task_name MRPC --epochs 1 --only_calibration --model_parameters ./output_dir/model_bert_MRPC_2.params --pad
```

`model_bert_MRPC_quantized_customize-symbol.json` and `model_bert_MRPC_quantized_customize-0000.params` will be saved in `output_dir`.

3. Run inference for latency

Use 4 cores on 1 socket for latency measurement.

```bash
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=4
# float32
numactl --physcpubind=0-3 --membind=0 python finetune_classifier.py --task_name MRPC --epochs 1 --only_inference --model_parameters ./output_dir/model_bert_MRPC_2.params --dev_batch_size 1 --pad
# int8
numactl --physcpubind=0-3 --membind=0 python finetune_classifier.py --task_name MRPC --epochs 1 --only_inference --deploy --model_prefix ./output_dir/model_bert_MRPC_quantized_customize --dev_batch_size 1 --pad
```

4. Run inference for throughput

Use full cores on 1 socket for throughput measurement. Change `--dev_batch_size` to any batch size you want. The dev dataset of MRPC only has 408 sentence pairs.

```bash
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=24
# float32
numactl --physcpubind=0-23 --membind=0 python finetune_classifier.py --task_name MRPC --epochs 1 --only_inference --model_parameters ./output_dir/model_bert_MRPC_2.params --dev_batch_size 32 --pad
# int8
numactl --physcpubind=0-23 --membind=0 python finetune_classifier.py --task_name MRPC --epochs 1 --only_inference --deploy --model_prefix ./output_dir/model_bert_MRPC_quantized_customize --dev_batch_size 32 --pad
```

## Question answering

1. Fine-tune the SQuAD task

```bash
python finetune_squad.py --optimizer adam --batch_size 12 --lr 3e-5 --epochs 2
```

2. Run calibration

Use 1 core on 1 socket for calibration.

```bash
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=1
numactl --physcpubind=0 --membind=0 python finetune_squad.py --only_calibration --model_parameters output_dir/net.params --pad
```

`model_bert_squad_quantized_customize-symbol.json` and `model_bert_squad_quantized_customize-0000.params` will be saved in `output_dir`.

3. Run inference for latency

Use 4 cores on 1 socket for latency measurement.

```bash
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=4
# float32
numactl --physcpubind=0-3 --membind=0 python finetune_squad.py --only_predict --model_parameters output_dir/net.params --test_batch_size 1 --pad
# int8
numactl --physcpubind=0-3 --membind=0 python finetune_squad.py --only_predict --deploy --model_prefix output_dir/model_bert_squad_quantized_customize --test_batch_size 1 --pad
```

4. Run inference for throughput

Use full cores on 1 socket for throughput measurement. Change `--test_batch_size` to any batch size you want.

```bash
export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
export OMP_NUM_THREADS=24
# float32
numactl --physcpubind=0-23 --membind=0 python finetune_squad.py --only_predict --model_parameters output_dir/net.params --test_batch_size 24 --pad
# int8
numactl --physcpubind=0-23 --membind=0 python finetune_squad.py --only_predict --deploy --model_prefix output_dir/model_bert_squad_quantized_customize --test_batch_size 24 --pad
```