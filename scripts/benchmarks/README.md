# Benchmarking the Performance of NLP Backbones

We benchmark the latency and peak memory usage of a single training (forward + backward) and inference (forward-only) step 
of the NLP backbones.
For comparison, we also provide the numbers of the models in huggingface.

## Backbones in HuggingFace

We use the [huggingface benchmark](https://github.com/huggingface/transformers/tree/master/examples/benchmarking) 
to benchmark the training + inference speed of common workloads in NLP. 

```bash
python3 -m pip install -U -r requirements.txt
python3 benchmark_hf.py
```

It will generate a list of csv files:

```
├── pytorch_train_fp32.csv
├── pytorch_train_fp16.csv
├── pytorch_infer_fp32.csv
├── pytorch_infer_fp16.csv
├── pytorch_infer_fp32_ts.csv
```

## GluonNLP Backbones based on MXNet-2.0

We profile three options: `NT` layout, `NT` layout with `TN` layout as the compute layout,
and `TN` layout.

```bash
python3 -m pip install -U -r requirements.txt
bash benchmark_gluonnlp.sh
```

It will generate csv files with `gluonnlp_` as the prefix
```
├── gluonnlp_train_fp32_NT_NT.csv
├── gluonnlp_train_fp32_NT_TN.csv
├── gluonnlp_train_fp32_TN_TN.csv
├── gluonnlp_infer_fp32_NT_NT_tvm0.csv
├── gluonnlp_infer_fp32_NT_TN_tvm0.csv
├── gluonnlp_infer_fp32_TN_TN_tvm0.csv
```

## GluonNLP + TVM for Inference

Install TVM as described in https://tvm.apache.org/docs/install/index.html

```bash
bash benchmark_gluonnlp_tvm.sh
```

```
├── gluonnlp_infer_fp32_NT_NT_tvm1.csv
├── gluonnlp_infer_fp32_NT_TN_tvm1.csv
├── gluonnlp_infer_fp32_TN_TN_tvm1.csv
```

## Generate the Benchmark Report
