import argparse
import pandas as pd
import math
import os
from multiprocessing import Process
import torch
from torch.cuda import empty_cache as torch_empty_cache
from transformers import HfArgumentParser, PyTorchBenchmark, PyTorchBenchmarkArguments


HF_MODELS = [
    'bert-base-uncased',
    'bert-large-uncased',
    'albert-base-v2',
    'albert-large-v2',
    'albert-xlarge-v2',
    'albert-xxlarge-v2',
    'google/electra-small-discriminator',
    'google/electra-base-discriminator',
    'google/electra-large-discriminator',
    'google/mobilebert-uncased',
    'facebook/bart-base',
    'facebook/bart-large'
]

# (batch_size, seq_length)
train_workloads =\
    [(4, 128),
     (8, 128),
     (16, 128),
     (32, 128),
     (1, 512),
     (2, 512),
     (4, 512),
     (8, 512)]


inference_workloads = [
    (1, 128),
    (1, 384),
    (1, 512),
    (8, 32),
    (8, 128),
    (8, 512),
    (32, 512),
    (256, 128),
    (400, 100),
]


if __name__ == '__main__':
    # Profile PyTorch
    parser = HfArgumentParser(PyTorchBenchmarkArguments)
    # Benchmark Training
    for use_fp16 in [False, True]:
        df = pd.DataFrame(columns=['model', 'batch_size', 'sequence_length',
                                   'latency', 'memory'])
        for model in HF_MODELS:
            for batch_size, seq_length in train_workloads:
                prefix = '{}_{}_{}'.format(model, batch_size, seq_length).replace('/', '_')
                args = ['--models', model,
                        '--batch_sizes', '{}'.format(batch_size),
                        '--sequence_lengths', '{}'.format(seq_length),
                        '--train_time_csv_file', '{}.train_time.csv'.format(prefix),
                        '--train_memory_csv_file', '{}.train_memory.csv'.format(prefix),
                        '--no_env_print',
                        '--repeat', '3',
                        '--save_to_csv', '--training', '--no_inference']
                if use_fp16:
                    args.append('--fp16')
                benchmark_args = parser.parse_args_into_dataclasses(args)[0]
                benchmark = PyTorchBenchmark(args=benchmark_args)
                p = Process(target=benchmark.run)
                p.start()
                p.join()
                try:
                    train_time_df = pd.read_csv('{}.train_time.csv'.format(prefix))
                    train_memory_df = pd.read_csv('{}.train_memory.csv'.format(prefix))
                    latency = train_time_df['result'][0]
                    memory = train_memory_df['result'][0]
                    os.remove('{}.train_time.csv'.format(prefix))
                    os.remove('{}.train_memory.csv'.format(prefix))
                except Exception:
                    latency = math.nan
                    memory = math.nan
                new_df = pd.DataFrame({'model': [model],
                                       'batch_size': [batch_size],
                                       'sequence_length': [seq_length],
                                       'latency': [latency],
                                       'memory': [memory]})
                df = df.append(new_df, ignore_index=True)
                if use_fp16:
                    df.to_csv('pytorch_train_fp16.csv')
                else:
                    df.to_csv('pytorch_train_fp32.csv')

    # Benchmark Inference
    for torch_script in [False, True]:
        for use_fp16 in [False, True]:
            if torch_script and use_fp16:
                # Cannot support both torch_script and use_fp16.
                continue
            df = pd.DataFrame(columns=['model', 'batch_size', 'sequence_length',
                                       'latency', 'memory'])
            for model in HF_MODELS:
                for batch_size, seq_length in inference_workloads:
                    prefix = '{}_{}_{}'.format(model, batch_size, seq_length).replace('/', '_')
                    args = ['--models', model,
                            '--batch_sizes', '{}'.format(batch_size),
                            '--sequence_lengths', '{}'.format(seq_length),
                            '--inference_time_csv_file', '{}.inference_time.csv'.format(prefix),
                            '--inference_memory_csv_file', '{}.inference_memory.csv'.format(prefix),
                            '--no_env_print',
                            '--repeat', '3',
                            '--save_to_csv']
                    if use_fp16:
                        args.append('--fp16')
                    if torch_script:
                        args.append('--torchscript')
                    benchmark_args = parser.parse_args_into_dataclasses(args)[0]
                    benchmark = PyTorchBenchmark(args=benchmark_args)
                    p = Process(target=benchmark.run)
                    p.start()
                    p.join()
                    try:
                        inference_time_df = pd.read_csv('{}.inference_time.csv'.format(prefix))
                        inference_memory_df = pd.read_csv('{}.inference_memory.csv'.format(prefix))
                        latency = inference_time_df['result'][0]
                        memory = inference_memory_df['result'][0]
                        os.remove('{}.inference_time.csv'.format(prefix))
                        os.remove('{}.inference_memory.csv'.format(prefix))
                    except Exception:
                        latency = math.nan
                        memory = math.nan
                    torch_empty_cache()
                    torch.cuda.synchronize()
                    new_df = pd.DataFrame({'model': [model],
                                           'batch_size': [batch_size],
                                           'sequence_length': [seq_length],
                                           'latency': [latency],
                                           'memory': [memory]})
                    df = df.append(new_df, ignore_index=True)
                    if use_fp16 and torch_script:
                        df.to_csv('pytorch_infer_fp16_ts.csv')
                    elif use_fp16 and not torch_script:
                        df.to_csv('pytorch_infer_fp16.csv')
                    elif not use_fp16 and torch_script:
                        df.to_csv('pytorch_infer_fp32_ts.csv')
                    else:
                        df.to_csv('pytorch_infer_fp32.csv')
