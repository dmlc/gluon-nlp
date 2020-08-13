import argparse
import pandas as pd
import math
import os
from multiprocessing import Process
import torch
from typing import Callable
from transformers import HfArgumentParser, PyTorchBenchmark, PyTorchBenchmarkArguments
import logging
import timeit
logger = logging.getLogger()


class CustomizedPyTorchBenchmark(PyTorchBenchmark):
    def _prepare_train_func(self, model_name: str, batch_size: int, sequence_length: int) -> Callable[[], None]:
        _train = super(CustomizedPyTorchBenchmark, self)._prepare_train_func(model_name,
                                                                             batch_size,
                                                                             sequence_length)
        def train_fn():
            _train()
            torch.cuda.synchronize()
        return train_fn

    def _measure_speed(self, func) -> float:
        try:
            if self.args.is_tpu or self.args.torchscript:
                # run additional 10 times to stabilize compilation for tpu and torchscript
                logger.info("Do inference on TPU or torchscript. Running model 5 times to stabilize compilation")
                timeit.repeat(
                    func, repeat=1, number=3,
                )

            # as written in https://docs.python.org/2/library/timeit.html#timeit.Timer.repeat, min should be taken rather than the average
            runtimes = timeit.repeat(func, repeat=self.args.repeat, number=3,)

            if self.args.is_tpu and self.args.torch_xla_tpu_print_metrics:
                import torch_xla.debug.metrics as met

                self.print_fn(met.metrics_report())

            return min(runtimes) / 3.0
        except RuntimeError as e:
            self.print_fn("Doesn't fit on GPU. {}".format(e))
            return "N/A"


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
                benchmark = CustomizedPyTorchBenchmark(args=benchmark_args)
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
