import os
import subprocess
import time
import mxnet as mx

import pytest

@pytest.mark.serial
@pytest.mark.remote_required
def test_pretrain():
    # test data creation
    process = subprocess.check_call(['python', './scripts/bert/create_pretraining_data.py',
                                     '--input_file', './scripts/bert/sample_text.txt',
                                     '--output_dir', 'test/bert/data',
                                     '--vocab', 'book_corpus_wiki_en_uncased',
                                     '--max_seq_length', '128',
                                     '--max_predictions_per_seq', '20',
                                     '--dupe_factor', '5',
                                     '--masked_lm_prob', '0.15',
                                     '--short_seq_prob', '0.1',
                                     '--verbose'])
    try:
        # TODO(haibin) update test once MXNet 1.5 is released.
        from mxnet.ndarray.contrib import adamw_update
        # test training
        process = subprocess.check_call(['python', './scripts/bert/run_pretraining.py',
                                         '--gpus', '0',
                                         '--data', './test/bert/data/*.npz',
                                         '--batch_size', '32',
                                         '--lr', '2e-5',
                                         '--warmup_ratio', '0.5',
                                         '--num_steps', '20',
                                         '--pretrained',
                                         '--log_interval', '2',
                                         '--data_eval', './test/bert/data/*.npz',
                                         '--batch_size_eval', '8',
                                         '--ckpt_dir', './test/bert/ckpt'])
        # test evaluation
        process = subprocess.check_call(['python', './scripts/bert/run_pretraining.py',
                                         '--gpus', '0',
                                         '--num_steps', '20',
                                         '--pretrained',
                                         '--log_interval', '2',
                                         '--data_eval', './test/bert/data/*.npz',
                                         '--batch_size_eval', '8',
                                         '--ckpt_dir', './test/bert/ckpt'])

        # test mixed precision training and use-avg-len
        from mxnet.ndarray.contrib import mp_adamw_update
        process = subprocess.check_call(['python', './scripts/bert/run_pretraining.py',
                                         '--gpus', '0',
                                         '--dtype', 'float16',
                                         '--data', './test/bert/data/*.npz',
                                         '--batch_size', '4096',
                                         '--use_avg_len',
                                         '--lr', '2e-5',
                                         '--warmup_ratio', '0.5',
                                         '--num_steps', '20',
                                         '--pretrained',
                                         '--log_interval', '2',
                                         '--data_eval', './test/bert/data/*.npz',
                                         '--batch_size_eval', '8',
                                         '--ckpt_dir', './test/bert/ckpt'])
        time.sleep(5)
    except ImportError:
        print("The test expects master branch of MXNet. Skipped now.")

@pytest.mark.serial
@pytest.mark.remote_required
def test_pretrain_hvd():
    # test data creation
    process = subprocess.check_call(['python', './scripts/bert/create_pretraining_data.py',
                                     '--input_file', './scripts/bert/sample_text.txt',
                                     '--output_dir', 'test/bert/data',
                                     '--vocab', 'book_corpus_wiki_en_uncased',
                                     '--max_seq_length', '128',
                                     '--max_predictions_per_seq', '20',
                                     '--dupe_factor', '5',
                                     '--masked_lm_prob', '0.15',
                                     '--short_seq_prob', '0.1',
                                     '--verbose'])
    try:
        # TODO(haibin) update test once MXNet 1.5 is released.
        from mxnet.ndarray.contrib import adamw_update
        import horovod.mxnet as hvd
        # test training
        process = subprocess.check_call(['python', './scripts/bert/run_pretraining_hvd.py',
                                         '--data', './test/bert/data/*.npz',
                                         '--batch_size', '32',
                                         '--lr', '2e-5',
                                         '--warmup_ratio', '0.5',
                                         '--num_steps', '20',
                                         '--pretrained',
                                         '--log_interval', '2',
                                         '--data_eval', './test/bert/data/*.npz',
                                         '--batch_size_eval', '8',
                                         '--ckpt_dir', './test/bert/ckpt'])
        # test evaluation
        process = subprocess.check_call(['python', './scripts/bert/run_pretraining_hvd.py',
                                         '--num_steps', '20',
                                         '--pretrained',
                                         '--log_interval', '2',
                                         '--data_eval', './test/bert/data/*.npz',
                                         '--batch_size_eval', '8',
                                         '--ckpt_dir', './test/bert/ckpt'])

        # test mixed precision training and use-avg-len
        from mxnet.ndarray.contrib import mp_adamw_update
        process = subprocess.check_call(['python', './scripts/bert/run_pretraining_hvd.py',
                                         '--dtype', 'float16',
                                         '--data', './test/bert/data/*.npz',
                                         '--batch_size', '4096',
                                         '--use_avg_len',
                                         '--lr', '2e-5',
                                         '--warmup_ratio', '0.5',
                                         '--num_steps', '20',
                                         '--pretrained',
                                         '--log_interval', '2',
                                         '--data_eval', './test/bert/data/*.npz',
                                         '--batch_size_eval', '8',
                                         '--ckpt_dir', './test/bert/ckpt'])
        time.sleep(5)
    except ImportError:
        print("The test expects master branch of MXNet and Horovod. Skipped now.")
