import mxnet as mx
from benchmark_utils import GluonNLPBackboneBenchmark
mx.npx.set_np()


MODELS = [
    # 'google_en_uncased_bert_base',
    # 'google_en_uncased_bert_large',
    # 'google_albert_base_v2',
    # 'google_albert_large_v2',
    # 'google_albert_xlarge_v2',
    # 'google_albert_xxlarge_v2',
    'google_electra_small',
    # 'google_electra_base',
    # 'google_electra_large',
    # 'google_uncased_mobilebert',
    'fairseq_bart_base',
    # 'fairseq_bart_large'
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
    for layout, compute_layout in [('NT', 'NT'),
                                   ('NT', 'TN'),
                                   ('TN', 'TN')]:

        if compute_layout != layout:
            profile_models = [ele for ele in MODELS if 'bart' not in ele]
        else:
            profile_models = [ele for ele in MODELS]
        inference_benchmark = GluonNLPBackboneBenchmark(
            workloads=inference_workloads,
            model_names=profile_models,
            profile_inference=True,
            profile_train=False,
            to_csv=True,
            inference_out_csv_file='gluonnlp_infer_fp32_{}_{}.csv'.format(layout, compute_layout))
        inference_benchmark.run()

        train_benchmark = GluonNLPBackboneBenchmark(
            workloads=train_workloads,
            model_names=profile_models,
            profile_inference=False,
            profile_train=True,
            to_csv=True,
            train_out_csv_file='gluonnlp_train_fp32_{}_{}.csv'.format(layout, compute_layout))
        train_benchmark.run()
