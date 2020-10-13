from gluonnlp.utils.config import CfgNode
import re


def base_cfg():
    cfg = CfgNode()
    cfg.model_name = 'google_albert_base_v2'
    cfg.version = 2.0
    cfg.batch_size = 4
    cfg.num_accumulated = 3
    cfg.epochs = 3
    cfg.lr = 2e-5
    cfg.warmup_ratio = 0.1
    cfg.wd = 0.01
    cfg.max_grad_norm = 0.1
    cfg.max_seq_length = 512
    cfg.layerwise_decay = -1
    return cfg


def albert_base_cfg():
    return base_cfg()


def albert_large_cfg():
    cfg = base_cfg()
    cfg.model_name = 'google_albert_large_v2'
    cfg.batch_size = 3
    cfg.num_accumulated = 4
    return cfg


def albert_xlarge_cfg():
    cfg = base_cfg()
    cfg.model_name = 'google_albert_xlarge_v2'
    cfg.batch_size = 1
    cfg.num_accumulated = 12
    return cfg


def albert_xxlarge_cfg():
    cfg = albert_xlarge_cfg()
    cfg.model_name = 'google_albert_xxlarge_v2'
    return cfg


def electra_base_cfg():
    cfg = base_cfg()
    cfg.model_name = 'google_electra_base'
    cfg.batch_size = 8
    cfg.num_accumulated = 1
    cfg.lr = 1e-4
    cfg.epochs = 2
    cfg.layerwise_decay = 0.8
    cfg.wd = 0
    return cfg


def electra_large_cfg():
    cfg = electra_base_cfg()
    cfg.model_name = 'google_electra_large'
    cfg.batch_size = 2
    cfg.num_accumulated = 4
    cfg.lr = 1e-5
    cfg.layerwise_decay = 0.9
    return cfg


def electra_small_cfg():
    cfg = electra_base_cfg()
    cfg.model_name = 'google_electra_small'
    cfg.batch_size = 8
    cfg.num_accumulated = 1
    cfg.lr = 3e-4
    cfg.epochs = 2
    cfg.layerwise_decay = 0.8
    return cfg


def mobilebert_cfg():
    cfg = base_cfg()
    cfg.model_name = 'google_uncased_mobilebert'
    cfg.batch_size = 8
    cfg.num_accumulated = 1
    cfg.lr = 4e-5
    cfg.epochs = 5
    cfg.max_seq_length = 384
    return cfg


def roberta_large_cfg():
    cfg = base_cfg()
    cfg.model_name = 'fairseq_roberta_large'
    cfg.batch_size = 2
    cfg.num_accumulated = 6
    cfg.epochs = 3
    cfg.lr = 3e-5
    cfg.warmup_ratio = 0.2
    cfg.wd = 0.01
    return cfg


def uncased_bert_base_cfg():
    cfg = base_cfg()
    cfg.model_name = 'google_en_uncased_bert_base'
    cfg.batch_size = 6
    cfg.num_accumulated = 2
    cfg.lr = 3e-5
    return cfg


def uncased_bert_large_cfg():
    cfg = uncased_bert_base_cfg()
    cfg.model_name = 'google_en_uncased_bert_large'
    cfg.batch_size = 2
    cfg.num_accumulated = 6
    return cfg


def gen_command(config, template_path, out_path):
    print(f'Generating from "{template_path}" to "{out_path}"')

    def replace_fn(match):
        return str(getattr(config, match.groups()[0]))

    with open(template_path, 'r') as in_f:
        with open(out_path, 'w') as out_f:
            dat = in_f.read()
            updated_dat = re.sub(r'{{ (.+) }}', replace_fn, dat)
            out_f.write(updated_dat)


if __name__ == '__main__':
    for cfg_func in [albert_base_cfg, albert_large_cfg, albert_xlarge_cfg, albert_xxlarge_cfg,
                     electra_base_cfg, electra_large_cfg, electra_small_cfg, mobilebert_cfg,
                     roberta_large_cfg, uncased_bert_base_cfg, uncased_bert_large_cfg]:
        prefix = cfg_func.__name__[:-len('_cfg')]
        gen_command(cfg_func(), 'run_squad.template',
                    f'run_squad2_{prefix}.sh')
