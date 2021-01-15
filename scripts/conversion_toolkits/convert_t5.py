

def parse_args(): 
    pass


def convert_vocab(): 
    pass


def convert_config(): 
    pass


PARAM_MAP = [
    # 0.
    ('shared.weight', 'input_embedding_layer.weight'), 
    # 1. encoder / decoder
    ('{}.block.0.layer.0.SelfAttention.relative_attention_bias.weight', '{}.relative_position_encoder._rel_pos_embed.weight'), 
    # 2. encoder / decoder, block/layer #, layer_norm->self_attn_layer_norm / SelfAttention.o->self_attn_proj
    ('{}.block.{}.layer.0.{}.weight', '{}.layers.{}.{}.weight'), 
    # 3. encoder / decoder, block/layer #, 0.Self->self / 1.EncDec->cross, q/k/v
    ('{}.block.{}.layer.{}Attention.{}.weight', '{}.layers.{}.{}_attn_{}.weight'), 
    # 4. block/layer #, layer_norm->cross_attn_layer_norm / EncDecAttention.o->cross_attn_proj
    ('decoder.block.{}.layer.1.{}.weight', 'decoder.layers.{}.{}.weight'), 
    # 5. encoder / decoder, block/layer #, (encoder: 1 / decoder: 2), DenseReluDense.wi/wi_0/wi_1/wo / layer_norm
    ('{}.block.{}.layer.{}.{}.weight', '{}.layers.{}.ffn.{}.weight'), 
    # 6. encoder / decoder
    ('{}.final_layer_norm.weight', '{}.final_layer_norm.weight'), 
]


def convert_params(hf_t5_model, gluon_t5_model, ctx): 
    gluon_t5_model.initialize(ctx=ctx)
    hf_params = hf_t5_model.state_dict()
    gluon_params = gluon_t5_model.collect_params()
    # TODO(yongyi-wu): add sanity check, eg. param #, layer #, ffn activation, etc.
    num_layers = gluon_t5_model.num_layers

    def convert(hf_param, gluon_param): 
        gluon_params[gluon_param].set_data(hf_params[hf_param].cpu().numpy())
        
    for idx, (hf_key, gluon_key) in enumerate(PARAM_MAP): 
        if idx == 0: 
            convert(hf_key, gluon_key)
        elif idx == 1: 
            for i in ['encoder', 'decoder']: 
                convert(hf_key.format(i), gluon_key.format(i))
        elif idx in [2, 3]: 
            for stack in ['encoder', 'decoder']: 
                for layer in range(num_layers): 
                    if 'Attention' not in hf_key: 
                        for i, j in [
                            ('layer_norm', 'self_attn_layer_norm'), 
                            ('SelfAttention.o', 'self_attn_proj')
                        ]: 
                            convert(
                                hf_key.format(stack, layer, i), 
                                gluon_key.format(stack, layer, j)
                            )
                    else: 
                        for i in ['q', 'k', 'v']: 
                            convert(
                                hf_key.format(stack, layer, '0.Self', i), 
                                gluon_key.format(stack, layer, 'self', i)
                            )
                            if stack == 'decoder': 
                                convert(
                                    hf_key.format(stack, layer, '1.EncDec', i), 
                                    gluon_key.format(stack, layer, 'cross', i)
                                )
        elif idx == 4:  
            for layer in range(num_layers): 
                for i, j in [
                    ('layer_norm', 'cross_attn_layer_norm'), 
                    ('EncDecAttention.o', 'cross_attn_proj')
                ]: 
                    convert(hf_key.format(layer, i), gluon_key.format(layer, j))
        elif idx == 5:
            for stack, i in [('encoder', 1), ('decoder', 2)]: 
                for layer in range(num_layers): 
                    if gluon_t5_model.activation == 'relu': 
                        denses = ['wi', 'wo']
                    elif gluon_t5_model.activation == 'gated-gelu': 
                        denses = ['wi_0', 'wi_1', 'wo']
                    else: 
                        raise ValueError('Unrecognized feed froward activation')
                    for j in denses + ['layer_norm']: 
                        convert(
                            hf_key.format(stack, layer, i, j if j == 'layer_norm' else 'DenseReluDense.{}'.format(j)), 
                            gluon_key.format(stack, layer, j)
                        )
        elif idx == 6: 
            for stack in ['encoder', 'decoder']: 
                convert(hf_key.format(stack), hf_key.format(stack))
    return gluon_t5_model


def convert_t5_model(): 
    pass
