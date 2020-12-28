SAVE_DIR=$1
SUBWORD_ALGO=${2:-yttm}
EPOCH_BEGIN=${3:-30}
EPOCH_END=${4:-60}
STOCHASTIC=${5:-0}
LP_ALPHA=${6:-0.6}
LP_K=${7:-5}
BEAM_SIZE=${8:-4}


for epoch in $( seq ${EPOCH_BEGIN} ${EPOCH_END})
do
    python3 evaluate_transformer.py \
        --param_path ${SAVE_DIR}/epoch${epoch}.params \
        --src_lang en \
        --tgt_lang de \
        --cfg ${SAVE_DIR}/config.yml \
        --src_tokenizer ${SUBWORD_ALGO} \
        --tgt_tokenizer ${SUBWORD_ALGO} \
        --src_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
        --tgt_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
        --src_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
        --tgt_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
        --src_corpus wmt2014_ende/test.raw.en \
        --tgt_corpus wmt2014_ende/test.raw.de \
        --lp_alpha ${LP_ALPHA} \
        --lp_k ${LP_K} \
        --beam-size ${BEAM_SIZE} \
        --save_dir ${SAVE_DIR}/epoch${epoch}_evaluation_alpha${LP_ALPHA}_K${LP_K}_beam${BEAM_SIZE} \
        --fp16

    python3 evaluate_transformer.py \
        --param_path ${SAVE_DIR}/epoch${epoch}.params \
        --src_lang en \
        --tgt_lang de \
        --cfg ${SAVE_DIR}/config.yml \
        --src_tokenizer ${SUBWORD_ALGO} \
        --tgt_tokenizer ${SUBWORD_ALGO} \
        --src_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
        --tgt_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
        --src_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
        --tgt_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
        --src_corpus wmt2014_ende/dev.raw.en \
        --tgt_corpus wmt2014_ende/dev.raw.de \
        --lp_alpha ${LP_ALPHA} \
        --lp_k ${LP_K} \
        --beam-size ${BEAM_SIZE} \
        --save_dir ${SAVE_DIR}/epoch${epoch}_evaluation_dev_alpha${LP_ALPHA}_K${LP_K}_beam${BEAM_SIZE} \
        --fp16
done
