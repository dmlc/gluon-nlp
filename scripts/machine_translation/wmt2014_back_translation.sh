SUBWORD_ALGO=$1
SRC=en
TGT=de

# prepare en_de data for the reverse model
cd ../datasets/machine_translation
bash wmt2014_ende.sh ${SUBWORD_ALGO}

# Fetch the raw mono text
nlp_data prepare_wmt \
        --mono \
        --mono_lang ${TGT} \
        --dataset newscrawl \
        --save-path wmt2014_mono


# Clean and tokenize the monolingual corpus
cd wmt2014_mono
nlp_preprocess clean_tok_mono_corpus \
                        --lang ${TGT} \
                        --corpus train.raw.${TGT} \
                        --min-num-words 1 \
                        --max-num-words 100 \
                        --save-path train.tok.${TGT} \
                        --num-process 16

cd ../../../machine_translation
datapath=../datasets/machine_translation

# train the reverse model to translate German to English
python train_transformer.py \
    --train_src_corpus ${datapath}/wmt2014_ende/train.tok.${SUBWORD_ALGO}.${TGT} \
    --train_tgt_corpus ${datapath}/wmt2014_ende/train.tok.${SUBWORD_ALGO}.${SRC} \
    --dev_src_corpus ${datapath}/wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${TGT} \
    --dev_tgt_corpus ${datapath}/wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${SRC} \
    --src_subword_model_path ${datapath}/wmt2014_ende/${SUBWORD_ALGO}.model \
    --src_vocab_path ${datapath}/wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --tgt_subword_model_path ${datapath}/wmt2014_ende/${SUBWORD_ALGO}.model \
    --tgt_vocab_path ${datapath}/wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --save_dir transformer_wmt2014_de_en_${SUBWORD_ALGO} \
    --cfg transformer_nmt_base \
    --lr 0.002 \
    --warmup_steps 4000 \
    --warmup_init_lr 0.0 \
    --seed 100 \
    --gpus 0,1,2,3

# Due to the limited memory, we need to split the data and process the data divided respectively 
split -l 400000 ${datapath}/wmt2014_mono/train.tok.${TGT} ${datapath}/wmt2014_mono/train.tok.${TGT}.split -d -a 3

# Infer the synthetic data
# Notice that some batches are too large and GPU memory may be not enough
GPUS=(0 1 2 3)
IDX=0
for NUM in ` seq -f %03g 0 193 `; do
    split_corpus=${datapath}/wmt2014_mono/train.tok.${TGT}.split${NUM}
    if [ ${IDX} -eq ${#GPUS[@]} ]; then
        let "IDX=0"
        wait
    fi
    {
        echo processing ${split_corpus}
        python evaluate_transformer.py \
            --param_path transformer_wmt2014_de_en_${SUBWORD_ALGO}/average.params \
            --src_lang ${TGT} \
            --tgt_lang ${SRC} \
            --cfg transformer_nmt_base \
            --src_tokenizer ${SUBWORD_ALGO} \
            --tgt_tokenizer ${SUBWORD_ALGO} \
            --src_subword_model_path ${datapath}/wmt2014_ende/${SUBWORD_ALGO}.model \
            --tgt_subword_model_path ${datapath}/wmt2014_ende/${SUBWORD_ALGO}.model \
            --src_vocab_path ${datapath}/wmt2014_ende/${SUBWORD_ALGO}.vocab \
            --tgt_vocab_path ${datapath}/wmt2014_ende/${SUBWORD_ALGO}.vocab \
            --src_corpus ${split_corpus} \
            --save_dir ${split_corpus/.${TGT}./.${SRC}.} \
            --beam-size 1 \
            --inference \
            --gpus ${GPUS[IDX]}
    } &
    let "IDX++"
done
wait

cat ` seq -f "${datapath}/wmt2014_mono/train.tok.${SRC}.split%03g/pred_sentences.txt" 0 193 ` \
    > ${datapath}/wmt2014_mono/syn.train.raw.${SRC}
cp ${datapath}/wmt2014_mono/train.tok.${TGT} ${datapath}/wmt2014_mono/syn.train.raw.${TGT}

# Clean the synthetic data
nlp_preprocess clean_tok_para_corpus --src-lang ${SRC} \
    --tgt-lang ${TGT} \
    --src-corpus ${datapath}/wmt2014_mono/syn.train.raw.${SRC} \
    --tgt-corpus ${datapath}/wmt2014_mono/syn.train.raw.${TGT} \
    --min-num-words 1 \
    --max-num-words 250 \
    --max-ratio 1.5 \
    --src-save-path ${datapath}/wmt2014_mono/syn.train.tok.${SRC} \
    --tgt-save-path ${datapath}/wmt2014_mono/syn.train.tok.${TGT} \
    --num-process 32

# Combine the synthetic data with upsampled original data
# TODO upsample
rm -rf ${datapath}/wmt2014_backtranslation
mkdir ${datapath}/wmt2014_backtranslation
for LANG in ${SRC} ${TGT} ; do
    cat ${datapath}/wmt2014_ende/train.tok.${LANG} ${datapath}/wmt2014_mono/syn.train.tok.${LANG} \
        > ${datapath}/wmt2014_backtranslation/bt.train.tok.${LANG}
done

# Tokenize
for LANG in ${SRC} ${TGT} ; do
    nlp_preprocess apply_subword --model ${SUBWORD_ALGO} \
        --output-type subword \
        --model-path ${datapath}/wmt2014_ende/${SUBWORD_ALGO}.model \
        --vocab-path ${datapath}/wmt2014_ende/${SUBWORD_ALGO}.vocab \
        --corpus ${datapath}/wmt2014_backtranslation/bt.train.tok.${LANG} \
        --save-path ${datapath}/wmt2014_backtranslation/bt.train.tok.${SUBWORD_ALGO}.${LANG}
done

# Use the combine data to train the new model
python train_transformer.py \
    --train_src_corpus ${datapath}/wmt2014_backtranslation/bt.train.tok.${SUBWORD_ALGO}.${SRC} \
    --train_tgt_corpus ${datapath}/wmt2014_backtranslation/bt.train.tok.${SUBWORD_ALGO}.${TGT} \
    --dev_src_corpus ${datapath}/wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${SRC} \
    --dev_tgt_corpus ${datapath}/wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${TGT} \
    --src_subword_model_path ${datapath}/wmt2014_ende/${SUBWORD_ALGO}.model \
    --src_vocab_path ${datapath}/wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --tgt_subword_model_path ${datapath}/wmt2014_ende/${SUBWORD_ALGO}.model \
    --tgt_vocab_path ${datapath}/wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --save_dir backtranslation_transformer_wmt2014_ende_${SUBWORD_ALGO} \
    --cfg transformer_nmt_base \
    --lr 0.002 \
    --max_update 60000 \
    --save_interval_update 1000 \
    --warmup_steps 4000 \
    --warmup_init_lr 0.0 \
    --seed 100 \
    --gpus 0,1,2,3

# TODO nlp_average_checkpoint
nlp_nmt average_checkpoint --prefix range() \
    --suffix \
    --save-path backtranslation_transformer_wmt2014_ende_${SUBWORD_ALGO}/average.params

# Finally, we can evaluate the model
python evaluate_transformer.py \
    --param_path backtranslation_transformer_wmt2014_ende_${SUBWORD_ALGO}/average.params \
    --src_lang ${SRC} \
    --tgt_lang ${TGT} \
    --cfg transformer_nmt_base \
    --src_tokenizer ${SUBWORD_ALGO} \
    --tgt_tokenizer ${SUBWORD_ALGO} \
    --src_subword_model_path ${datapath}/wmt2014_ende/${SUBWORD_ALGO}.model \
    --tgt_subword_model_path ${datapath}/wmt2014_ende/${SUBWORD_ALGO}.model \
    --src_vocab_path ${datapath}/wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --tgt_vocab_path ${datapath}/wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --src_corpus ${datapath}/wmt2014_ende/test.raw.${SRC} \
    --tgt_corpus ${datapath}/wmt2014_ende/test.raw.${TGT} \
    --gpus 0
