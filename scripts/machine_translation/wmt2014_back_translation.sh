SUBWORD_ALGO=$1
SRC=en
TGT=de

# prepare en_de data for the reverse model
bash ../datasets/machine_translation/wmt2014_ende.sh ${SUBWORD_ALGO}

# Fetch the raw mono text
nlp_data prepare_wmt \
    --mono \
    --mono_lang ${TGT} \
    --dataset newscrawl \
    --save-path wmt2014_mono


# Clean and tokenize the monolingual corpus
cd wmt2014_mono
nlp_process clean_tok_mono_corpus \
    --lang ${TGT} \
    --corpus train.raw.${TGT} \
    --min-num-words 1 \
    --max-num-words 100 \
    --save-path train.tok.${TGT}


# train the reverse model to translate German to English
python3 train_transformer.py \
    --train_src_corpus wmt2014_ende/train.tok.${SUBWORD_ALGO}.${TGT} \
    --train_tgt_corpus wmt2014_ende/train.tok.${SUBWORD_ALGO}.${SRC} \
    --dev_src_corpus wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${TGT} \
    --dev_tgt_corpus wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${SRC} \
    --src_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --src_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --tgt_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --tgt_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --save_dir transformer_big_wmt2014_de_en_${SUBWORD_ALGO} \
    --cfg transformer_wmt_en_de_big \
    --lr 0.001 \
    --sampler BoundedBudgetSampler \
    --max_num_tokens 3584 \
    --max_update 15000 \
    --warmup_steps 4000 \
    --warmup_init_lr 0.0 \
    --seed 123 \
    --gpus 0,1,2,3

Average the last 10 checkpoints

gluon_average_checkpoint --checkpoints transformer_big_wmt2014_de_en_${SUBWORD_ALGO}/update*.params \
    --begin 21 \
    --end 30 \
    --save-path transformer_big_wmt2014_de_en_${SUBWORD_ALGO}/avg.params


# Due to the limited memory, we need to split the data and process the data divided respectively 
split -l 400000 ${datapath}/wmt2014_mono/train.tok.${TGT} ${datapath}/wmt2014_mono/train.tok.${TGT}.split -d -a 3

# Infer the synthetic data
# Notice that some batches are too large and GPU memory may be not enough
GPUS=(0 1 2 3)
IDX=0
for NUM in ` seq -f %03g 0 193 `; do
    split_corpus=wmt2014_mono/train.tok.${TGT}.split${NUM}
    if [ ${IDX} -eq ${#GPUS[@]} ]; then
        let "IDX=0"
        wait
    fi
    {
        echo processing ${split_corpus}
        python3 evaluate_transformer.py \
            --param_path transformer_big_wmt2014_de_en_${SUBWORD_ALGO}/avg.params \
            --src_lang ${TGT} \
            --tgt_lang ${SRC} \
            --cfg transformer_base \
            --src_tokenizer ${SUBWORD_ALGO} \
            --tgt_tokenizer ${SUBWORD_ALGO} \
            --src_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
            --tgt_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
            --src_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
            --tgt_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
            --src_corpus ${split_corpus} \
            --save_dir ${split_corpus/.${TGT}./.${SRC}.} \
            --beam-size 1 \
            --inference \
            --gpus ${GPUS[IDX]}
    } &
    let "IDX++"
done
wait

cat ` seq -f "wmt2014_mono/train.tok.${SRC}.split%03g/pred_sentences.txt" 0 193 ` \
    > wmt2014_mono/syn.train.raw.${SRC}
cp wmt2014_mono/train.tok.${TGT} wmt2014_mono/syn.train.raw.${TGT}

# Clean the synthetic data
nlp_process clean_tok_para_corpus --src-lang ${SRC} \
    --tgt-lang ${TGT} \
    --src-corpus wmt2014_mono/syn.train.raw.${SRC} \
    --tgt-corpus wmt2014_mono/syn.train.raw.${TGT} \
    --min-num-words 1 \
    --max-num-words 100 \
    --max-ratio 1.5 \
    --src-save-path wmt2014_mono/syn.train.tok.${SRC} \
    --tgt-save-path wmt2014_mono/syn.train.tok.${TGT}

# Combine the synthetic data with upsampled original data
# TODO upsample
mkdir -p wmt2014_backtranslation
for LANG in ${SRC} ${TGT} ; do
    cat wmt2014_ende/train.tok.${LANG} wmt2014_mono/syn.train.tok.${LANG} \
        > wmt2014_backtranslation/bt.train.tok.${LANG}
done

# Tokenize
for LANG in ${SRC} ${TGT} ; do
    nlp_process apply_subword --model ${SUBWORD_ALGO} \
        --output-type subword \
        --model-path wmt2014_ende/${SUBWORD_ALGO}.model \
        --vocab-path wmt2014_ende/${SUBWORD_ALGO}.vocab \
        --corpus wmt2014_backtranslation/bt.train.tok.${LANG} \
        --save-path wmt2014_backtranslation/bt.train.tok.${SUBWORD_ALGO}.${LANG}
done

# Use the combine data to train the new model
python3 train_transformer.py \
    --train_src_corpus wmt2014_backtranslation/bt.train.tok.${SUBWORD_ALGO}.${SRC} \
    --train_tgt_corpus wmt2014_backtranslation/bt.train.tok.${SUBWORD_ALGO}.${TGT} \
    --dev_src_corpus wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${SRC} \
    --dev_tgt_corpus wmt2014_ende/dev.tok.${SUBWORD_ALGO}.${TGT} \
    --src_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --src_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --tgt_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --tgt_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --save_dir backtranslation_transformer_big_wmt2014_ende_${SUBWORD_ALGO} \
    --cfg transformer_wmt_en_de_big \
    --lr 0.0007 \
    --sampler BoundedBudgetSampler \
    --max_num_tokens 3584 \
    --warmup_steps 4000 \
    --max_update 100000 \
    --save_interval_update 1000 \
    --warmup_init_lr 0.0 \
    --seed 123 \
    --gpus 0,1,2,3

# avg the checkpoints

# Finally, we can evaluate the model
python3 evaluate_transformer.py \
    --param_path backtranslation_transformer_big_wmt2014_ende_${SUBWORD_ALGO}/avg.params \
    --src_lang ${SRC} \
    --tgt_lang ${TGT} \
    --cfg backtranslation_transformer_big_wmt2014_ende_${SUBWORD_ALGO}/config.yml \
    --src_tokenizer ${SUBWORD_ALGO} \
    --tgt_tokenizer ${SUBWORD_ALGO} \
    --src_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --tgt_subword_model_path wmt2014_ende/${SUBWORD_ALGO}.model \
    --src_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --tgt_vocab_path wmt2014_ende/${SUBWORD_ALGO}.vocab \
    --src_corpus wmt2014_ende/test.raw.${SRC} \
    --tgt_corpus wmt2014_ende/test.raw.${TGT}
