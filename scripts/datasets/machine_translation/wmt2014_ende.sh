SUBWORD_ALGO=$1
SRC=en
TGT=de
SAVE_PATH=wmt2014_ende

# Fetch the raw text
nlp_data prepare_wmt \
        --dataset wmt2014 \
        --lang-pair ${SRC}-${TGT} \
        --save-path ${SAVE_PATH}

# We use sacrebleu to fetch the dev set (newstest2013) and test set (newstest2014)
sacrebleu -t wmt13 -l ${SRC}-${TGT} --echo src > ${SAVE_PATH}/dev.raw.${SRC}
sacrebleu -t wmt13 -l ${SRC}-${TGT} --echo ref > ${SAVE_PATH}/dev.raw.${TGT}
sacrebleu -t wmt14/full -l ${SRC}-${TGT} --echo src > ${SAVE_PATH}/test.raw.${SRC}
sacrebleu -t wmt14/full -l ${SRC}-${TGT} --echo ref > ${SAVE_PATH}/test.raw.${TGT}


# Clean and tokenize the training + dev + test corpus
cd ${SAVE_PATH}
nlp_process clean_tok_para_corpus --src-lang ${SRC} \
                      --tgt-lang ${TGT} \
                      --src-corpus train.raw.${SRC} \
                      --tgt-corpus train.raw.${TGT} \
                      --src-save-path train.tok.${SRC} \
                      --tgt-save-path train.tok.${TGT}

nlp_process clean_tok_para_corpus --src-lang ${SRC} \
                      --tgt-lang ${TGT} \
                      --src-corpus dev.raw.${SRC} \
                      --tgt-corpus dev.raw.${TGT} \
                      --src-save-path dev.tok.${SRC} \
                      --tgt-save-path dev.tok.${TGT}

nlp_process clean_tok_para_corpus --src-lang ${SRC} \
                      --tgt-lang ${TGT} \
                      --src-corpus test.raw.${SRC} \
                      --tgt-corpus test.raw.${TGT} \
                      --src-save-path test.tok.${SRC} \
                      --tgt-save-path test.tok.${TGT}

# Learn BPE with the training data
nlp_process learn_subword --corpus train.tok.${SRC} train.tok.${TGT} \
                             --model ${SUBWORD_ALGO} \
                             --save-dir . \
                             --vocab-size 32768

# Apply the learned codes to the training set
for LANG in ${SRC} ${TGT}
do
nlp_process apply_subword --model ${SUBWORD_ALGO}\
                             --output-type subword \
                             --model-path ${SUBWORD_ALGO}.model \
                             --vocab-path ${SUBWORD_ALGO}.vocab \
                             --corpus train.tok.${LANG} \
                             --save-path train.tok.${SUBWORD_ALGO}.${LANG}.unclean
done

# In addition, trim the source and target sentence of the training set to 1 - 250
nlp_process clean_tok_para_corpus --src-lang ${SRC} \
                      --tgt-lang ${TGT} \
                      --src-tokenizer whitespace \
                      --tgt-tokenizer whitespace \
                      --src-corpus train.tok.${SUBWORD_ALGO}.${SRC}.unclean \
                      --tgt-corpus train.tok.${SUBWORD_ALGO}.${TGT}.unclean \
                      --src-save-path train.tok.${SUBWORD_ALGO}.${SRC} \
                      --tgt-save-path train.tok.${SUBWORD_ALGO}.${TGT} \
                      --min-num-words 1 \
                      --max-num-words 250 \
                      --max-ratio 1.5

# Apply the learned codes to the dev set
for LANG in ${SRC} ${TGT}
do
  nlp_process apply_subword --model ${SUBWORD_ALGO} \
                               --output-type subword \
                               --model-path ${SUBWORD_ALGO}.model \
                               --vocab-path ${SUBWORD_ALGO}.vocab \
                               --corpus dev.tok.${LANG} \
                               --save-path dev.tok.${SUBWORD_ALGO}.${LANG}.unclean

done
# Trim the source and target sentence of the dev set to 1 - 250
nlp_process clean_tok_para_corpus --src-lang ${SRC} \
                      --tgt-lang ${TGT} \
                      --src-tokenizer whitespace \
                      --tgt-tokenizer whitespace \
                      --src-corpus dev.tok.${SUBWORD_ALGO}.${SRC}.unclean \
                      --tgt-corpus dev.tok.${SUBWORD_ALGO}.${TGT}.unclean \
                      --src-save-path dev.tok.${SUBWORD_ALGO}.${SRC} \
                      --tgt-save-path dev.tok.${SUBWORD_ALGO}.${TGT} \
                      --min-num-words 1 \
                      --max-num-words 250 \
                      --max-ratio 1.5


# Apply the learned codes to the test set
for LANG in ${SRC} ${TGT}
do
  nlp_process apply_subword --model ${SUBWORD_ALGO} \
                               --output-type subword \
                               --model-path ${SUBWORD_ALGO}.model \
                               --vocab-path ${SUBWORD_ALGO}.vocab \
                               --corpus test.tok.${LANG} \
                               --save-path test.tok.${SUBWORD_ALGO}.${LANG}

done
