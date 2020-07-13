SUBWORD_ALGO=$1
SRC=zh
TGT=en
SAVE_PATH=wmt2017_zhen

# Fetch the raw text
nlp_data prepare_wmt \
        --dataset wmt2017 \
        --lang-pair ${SRC}-${TGT} \
        --save-path ${SAVE_PATH}

# We use sacrebleu to fetch the dev set and test set of wmt17
sacrebleu -t wmt17/dev -l ${SRC}-${TGT} --echo src > ${SAVE_PATH}/dev.raw.${SRC}
sacrebleu -t wmt17/dev -l ${SRC}-${TGT} --echo ref > ${SAVE_PATH}/dev.raw.${TGT}
sacrebleu -t wmt17 -l ${SRC}-${TGT} --echo src > ${SAVE_PATH}/test.raw.${SRC}
sacrebleu -t wmt17 -l ${SRC}-${TGT} --echo ref > ${SAVE_PATH}/test.raw.${TGT}


# Clean and tokenize the training + dev corpus
cd ${SAVE_PATH}
nlp_preprocess clean_tok_para_corpus --src-lang ${SRC} \
                      --tgt-lang ${TGT} \
                      --src-corpus train.raw.${SRC} \
                      --tgt-corpus train.raw.${TGT} \
                      --src-tokenizer jieba \
                      --tgt-tokenizer moses \
                      --max-ratio 1.3 \
                      --min-num-words 3 \
                      --max-num-words 70 \
                      --src-save-path train.tok.${SRC} \
                      --tgt-save-path train.tok.${TGT}

nlp_preprocess clean_tok_para_corpus --src-lang ${SRC} \
                      --tgt-lang ${TGT} \
                      --src-corpus dev.raw.${SRC} \
                      --tgt-corpus dev.raw.${TGT} \
                      --src-tokenizer jieba \
                      --tgt-tokenizer moses \
                      --max-ratio 1.3 \
                      --min-num-words 3 \
                      --max-num-words 70 \
                      --src-save-path dev.tok.${SRC} \
                      --tgt-save-path dev.tok.${TGT}

# For test corpus, we will just tokenize the data
nlp_preprocess clean_tok_para_corpus --src-lang ${SRC} \
                      --tgt-lang ${TGT} \
                      --src-corpus test.raw.${SRC} \
                      --tgt-corpus test.raw.${TGT} \
                      --src-tokenizer jieba \
                      --tgt-tokenizer moses \
                      --src-save-path test.tok.${SRC} \
                      --tgt-save-path test.tok.${TGT}

# Learn BPE with the training data. We learn independent source/target vocabulary

nlp_preprocess learn_subword --corpus train.tok.${SRC} \
                             --model ${SUBWORD_ALGO} \
                             --save-dir ./${SRC}_model \
                             --vocab-size 44000
nlp_preprocess learn_subword --corpus train.tok.${TGT} \
                             --model ${SUBWORD_ALGO} \
                             --save-dir ./${TGT}_model \
                             --vocab-size 33000

# Apply the learned codes to the training set
for LANG in ${SRC} ${TGT}
do
nlp_preprocess apply_subword --model ${SUBWORD_ALGO}\
                             --output-type subword \
                             --model-path ${LANG}_model/${SUBWORD_ALGO}.model \
                             --vocab-path ${LANG}_model/${SUBWORD_ALGO}.vocab \
                             --corpus train.tok.${LANG} \
                             --save-path train.tok.${SUBWORD_ALGO}.${LANG}
done

# Apply the learned codes to the dev/test set
for LANG in ${SRC} ${TGT}
do
  for SPLIT in dev test
  do
    nlp_preprocess apply_subword --model ${SUBWORD_ALGO} \
                                 --output-type subword \
                                 --model-path ${LANG}_model/${SUBWORD_ALGO}.model \
                                 --vocab-path ${LANG}_model/${SUBWORD_ALGO}.vocab \
                                 --corpus ${SPLIT}.tok.${LANG} \
                                 --save-path ${SPLIT}.tok.${SUBWORD_ALGO}.${LANG}
  done
done
