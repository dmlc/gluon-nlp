SUBWORD_ALGO=$1
SRC=en
TGT=de


# Fetch the raw text
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
                        --save-path train.tok.${TGT}
                        --num-process 16


# Apply the learned codes to the monolingual dataset
nlp_preprocess apply_subword --model ${SUBWORD_ALGO}\
                             --output-type subword \
                             --model-path ../wmt2014_ende/${SUBWORD_ALGO}.model \
                             --vocab-path ../wmt2014_ende/${SUBWORD_ALGO}.vocab \
                             --corpus train.tok.${TGT} \
                             --save-path train.tok.${SUBWORD_ALGO}.${TGT}

cd ../../../machine_translation

# train the reverse model to translate German to English
SUBWORD_MODEL=yttm
python train_transformer.py \
    --train_src_corpus ../datasets/machine_translation/wmt2014_ende/train.tok.${SUBWORD_MODEL}.${TGT} \
    --train_tgt_corpus ../datasets/machine_translation/wmt2014_ende/train.tok.${SUBWORD_MODEL}.${SRC} \
    --dev_src_corpus ../datasets/machine_translation/wmt2014_ende/dev.tok.${SUBWORD_MODEL}.${TGT} \
    --dev_tgt_corpus ../datasets/machine_translation/wmt2014_ende/dev.tok.${SUBWORD_MODEL}.${SRC} \
    --src_subword_model_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.model \
    --src_vocab_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.vocab \
    --tgt_subword_model_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.model \
    --tgt_vocab_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.vocab \
    --save_dir transformer_wmt2014_de_en_yttm \
    --cfg wmt_de_en_base.yml \
    --lr 0.002 \
    --warmup_steps 4000 \
    --warmup_init_lr 0.0 \
    --seed 100 \
    --gpus 0,1,2,3


cd ../datasets/machine_translation/
cp ./wmt2014_ende ./mono_dir

# Due to the limited memory, we need to split the data and process the data divided respectively 
cd mono_dir
split -l 400000 train.raw.${TGT} raw_split -d -a 2

# Get the synthetic data
cd ../../../machine_translation
for NUM in $(seq -w 0 96); do \
    printf "%s %d\n" "NUM:" $NUM | \
    python evaluate_transformer.py \
        --param_path transformer_wmt2014_de_en_yttm/average.params \
        --src_lang ${TGT} \
        --tgt_lang ${SRC} \
        --cfg wmt_en_de_base.yml \
        --src_tokenizer ${SUBWORD_MODEL} \
        --tgt_tokenizer ${SUBWORD_MODEL} \
        --src_subword_model_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.model \
        --tgt_subword_model_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.model \
        --src_vocab_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.vocab \
        --tgt_vocab_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.vocab \
        --src_corpus ../datasets/machine_translation/mono_dir/raw_split$NUM \
        --gpus 0,1,2,3
done


# Combine the synthetic data with original data
# Tokenize and clean the combined data
# Use the combine data to train the new model
SUBWORD_MODEL=yttm
python train_transformer.py \
    --train_src_corpus ../datasets/machine_translation/backtranslation/train.tok.${SUBWORD_MODEL}.${SRC} \
    --train_tgt_corpus ../datasets/machine_translation/backtranslation/train.tok.${SUBWORD_MODEL}.${TGT} \
    --dev_src_corpus ../datasets/machine_translation/wmt2014_ende/dev.tok.${SUBWORD_MODEL}.${SRC} \
    --dev_tgt_corpus ../datasets/machine_translation/wmt2014_ende/dev.tok.${SUBWORD_MODEL}.${TGT} \
    --src_subword_model_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.model \
    --src_vocab_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.vocab \
    --tgt_subword_model_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.model \
    --tgt_vocab_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.vocab \
    --save_dir backtranslation_wmt2014_ende_yttm \
    --cfg wmt_en_de_base.yml \
    --lr 0.002 \
    --warmup_steps 4000 \
    --warmup_init_lr 0.0 \
    --seed 100 \
    --gpus 0,1,2,3


# Finally, we can evaluate the model
python evaluate_transformer.py \
    --param_path backtranslation_wmt2014_ende_yttm/average.params \
    --src_lang ${SRC} \
    --tgt_lang ${TGT} \
    --cfg wmt_en_de_base.yml \
    --src_tokenizer ${SUBWORD_MODEL} \
    --tgt_tokenizer ${SUBWORD_MODEL} \
    --src_subword_model_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.model \
    --tgt_subword_model_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.model \
    --src_vocab_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.vocab \
    --tgt_vocab_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.vocab \
    --src_corpus ../datasets/machine_translation/wmt2014_ende/test.raw.${SRC} \
    --tgt_corpus ../datasets/machine_translation/wmt2014_ende/test.raw.${TGT} \
    --inference \
    --gpus 0,1,2,3