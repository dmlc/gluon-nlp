SUBWORD_ALGO=$1
SRC=de


# Fetch the raw text
nlp_data prepare_wmt \
        --mono \
        --mono_lang ${SRC} \
        --dataset newscrawl \
        --save-path wmt2014_mono


# Clean and tokenize the monolingual corpus
cd wmt2014_mono
nlp_preprocess clean_tok_para_corpus --mono \
                        --lang ${SRC} \
                        --src-corpus train.raw.${SRC} \
                        --min-num-words 1 \
                        --max-num-words 100 \
                        --src-save-path train.tok.${SRC}


# Apply the learned codes to the monolingual dataset
nlp_preprocess apply_subword --model ${SUBWORD_ALGO}\
                             --output-type subword \
                             --model-path ../wmt2014_ende/${SUBWORD_ALGO}.model \
                             --vocab-path ../wmt2014_ende/${SUBWORD_ALGO}.vocab \
                             --corpus train.tok.${SRC} \
                             --save-path train.tok.${SUBWORD_ALGO}.${SRC}

cd ../../../machine_translation

# train the model to translate German to English
SUBWORD_MODEL=yttm
python train_transformer.py \
    --train_src_corpus ../datasets/machine_translation/wmt2014_ende/train.tok.${SUBWORD_MODEL}.de \
    --train_tgt_corpus ../datasets/machine_translation/wmt2014_ende/train.tok.${SUBWORD_MODEL}.en \
    --dev_src_corpus ../datasets/machine_translation/wmt2014_ende/dev.tok.${SUBWORD_MODEL}.de \
    --dev_tgt_corpus ../datasets/machine_translation/wmt2014_ende/dev.tok.${SUBWORD_MODEL}.en \
    --src_subword_model_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.model \
    --src_vocab_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.vocab \
    --tgt_subword_model_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.model \
    --tgt_vocab_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.vocab \
    --save_dir transformer_wmt2014_deen_yttm \
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
split -l 400000 train.raw.de raw_split -d -a 2

# Get the synthetic data
cd ../../../machine_translation
for NUM in $(seq -w 0 96); do \
    printf "%s %d\n" "NUM:" $NUM | \
    python evaluate_transformer.py \
        --param_path transformer_wmt2014_deen_yttm/average.params \
        --src_lang de \
        --tgt_lang en \
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
    --train_src_corpus ../datasets/machine_translation/backtranslation/train.tok.${SUBWORD_MODEL}.en \
    --train_tgt_corpus ../datasets/machine_translation/backtranslation/train.tok.${SUBWORD_MODEL}.de \
    --dev_src_corpus ../datasets/machine_translation/wmt2014_ende/dev.tok.${SUBWORD_MODEL}.en \
    --dev_tgt_corpus ../datasets/machine_translation/wmt2014_ende/dev.tok.${SUBWORD_MODEL}.de \
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
    --src_lang en \
    --tgt_lang de \
    --cfg wmt_en_de_base.yml \
    --src_tokenizer ${SUBWORD_MODEL} \
    --tgt_tokenizer ${SUBWORD_MODEL} \
    --src_subword_model_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.model \
    --tgt_subword_model_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.model \
    --src_vocab_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.vocab \
    --tgt_vocab_path ../datasets/machine_translation/wmt2014_ende/${SUBWORD_MODEL}.vocab \
    --src_corpus ../datasets/machine_translation/wmt2014_ende/test.raw.en \
    --tgt_corpus ../datasets/machine_translation/wmt2014_ende/test.raw.de \
    --inference \
    --gpus 0,1,2,3