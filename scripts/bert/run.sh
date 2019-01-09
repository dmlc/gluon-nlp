PYTHONPATH=~/gluon-nlp:~/bert/:~/mxnet/python/ MXNET_GPU_MEM_POOL_TYPE=Round python3 finetune_classifier.py --vocab_file $BERT_BASE_DIR/vocab.txt   --batch_size 32  --optimizer adam   --output_dir mrpc_output --epochs 3 --gpu --scaled



PYTHONPATH=~/gluon-nlp/src/:~/bert/:~/mxnet/python/ python3 convert.py --path /home/ubuntu/bert/uncased_L-24_H-1024_A-16 --dataset book_corpus_wiki_en_uncased --large;
PYTHONPATH=~/gluon-nlp/src/:~/bert/:~/mxnet/python/ python3 convert.py --path /home/ubuntu/bert/uncased_L-12_H-768_A-12 --dataset book_corpus_wiki_en_uncased;
PYTHONPATH=~/gluon-nlp/src/:~/bert/:~/mxnet/python/ python3 convert.py --path /home/ubuntu/bert/cased_L-12_H-768_A-12 --dataset book_corpus_wiki_en_cased --no-lower-case;
PYTHONPATH=~/gluon-nlp/src/:~/bert/:~/mxnet/python/ python3 convert.py --path /home/ubuntu/bert/multilingual_L-12_H-768_A-12 --dataset wiki_multilingual;



PYTHONPATH=~/gluon-nlp/src/:~/mxnet/python/ python run_pretraining.py --batch_size 8 --optimizer adam --epochs 30 --max_len 512 --dtype float32 --dropout 0.1  --static --num_gpus 1 --num_acc 0 --log_interval=10

python run_pretraining.py --gpu --data /tmp/tf_examples.npz  --do-training --num_steps 1250





python create_pretraining_data.py \
  --input_file=./sample_text.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5



PYTHONPATH=~/gluon-nlp/src/:~/mxnet/python/ python create_pretraining_data.py --input_file sample_text.txt --output_file data_gen/sample.h5py --vocab_file /home/ubuntu/.mxnet/models/book_corpus_wiki_en_uncased-c3e2bd00.vocab --do_lower_case --max_seq_length 128 --max_predictions_per_seq 20 --dupe_factor 5


PYTHONPATH=~/gluon-nlp/src/:~/mxnet/python/ python run_pretraining.py --gpu --data data_gen/sample_h5.h5py --num_steps 20 --pretrained --log_interval=1  --data_eval data_gen/sample_h5.h5py --batch_size_eval 8 --lr 2e-5 --batch_size 32  --warmup_ratio 0.5 --do-training --do-eval


PYTHONPATH=~/gluon-nlp/src/:~/mxnet/python/ python create_pretraining_data.py --input_file "sample_text.txt" --output_dir data_dir --vocab_file /home/ubuntu/.mxnet/models/book_corpus_wiki_en_uncased-c3e2bd00.vocab --do_lower_case --max_seq_length 128 --max_predictions_per_seq 20 --dupe_factor 5  --num_workers 1 --num_outputs 1 --format numpy

PYTHONPATH=~/gluon-nlp/src/:~/mxnet/python/ python run_pretraining.py --gpu --data "data_dir/*" --num_steps 20 --pretrained --log_interval=1  --data_eval "data_dir/*" --batch_size_eval 8 --lr 2e-5 --batch_size 32  --warmup_ratio 0.5 --do-training --warmup_ratio 0.5 --do-eval --seed 2

PYTHONPATH=~/gluon-nlp/src/:~/mxnet/python/ python run_pretraining.py --data "/newvolume/enwiki-samples-tokenized/part-000.npz" --num_steps 200 --pretrained --log_interval 32  --data_eval "data_dir/*" --batch_size_eval 8 --lr 2e-5 --batch_size 8  --warmup_ratio 0.5 --do-training --warmup_ratio 0.5 --do-eval --seed 2 --gpus 0,1,2,3

PYTHONPATH=~/gluon-nlp/src/:~/mxnet/python/ python create_pretraining_data.py --input_file "/newvolume/enwiki-doc-tokens/AA/wiki_00.tokens" --output_dir /newvolume/enwiki-samples-tokenized/ --vocab_file /home/ubuntu/.mxnet/models/book_corpus_wiki_en_uncased-c3e2bd00.vocab --do_lower_case --max_seq_length 512 --max_predictions_per_seq 78 --dupe_factor 5 --num_workers 72 --num_outputs 1 --format numpy --tokenized
