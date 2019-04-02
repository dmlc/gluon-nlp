python --version
python -c "import mxnet; print('successfully imported mxnet', mxnet); import gluonnlp; print('successfully imported gluonnlp', gluonnlp)"
python run_pretraining.py --gpus 0 --batch_size 32 --lr 2e-5 --data 'out/*.npz' --warmup_ratio 0.5 --num_steps 20 --pretrained --log_interval=2 --data_eval 'out/*.npz' --batch_size_eval 8 --ckpt_dir ckpt
