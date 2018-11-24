data=data/snli_1.0
exp=2
gpu=0
batch_size=128
print=1000
lr=0.05
train:
	python3 main.py --train-file $(data)/train.txt --test-file $(data)/dev.txt --output-dir output/snli-$(exp) --batch-size $(batch_size) --print-interval $(print) --lr $(lr) --epochs 300 --gpu $(gpu) --dropout 0.2 --weight-decay 1e-5

test:
	python3 main.py --train-file $(data)/train.txt --test-file $(data)/test.txt --model-dir output/snli-$(exp) --mode test --output-dir output/snli-$(exp)/test


split=dev
preprocess:
	python3 preprocess.py --input data/snli_1.0/snli_1.0_$(split).txt --output data/snli_1.0/$(split).txt
