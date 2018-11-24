data=data/snli_1.0
train:
	python3 main.py --train-file $(data)/train.txt --test-file $(data)/dev.txt --output-dir output/snli --batch-size 512 --print-interval 100 --lr 0.05 --epochs 50

test:
	python3 main.py --train-file $(data)/train.txt --test-file $(data)/test.txt --model-dir output/snli --mode test


split=dev
preprocess:
	python3 preprocess.py --input split/snli_1.0/snli_1.0_$(split).txt --output split/snli_1.0/$(split).txt
