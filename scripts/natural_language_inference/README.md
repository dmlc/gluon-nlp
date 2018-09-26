# NLIGluon
Natural language inference models in Gluon.

The model is following [A Decomposable Attention Model for Natural Language Inference](https://arxiv.org/abs/1606.01933).

## Usage
Put SNLI dataset into `data/`, then train the model with

> python3 main.py


The default path for dumping models is `checkpoints/`. Then test it with:

> python3 main.py --mode test --model checkpoints/epoch-xx.gluonmodel


## Files
* `main.py`: entrance of the program
* `main_intra.py`: entrance of the program to train/test models with intra-sentence attention.
* `decomposable_atten.py`: the implementaion of decomposable attention.
* `utils.py`: some utility functions
* `nlidataset.py`: parse NLI datasets like SNLI.

## Contact
Mengxiao Lin <linmx0130@gmail.com>

