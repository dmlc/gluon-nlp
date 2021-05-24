# RegionEmbedding
MXNet implementation of ICLR 2018 paper: [A new method of region embedding for text classification](https://openreview.net/forum?id=BkSDMA36Z).

Official implementation in [TensorFlow](https://github.com/text-representation/local-context-unit).

## 0.Notes

- I implemented both Word-Context and Context-Word Region Embedding in the paper.
- Please see the original papar about the datasets and pre-pocessing.
- All the hyper-parameters I used are copied from the official implementation.
## 1.Requirements

- Python2 or Python3
- Mxnet 1.2.1
## 2.Results


|Datasets| Accuracy(%)<br>WordContext|Best Epoch<br>WordContext|Accuracy(%)<br>ContextWord|Best Epoch<br>ContextWord|Running Time<br>Per Epoch(mins)
| :-- | :--: | :--: | :--: | :--: | :--: |
|Yahoo Answer|73.07(73.7)|2|73.42(73.4)|3|110|
|Amazon Polarity|95.27(95.1)|2|95.36(95.3)|3|247|
|Amazon Full|61.58(60.9)|2|61.59(60.8)|2|183|
|Ag news| 92.96(92.8)|6|92.89(92.8)|8|2|
|DBPedia|98.91(98.9)|4|98.88(98.9)|3|23|
|Yelp Full| 64.98(64.9)|3|64.94(64.5)|2|25|

Note: 
- The accuracy in brackets are results reported in the original paper.
- The running speed is much faster than the origin implementation in Tensorflow. 
  <br>The running time was tested on the model of context-word region embedding, which run roughly the same as the word-context region embedding. 
- The code run on a Titan Xp GPU.
