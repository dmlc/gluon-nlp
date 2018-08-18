Document Classification
-----------------------

Use the following command to train the FastText classification model on the Yelp review dataset.
The model we have implemented is a slight variant of :

- Joulin, Armand, et al. "`Bag of tricks for efficient text classification <https://arxiv.org/pdf/1607.01759.pdf>`__"

We have added dropout to the final layer, and the optimizer is changed from 'sgd' to 'adam'
These are made for demo purposes and we can get very good numbers with original settings too,
but a complete async sgd with batch size = 1, might be very slow for training using a GPU.

The datasets used in this script can be obtained with
`this script <https://github.com/facebookresearch/fastText/blob/master/classification-results.sh>`__ from fasttext.

.. code-block:: console

   $ python train_classification_fasttext.py --input yelp_review_polarity.train \
						--output yelp_review_polarity.gluon \
						 --validation dbpedia.test \
						 --ngrams 1 --epochs 25 --lr 0.1 --emsize 100 --gpu 0


It gets validation accuracy score of 93.96%. Yelp review is a binary classification dataset. (It has 2 classes)
Training logs : `log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/classification/fasttext-yelp-review.log>`__

We can call the script for multiclass classification as well without any change, it automatically figures out the number of classes
and chooses to use sigmoid or softmax loss corresponding to the problem.

Use the following commands to train a Classification model on the dbpedia dataset which has 14 labels

.. code-block:: console

   $ python train_classification_fasttext.py --input dbpedia.train \
                                                --output yelp_review_polarity.gluon \
                                                 --validation dbpedia.test \
                                                 --ngrams 1 --epochs 25 --lr 0.1 --emsize 100 --gpu 0

It gives validation accuracy of 98%. Try tweaking --ngrams to 2 or 3 for improved accuracy numbers.
Training logs : `log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/classification/fasttext-dbpedia.log>`__


Use the following command to train a Classification model on the ag_news dataset:

.. code-block:: console

   $ python train_classification_fasttext.py --input ag_news.train \
                                                --output ag_news.gluon \
                                                 --validation ag_news.test \
                                                 --ngrams 1 --epochs 25 --lr 0.1 --emsize 100 --gpu 0

It gives a validation accuracy of 91%
Training logs : `log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/classification/fasttext-ag-news.log>`__

Note: Its not advised to try higher order n-grams with large datasets since it would cause OOM on the GPU's.
You can try running it when you disable the --gpu option as many AWS EC2 instances support > 64GB RAM.
In general, larger learning rate and higher order n-grams yield better accuracy. Too high learning rate might
cause very high oscillations in the accuracy during the training.

Custom Datasets:

The training can benefit from preprocessing the dataset to lower case all the text and remove punctuations.
Use the following linux utility for achieving the same:

.. code-block:: console

	cat <input.txt> | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" > input.preprocessed.txt

