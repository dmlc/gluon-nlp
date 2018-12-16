Named Entity Recognition
---------------------------------

Named entity recognition generally refers to entities that have a specific meaning or referentiality
from unstructured text, usually including names of people, places, organizations, date and time, proper
nouns, and so on.

It is a fundamental task in natural language processing and has a wide range of applications. Traditional
named entity recognition recognition relies on a large number of manual features and domain-specific
knowledge. Although good performance is achieved in a specific field, the construction of manual features
is heavy, and such methods are generally weak in general field. In recent years, deep learning technology
has achieved rapid development and has made many breakthroughs in the field of NLP. Many methods for named entity recognition based on deep learning have been proposed and achieved excellent results.

For example, `Neural Architectures for Named Entity Recognition <https://arxiv.org/pdf/1603.01360.pdf>`__ was proposed by Guillaume Lample et al in 2016.
`End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF <https://arxiv.org/pdf/1603.01354.pdf>`__  was proposed by Xuezhe Ma et al in 2016.
And recently `ELMo (Peters, Matthew E., et al. 2018) <https://arxiv.org/pdf/1802.05365.pdf>`__ and
`BERT(Jacob Devlin, et al. 2018) <https://arxiv.org/pdf/1810.04805.pdf>`__ were published. Benefiting from the advancement of these advanced methods, the performance 
of the named entity recognition model based on the deep learning method has been greatly improved.

CharacterCNN + BiLSTM + CRF
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This method was proposed by Xuezhe Ma et al. in their 2016 paper, `End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF <https://arxiv.org/pdf/1603.01354.pdf>`__. 
The model consists of four parts.

1. First, all the characters of each word in the sentence are convoluted so that each word can have a 
representation of the convolution result, which is used to make up for the oov deficiency.

2. Use word embedding for each word, you can use different pre-training words to embed or randomize. 
Finally, the word embedding vector corresponding to each word is spliced with the convolution result 
of each word as the final representation of the word.

3. Encode the word representation of 
a sequence using bidirectional LSTM.

4. Instead of decoding each tag 
independently, attach CRF to the LSTM.


Train
""""""""""
Provide two ways to train the model.

1.  You can use the following command to run the pre-stored run_train.sh script, which uses the default 
parameters to train and validate the model on conll2003 data.
Note: This experiment code does not upload this data because of the conll2003 data license. If you run 
run_train.sh with the default parameters, you will need to download the data to the data directory and 
ensure that the --train, --valid, --test parameters in run_train.sh match the data path.

	.. code-block:: console

		$ cd code 

		$ ./run_train.sh

2. Train and validate the model on the specified data after configuring the custom parameters with the 
following command.

	.. code-block:: console

		$ cd code 

		$ python3 train_model.py --train "../data/eng_train.txt" --valid "../data/eng_testa.txt" --test "../data/eng_testb.txt" --wvp "../data/word_vocab.pkl" --cvp "../data/char_vocab.pkl" --tvp "../data/tag_vocab.pkl" --embedding glove --clpw 12 --nce 30 --nwe 100 --nf 30 --ks 3 --nhiddens 256 --nlayers 1 --nts 128 --edp 0.33 --odp 0.33 --rdp 0.33 0.5 --nepochs 200 --lr 0.01 -bc 16 --lds 1 --ldr 0.05 --op_name sgd --lp "../data/eval_files/logs.log"

Report
""""""""""
The original paper used the conll2003 data to train and validate the model. The program was also 
trained and verified on a similar conll2003 data (unofficial version). Because of the conll2003 data 
license, the data used in this experiment is slightly different from the official data. The main 
different expenditure is that the number of results of the entity tag is different from that described 
in the official version. The final experimental result reached 89.560% (±0.5)(`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/named_entity_recognition/xuzhe_charcnn_bilstm_crf_logs.log>`__).

The data difference is explained as follows:

1. Official data 
introduction:

	+----------------+--------+---------+---------+--------+
	| English data   | LOC    | MISC    | ORG     | PER    |
	+================+========+=========+=========+========+
	| Training set   |   7140 |   3438  |   6321  |   6600 |
	+----------------+--------+---------+---------+--------+
	| Development set|   1837 |   922   |   1341  |   1842 |
	+----------------+--------+---------+---------+--------+
	| Test set       |   1668 |   702   |   1661  |   1617 |
	+----------------+--------+---------+---------+--------+

2. The data used in this 
experiment(`log <https://github.com/dmlc/web-data/blob/master/gluonnlp/logs/named_entity_recognition/data_statistic.log>`__):

	+----------------+--------+---------+---------+--------+
	| English data   | LOC    | MISC    | ORG     | PER    |
	+================+========+=========+=========+========+
	| Training set   |   8297 |   4593  |   10023 |  11128 |
	+----------------+--------+---------+---------+--------+
	| Development set|   2094 |   1268  |   2092  |   3149 |
	+----------------+--------+---------+---------+--------+
	| Test set       |   1925 |   918   |   2496  |   2773 |
	+----------------+--------+---------+---------+--------+

Improvement
"""""""""""""
For the improvement of this experimental result, you can use mask(not yet completed) to reduce the 
impact of sequence filling and try different optimization methods and hyperparameters.

Follow-up Work
""""""""""""""""
The ELMo method will be used to improve the model performance based on the current experiment and make a comparative analysis.


