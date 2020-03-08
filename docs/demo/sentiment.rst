Sentiment Analysis
==================

Introduction
------------

Sentiment Analysis predicts whether an input is positive or negative. The model is based on BERT base, and are trained on the binary classification setting of the Stanford Sentiment Treebank. It achieves about 87% and 93.4% accuracy on the test set.

Demo
----

Please input the following into the text box:

.. raw:: html

   ["Positive sentiment", "Negative sentiment"]
   <form action="http://34.222.89.17:8888/bert_sst/predict" method="post">
     <label>
       <input type="text" name="data">
     </label>
     <input type="submit" value="Submit">
   </form>
