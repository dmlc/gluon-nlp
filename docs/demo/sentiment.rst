Sentiment Analysis
==================

Introduction
------------

Sentiment Analysis predicts whether an input is positive or negative. The model is based on BERT base, and are trained on the binary classification setting of the Stanford Sentiment Treebank. It achieves about 93.4% accuracy on the dev set.

Demo
----

Please input the following into the text box:

   ["Positive sentiment", "Negative sentiment"]

.. raw:: html

    <form action="http://34.222.89.17:8888/bert_sst/predict" method="post">
      <div class="mdl-textfield mdl-js-textfield">
        <input class="mdl-textfield__input" type="text" id="sample3" name="data"
               value="[&quot;Positive sentiment&quot;, &quot;Negative sentiment&quot;]">
      </div>
      <input type="submit" value="Submit" class="mdl-button mdl-js-button mdl-button--raised mdl-js-ripple-effect">
    </form>
