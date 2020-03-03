A Simple Demo for GluonNLP
==========================

Introduction
------------

This demo is using MMS Server

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
