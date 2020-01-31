Model Conversion Tools
----------------------

:download:`Download scripts </model_zoo/conversion_tools.zip>`

Converting DistilBERT from PyTorch Transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following command downloads the distilBERT model from pytorch-transformer,
and converts the model to Gluon.

.. code-block:: bash

    pip3 install pytorch-transformers
    python3 convert_pytorch_transformers.py --out_dir converted-model

Converting RoBERTa from Fairseq
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following command converts the `roberta checkpoint <https://github.com/pytorch/fairseq/tree/master/examples/roberta#pre-trained-models>` from fairseq to Gluon.
The converted Gluon model is saved in the same folder as the checkpoint's.

.. code-block:: bash

    pip3 install fairseq
    # download the roberta checkpoint from the website, then do:
    python3 convert_fairseq_model.py --ckpt_dir ./roberta/roberta.base --model roberta_12_768_12
