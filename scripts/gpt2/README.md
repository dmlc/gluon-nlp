GPT2 Generator Demo in GluonNLP
-------------------------------

First, you need to convert the pretrained models in [openai/gpt-2](https://github.com/openai/gpt-2) to GluonNLP

Step-1: Download the pretrained models provided by OpenAI.
```bazaar
git clone https://github.com/openai/gpt-2.git
cd gpt-2
python download_model.py 117M
python download_model.py 345M
cd ..
```

Step-2: Convert the model to GluonNLP (Make sure the current directory is gluonnlp-gpt2)
```bazaar
python gpt2_tf_to_gluonnlp.py --src_dir gpt-2/models --dst_dir models --model 117M
python gpt2_tf_to_gluonnlp.py --src_dir gpt-2/models --dst_dir models --model 345M

```

Step-3: Test the correctness of the pretrained models:
```bazaar
nosetests3 test_transform.py
python test_model.py
```

Step-4: Try the generation demo

For the conditional generator, use the following command
```bazaar
python sampling_demo.py --model 117M
```

For the unconditional generator, use the following command
```bazaar
python sampling_demo.py --model 117M --unconditional
```

You can also try the large model with 345M parameters
```bazaar
python sampling_demo.py --model 117M
python sampling_demo.py --model 117M --unconditional

```

Dependencies
------------
```bazaar
mxnet
gluonnlp
regex
tensorflow
```
