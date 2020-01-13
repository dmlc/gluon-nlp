import subprocess
import time
import sys
import mxnet as mx


arguments = ['--optimizer', 'adam', '--batch_size', '12',
             '--cpu', '0', '--epochs', '2', '--debug']
if True:
    # the downloaded bpe vocab
    url = 'http://repo.mxnet.io/gluon/dataset/vocab/test-682b5d15.bpe'
    f = mx.test_utils.download(url, overwrite=True)
    print(f)
    arguments += ['--sentencepiece', f]
print(arguments)

    #process = subprocess.check_call([sys.executable, './finetune_squad.py']
     #                               + arguments)
