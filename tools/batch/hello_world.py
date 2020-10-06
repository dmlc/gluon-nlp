from gluonnlp.data.vocab import Vocab
import mxnet as mx


if __name__ == '__main__':
    vocab = Vocab(['Hello', 'World!'], unk_token=None)
    print(vocab)
    num_gpus = mx.context.num_gpus()
    print('Number of GPUS:', num_gpus)

