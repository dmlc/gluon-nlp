from collections import Counter
import itertools
import numpy as np
import re
import gluonnlp
from collections import Counter
# import dependencies
import mxnet as mx
import mxnet.ndarray as F
import mxnet.gluon as gluon
from mxnet.gluon import nn
from mxnet import autograd
from mxnet.contrib import text
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, rnn, Block, HybridBlock
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
import gluonnlp.data.batchify as btf
import argparse
import evaluation
import logging

class FastTextClassificationModel(HybridBlock):
    def __init__(self, vocab_size, embedding_dim, num_classes, **kwargs):
        super(FastTextClassificationModel, self).__init__(**kwargs)
        with self.name_scope():
            self.vs = vocab_size
            self.ed = embedding_dim
            self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                          weight_initializer = mx.init.Xavier(), dtype='float32')
            self.dense = nn.Dense(num_classes)

    def initialize_embedding(self,ctx):
        self.embedding.initialize(ctx=ctx)
    
    def hybrid_forward(self,F, x):
        embeddings = self.embedding(x)
        return F.Dropout(self.dense(embeddings.mean(axis = 1)),0.1)


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


def evaluate_accuracy(data_iterator, net, ctx, loss_fun):
    acc = mx.metric.Accuracy()
    loss_avg = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)#.reshape((-1,784))
        label = label.as_in_context(ctx)
        output = net(data)
        loss = loss_fun(output, label)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=nd.argmax(label, axis=1))
        loss_avg = loss_avg*i/(i+1) + nd.mean(loss).asscalar()/(i+1)
    return acc.get()[1], loss_avg


def read_input_data(filename):
    logging.info("Opening file %s for reading input",filename);
    input_file = open(filename,"r")
    data = []
    labels = []
    for line in input_file:
        tokens = line.split(",",1)
        labels.append(tokens[0].strip())
        data.append(tokens[1].strip()) 
    return labels,data

###############################################################################
# Utils
###############################################################################
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Text Classification with FastText',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Computation options
    group = parser.add_argument_group('Computation arguments')
    group.add_argument('--input', type=str, help="Input file location")
    group.add_argument('--validation', type=str, help="Validation file Location ")
    group.add_argument('--output', type=str, help="Location to save trained model")
    group.add_argument('-ngrams',type=int, default =1,help="NGrams used for training")
    group.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training.')
    group.add_argument('--epochs', type=int, default=10, help='Epoch limit')
    group.add_argument('--gpu', type=int, nargs='+',
                       help=('Number (index) of GPU to run on, e.g. 0. '
                             'If not specified, uses CPU.'))
    group.add_argument('--no-hybridize', action='store_true',
                       help='Disable hybridization of gluon HybridBlocks.')
    group.add_argument('--no-sparse-grad', action='store_true',
                       help='Disable sparse gradient support.')

    # Model
    group = parser.add_argument_group('Model arguments')
    group.add_argument('--emsize', type=int, default=100,
                       help='Size of embedding vectors.')
   

    # Optimization options
    group = parser.add_argument_group('Optimization arguments')
    group.add_argument('--optimizer', type=str, default='adagrad')
    group.add_argument('--lr', type=float, default=0.05)
    group.add_argument('--batch_size', type=float, default = 16)

    # Logging
    group = parser.add_argument_group('Logging arguments')
    group.add_argument('--logdir', type=str, default='logs',
                       help='Directory to store logs.')
    group.add_argument('--log-interval', type=int, default=100)
    group.add_argument('--eval-interval', type=int,
                       help='Evaluate every --eval-interval iterations '
                       'in addition to at the end of every epoch.')
    # Evaluation options
    evaluation.add_parameters(parser)

    args = parser.parse_args()
    evaluation.validate_args(args)
    return args



###############################################################################
# Training code
###############################################################################
def train(args):	# Load and clean data
    train_file = args.input #"/home/ubuntu/fastText/data/dbpedia.train"
    test_file = args.validation #"/home/ubuntu/fastText/data/dbpedia.test"
    ngram_range = args.ngrams
    logging.info("Ngrams range for the training run : %s",ngram_range)
    print("Loading Training data")
    labels, data = read_input_data(train_file)
    tokens_list = []
    for x in data:
        tokens_list.extend(x.split())
        
    cntr = Counter(tokens_list)
    train_vocab = gluonnlp.Vocab(cntr)
    print("Vocabulary size:",len(train_vocab))
    print("Training data converting to sequences...")
    train_sequences = [train_vocab.to_indices(x.split()) for x in data]
    train_labels = labels
    print("Reading test dataset")
    test_labels, test_data = read_input_data(test_file)
    test_sequences = [train_vocab.to_indices(x.split()) for x in test_data]

    if ngram_range >= 2:
        print('Adding {}-gram features'.format(ngram_range))
        # Create set of unique n-gram from the training set.
        ngram_set = set()
        for input_list in train_sequences:
            for i in range(2, ngram_range + 1):
                    set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                    ngram_set.update(set_of_ngram)
        start_index = len(cntr)
        token_indices = {v: k + start_index for k, v in enumerate(ngram_set)}	    
        train_sequences = add_ngram(train_sequences, token_indices, ngram_range)
        test_sequences = add_ngram(test_sequences, token_indices, ngram_range)
        print("Added n-gram features to train and test datasets!! ")
    print("Encoding labels")
    lb = LabelBinarizer()
    lb.fit(train_labels)
    encoded_y_train = lb.transform(train_labels)
    encoded_y_test = lb.transform(test_labels)
    y_train_final = encoded_y_train
    y_test_final = encoded_y_test
    #y_train_final = to_categorical(encoded_y_train)
    #y_test_final = to_categorical(encoded_y_test)
    n_labels = len(np.unique(train_labels))
    print("Number of labels:",n_labels)
    print("Initializing network")
    ctx = mx.gpu(0)
    embedding_dim = args.emsize	
    num_classes = len(np.unique(train_labels))
    print("Number of labels :",num_classes)
    net = FastTextClassificationModel(len(train_vocab), 100, num_classes)
    net.hybridize()
    net.collect_params().initialize(mx.init.Xavier(), ctx = ctx )
    print("Network initialized")

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)    

    num_epochs = args.epochs
    batch_size = args.batch_size
    print("Starting Training!")
    learning_rate = args.lr
    trainer1 = gluon.Trainer(net.embedding.collect_params(), 'adam', {'learning_rate': learning_rate})
    trainer2 = gluon.Trainer(net.dense.collect_params(), 'adam', {'learning_rate': learning_rate})
    train_batchify_fn = btf.Tuple(btf.Pad(),btf.Pad())
    train_data = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(np.array(train_sequences),
                                   mx.nd.array(y_train_final)),
                                   batch_size = batch_size, shuffle=False, batchify_fn=train_batchify_fn)
    test_data = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(test_sequences,
                                   mx.nd.array(y_test_final)),
                                   batch_size=512, shuffle=False, batchify_fn = train_batchify_fn)

    num_batches = len(train_data)/batch_size;
    display_batch_cadence = num_batches/5;
    logging.info("Number of batches for each epoch : %s, Display cadence: %s",num_batches, display_batch_cadence);
    for e in range(num_epochs):
        for batch, (data,label) in enumerate(train_data):
            #num_batches += 1
            data = data.as_in_context(ctx) 
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                cross_entropy = softmax_cross_entropy(output, label)
            cross_entropy.backward()
            trainer1.step(data.shape[0])
            trainer2.step(data.shape[0])
            if(batch%display_batch_cadence == 0):
            	print("Epoch : {}, Batches complete :{}".format(e, batch))
        print("Epoch complete :{}, Computing Accuracy".format(e))
        #test_accuracy, test_loss = evaluate_accuracy(test_data, net, softmax_cross_entropy)
        #train_accuracy, train_loss = evaluate_accuracy(train_data, net, softmax_cross_entropy)
        #print("Epochs completed : {}, Train Accuracy: {}, Train Loss: {}".format(e, train_accuracy,train_loss))
        test_accuracy, test_loss = evaluate_accuracy(test_data, net, ctx, softmax_cross_entropy)
        print("Epochs completed : {}, Test Accuracy: {}, Test Loss: {}".format(e, test_accuracy,test_loss))
        learning_rate = learning_rate * 0.5
        trainer1.set_learning_rate(learning_rate)
        trainer2.set_learning_rate(learning_rate)


if __name__=='__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    args_ = parse_args()
    train(args_)