import re
import multiprocessing as mp
import gluonnlp as nlp
import time
from mxnet import nd, gluon

vocab = None
max_len = 0

def _load_file(data_name):
    if data_name == 'MR':
        train_dataset = nlp.data.MR(root='data/mr', segment='all')
        output_size = 2
        return train_dataset, output_size
    elif data_name == 'SST-1':
        train_dataset, test_dataset = [nlp.data.SST_1(root='data/sst-1', segment=segment)
                                       for segment in ('train', 'test')]
        output_size = 5
        return train_dataset, test_dataset, output_size
    elif data_name == 'SST-2':
        train_dataset, test_dataset = [nlp.data.SST_2(root='data/sst-2', segment=segment)
                                       for segment in ('train', 'test')]
        output_size = 2
        return train_dataset, test_dataset, output_size
    elif data_name == 'Subj':
        train_dataset = nlp.data.SUBJ(root='data/Subj', segment='all')
        output_size = 2
        return train_dataset, output_size
    elif data_name == 'TREC':
        train_dataset, test_dataset = [nlp.data.TREC(root='data/trec', segment=segment)
                                       for segment in ('train', 'test')]
        output_size = 6
        return train_dataset, test_dataset, output_size

def _clean_str(string, data_name):
    if data_name == 'SST-1' or data_name == 'SST-2':
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string) 
        string = re.sub(r"\s{2,}", " ", string) 
        return string.strip().lower()
    else:
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " 's", string)
        string = re.sub(r"\'ve", " 've", string)
        string = re.sub(r"n\'t", " n't", string)
        string = re.sub(r"\'re", " 're", string)
        string = re.sub(r"\'d", " 'd", string)
        string = re.sub(r"\'ll", " 'll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " ( ", string)
        string = re.sub(r"\)", " ) ", string)
        string = re.sub(r"\?", " ? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip() if data_name == 'TREC' else string.strip().lower()

def _build_vocab(data_name, train_dataset, test_dataset=[]):
    global vocab
    global max_len
    all_token = []

    for i, line in enumerate(train_dataset):
        train_dataset[i][0] = _clean_str(line[0], data_name)
        line = train_dataset[i][0].split()
        max_len = max_len if max_len > len(line) else len(line)
        all_token.extend(line)
    for i, line in enumerate(test_dataset):
        test_dataset[i][0] = _clean_str(line[0], data_name)
        line = test_dataset[i][0].split()
        max_len = max_len if max_len > len(line) else len(line)
    all_token.extend(line)
    vocab = nlp.Vocab(nlp.data.count_tokens(all_token))
    vocab.set_embedding(nlp.embedding.create('Word2Vec', source='GoogleNews-vectors-negative300'))
    for line in vocab.embedding._idx_to_token:
        if (vocab.embedding[line] == nd.zeros(300)).sum() == 300:
            vocab.embedding[line]=nd.random.uniform(-0.25, 0.25, 300)
    vocab.embedding['<unk>'] = nd.zeros(300)
    vocab.embedding['<pad>'] = nd.zeros(300)
    vocab.embedding['<bos>'] = nd.zeros(300)
    vocab.embedding['<eos>'] = nd.zeros(300)
    #return vocab, max_len

# Dataset preprocessing
def preprocess(x):
    data, label = x
    data = vocab[data.split()]
    if len(data) > max_len:
        data = data[:max_len]
    else:
        while len(data) < max_len:
            data.append(1)
    return data, label

def get_length(x):
    return float(len(x[0]))

def preprocess_dataset(dataset):
    start = time.time()
    pool = mp.Pool(8)
    dataset = pool.map(preprocess, dataset)
    lengths = gluon.data.SimpleDataset(pool.map(get_length, dataset))
    end = time.time()
    print('Done! Tokenizing Time={:.2f}s, #Sentences={}'.format(end - start, len(dataset)))
    return dataset, lengths

def load_dataset(data_name):
    if data_name == 'MR' or data_name == 'Subj':
        train_dataset, output_size = _load_file(data_name)
        _build_vocab(data_name, train_dataset)
        train_dataset, train_data_lengths = preprocess_dataset(train_dataset)
        return vocab, max_len, output_size, train_dataset, train_data_lengths
    else:
        train_dataset, test_dataset, output_size = _load_file(data_name)
        _build_vocab(data_name, train_dataset, test_dataset)
        train_dataset, train_data_lengths = preprocess_dataset(train_dataset)
        test_dataset, test_data_lengths = preprocess_dataset(test_dataset)
        return vocab, max_len, output_size, train_dataset, train_data_lengths, test_dataset, \
               test_data_lengths
