import numpy as np
def read_data(path, slot_indexes, slots_lengthes, delim=';', pad=0, type_dict=None):
    n_slots = len(slot_indexes)
    slots = [[] for _ in range(n_slots)]
    if not type_dict:
        type_dict = {}
    with open(path, 'r') as fin:
        for i, line in enumerate(fin):
            items = line.strip().split(delim)    
            i += 1
            if i % 10000 == 1:
                print('read %d lines' % i)
            raw = []
            for index in slot_indexes:
                slot_value = items[index].split()
                tp = type_dict.get(index, int)
                raw.append([tp(x) for x in slot_value])
            for index in range(len(raw)):
                slots[index].append(pad_and_trunc(raw[index],slots_lengthes[index],pad=pad,sequence=slots_lengthes[index]>1))
    return slots
def pad_and_trunc(data, length, pad=0, sequence=False):
    if pad < 0:
        return data
    if sequence: 
        data.insert(0, pad)
        data.insert(0, pad)
        data.insert(0, pad)
        data.insert(0, pad)
    if len(data) > length:
        return data[:length]
    while len(data) < length:
        data.append(pad)
    return data
def load_data(path,max_sequence_length):
    print('Loading data...')
    indexes = [0,1]
    lengths = [1,max_sequence_length]
    print('Loading train...')
    train_path = path+'train.csv.id'
    labels_train, sequence_train = read_data(train_path, indexes, lengths)
    print('Loading test...')
    test_path = path+'test.csv.id'
    labels_test, sequence_test = read_data(test_path, indexes, lengths)
    return list(zip(sequence_test, labels_test)),list(zip(sequence_train, labels_train))
def batch_iter(data, batch_size, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield batch_num * 100.0 / num_batches_per_epoch,shuffled_data[start_index:end_index]
