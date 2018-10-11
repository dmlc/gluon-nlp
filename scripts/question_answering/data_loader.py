r"""
This file contains the class of data loader.
"""
import json
import random

try:
    from config import DATA_PATH, DEV_FILE_NAME, TRAIN_FILE_NAME
except ImportError:
    from .config import DATA_PATH, DEV_FILE_NAME, TRAIN_FILE_NAME


class DataLoader(object):
    r"""
    An implementation of SQuAD data loader.
    """
    def __init__(self, batch_size=32, **kwargs):
        self.batch_size = batch_size
        self.data_path = DATA_PATH

        if kwargs['dev_set'] is True:
            self.data_file = DEV_FILE_NAME
            self._is_dev = True
        else:
            self.data_file = TRAIN_FILE_NAME
            self._is_dev = False

        self.data = self._load_data()
        self.num_instance = len(self.data)
        self.total_batchs = self.num_instance // self.batch_size

    def random_next_batch(self):
        r"""
        return: List
        --------
            Batchify the dataset in an ordered way.
        """
        i = 0
        while i * self.batch_size < self.num_instance:
            yield self._format_data(random.sample(self.data, self.batch_size))
            i += 1

    def next_batch(self):
        r"""
        return: List
        --------
            Batchify the dataset in random way.
        """
        i = 0
        while i * self.batch_size < self.num_instance:
            yield self._format_data(self.data[i * self.batch_size: (i + 1) * self.batch_size])
            i += 1

    def _format_data(self, data):
        def format_one_instance(instance):
            if self._is_dev:
                return instance
            else:
                return instance[1:7]
        return list(map(format_one_instance, data))

    def _load_data(self):
        with open(self.data_path + self.data_file, 'r') as f:
            line = f.readline()
        return json.loads(line)
