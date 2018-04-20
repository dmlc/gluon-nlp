from mxnet.gluon.data import Dataset
import numpy as np
import zipfile
import json


def get_tokenized_dataset(dataset, tokenizer=lambda ele: ele.split(), vocab=None):
    """Compress the dataset into

    Parameters
    ----------
    dataset : TextLineDataset
    tokenizer : function
    vocab : Vocabulary or None

    Returns
    -------
    data : np.ndarray
    indptr : np.ndarray
    word_list :
    """
    if vocab is None:
        words = set()
        for line in dataset:
            words.update(tokenizer(line))
    words = list(words)
    vocab = {ele: i for i, ele in enumerate(words)}
    data = []
    indptr = [0]
    for line in dataset:
        inds = [vocab[ele] for ele in tokenizer(line)]
        data.extend(inds)
        indptr.append(indptr[-1] + len(inds))
    return TokenizedDataset(data, indptr, words)


class TokenizedDataset(Dataset):
    def __init__(self, data, indptr, words):
        """

        Parameters
        ----------
        data : np.ndarray
        indptr : np.ndarray
        words : list of str
        """
        assert data.ndim == 1 and indptr.ndim == 1
        self._data = data
        self._indptr = indptr
        self._words = words

    def __len__(self):
        return self._indptr.size - 1

    def __getitem__(self, idx):
        return self._words[self._data[self._indptr[idx]:self._indptr[idx + 1]]]

    def save(self, file_path, compressed=True):
        if compressed:
            np.savez_compressed(file_path, data=self._data, indptr=self._indptr, words=self._words)
        else:
            np.savez(file_path, data=self._data, indptr=self._indptr, words=self._words)

    @classmethod
    def load(cls, file_path):
        npz_dict = np.load(file_path, allow_pickle=False)
        words = npz_dict['words'].tolist()
        data = np.array(npz_dict['data'])
        indptr = np.array(npz_dict['indptr'])
        return cls(data, indptr, words)
