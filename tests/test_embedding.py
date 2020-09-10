import numpy as np
import collections
import os
import tempfile
import pytest
from gluonnlp.embedding import load_embeddings, get_fasttext_model
from gluonnlp.data import Vocab

def test_load_embeddings():
    text_data = ['hello', 'world', 'hello', 'nice', 'world', 'hi', 'world', 'sadgood']
    counter = collections.Counter(text_data)
    vocab1 = Vocab(counter)
    # load with vocab
    matrix1 = load_embeddings(vocab1)
    assert len(matrix1) == len(vocab1)
    # load without vocab
    matrix2, vocab2 = load_embeddings()
    assert len(matrix2) == len(vocab2)
    np.testing.assert_almost_equal(matrix1[vocab1["hello"]], matrix2[vocab2["hello"]])

    # test_unk_method
    def simple(words):
        return np.ones((len(words), 50))
    matrix3 = load_embeddings(vocab1, unk_method=simple)
    assert sum(matrix3[vocab1['sadgood']] == 1) == matrix3.shape[-1]
    np.testing.assert_almost_equal(matrix3[vocab1["hello"]], matrix2[vocab2["hello"]])

    # load txt
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "tmp.txt")
        with open(path, "w") as f:
            f.write("{} {}\n".format(matrix1.shape[0], matrix1.shape[1]))
            for word, vec in zip(vocab1.all_tokens, matrix1):
                f.write(word + " ")
                f.write(" ".join([str(num) for num in vec.tolist()]))
                f.write("\n")
        matrix4 = load_embeddings(vocab1, path)
        np.testing.assert_almost_equal(matrix4, matrix1)


@pytest.mark.slow
@pytest.mark.remote_required
def test_get_fasttext_model():
    text_data = ['hello', 'world', 'hello', 'nice', 'world', 'hi', 'world']
    counter = collections.Counter(text_data)
    vocab1 = Vocab(counter)
    matrix1 = load_embeddings(vocab1, 'wiki.en')
    ft = get_fasttext_model('wiki.en')
    np.testing.assert_almost_equal(matrix1[vocab1["hello"]], ft['hello'], decimal=4)
    with pytest.raises(ValueError):
        get_fasttext_model('wiki.multi.ar')

