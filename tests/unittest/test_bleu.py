import string
import sys
import os
import re
import subprocess
import codecs
import numpy as np
import gluonnlp
from numpy.testing import assert_allclose

path = os.path.realpath(os.path.join(os.path.dirname(gluonnlp.__file__), '..', 'scripts', 'nmt'))
sys.path.append(path)
from bleu import compute_bleu


actions = ['deletion', 'replacement', 'add']


def _sample_translation(reference, max_len):
    translation = reference[:]
    while np.random.uniform() < 0.8 and 1 < len(translation) < max_len:
        trans_len = len(translation)
        ind = np.random.randint(trans_len)
        action = np.random.choice(actions)
        if action == 'deletion':
            del translation[ind]
        elif action == 'replacement':
            ind_rep = np.random.randint(trans_len)
            translation[ind] = translation[ind_rep]
        else:
            ind_insert = np.random.randint(trans_len)
            translation.insert(ind, translation[ind_insert])
    return translation


def _sample_reference(vocabulary, k):
    return np.random.choice(vocabulary, size=k).tolist()


def _sample_translation_corpus(reference_corpus, max_len):
    translation_corpus = []
    for references in reference_corpus:
        n_refs = len(references)
        ref_ind = np.random.randint(n_refs)
        translation = _sample_translation(references[ref_ind], max_len)
        translation_corpus.append(translation)
    return translation_corpus


def _sample_reference_corpus(vocabulary, n, max_len, n_refs=5):
    reference_corpus = []
    for i in range(n):
        references = []
        for _ in range(n_refs):
            ref_len = np.random.randint(1, max_len + 1)
            reference = _sample_reference(vocabulary, ref_len)
            references.append(reference)
        reference_corpus.append(references)
    return reference_corpus


def _write_translaton(translations, path='hypothesis'):
    out_file = codecs.open(path, 'w', 'utf-8')
    preds = [" ".join(translation) for translation in translations]
    out_file.write('\n'.join(preds) + '\n')
    out_file.flush()
    out_file.close()


def _write_reference(references, path='reference'):
    for i, reference in enumerate(zip(*references)):
        out_file = codecs.open(path + str(i), 'w', 'utf-8')
        refs = [" ".join(ref) for ref in reference]
        out_file.write('\n'.join(refs) + '\n')
        out_file.flush()
        out_file.close()


def test_bleu():
    n = 100
    max_len = 50
    n_refs = 5
    test_path = os.path.dirname(os.path.realpath(__file__))
    ref_path = os.path.join(test_path, 'reference')
    trans_path = os.path.join(test_path, 'hypothesis')
    vocabulary = list(string.ascii_lowercase)
    reference_corpus = _sample_reference_corpus(vocabulary, n, max_len, n_refs)
    translation_corpus = _sample_translation_corpus(reference_corpus, max_len)
    _write_reference(reference_corpus, path=ref_path)
    _write_translaton(translation_corpus, path=trans_path)
    ret_bleu, _, _, _, _ = compute_bleu(reference_corpus, translation_corpus)
    mose_ret = subprocess.check_output("perl %s/multi-bleu.perl %s < %s"
                                       % (path, ref_path, trans_path),
                                       shell=True).decode("utf-8")
    m = re.search("BLEU = (.+?),", mose_ret)
    gt_bleu = float(m.group(1))
    assert_allclose(round(ret_bleu * 100, 2), gt_bleu)
    os.remove(trans_path)
    for i in range(n_refs):
        os.remove(ref_path + str(i))


if __name__ == "__main__":
    import nose
    nose.runmodule()
