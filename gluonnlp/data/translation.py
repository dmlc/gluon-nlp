import os
import zipfile
import shutil
import io

from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url
from mxnet.gluon.data import ArrayDataset

from .dataset import CorpusDataset
from ..vocab import Vocab


def _get_pair_key(src_lang, tgt_lang):
    return "_".join(sorted([src_lang, tgt_lang]))


class _TranslationDataset(ArrayDataset):
    def __init__(self, namespace, segment, src_lang, tgt_lang, root):
        assert _get_pair_key(src_lang, tgt_lang) in self._archive_file, \
            "The given language combination: src_lang=%s, tgt_lang=%s, is not supported. " \
            "Only supports language pairs = %s." \
            % (src_lang, tgt_lang, str(self._archive_file.keys()))
        self._namespace = 'gluon/dataset/{}'.format(namespace)
        self._segment = segment
        self._src_lang = src_lang
        self._tgt_lang = tgt_lang
        self._pair_key = _get_pair_key(src_lang, tgt_lang)
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        self._root = root
        [self._src_corpus_path, self._tgt_corpus_path] = self._get_data()
        src_corpus = CorpusDataset(self._src_corpus_path, skip_empty=False)
        tgt_corpus = CorpusDataset(self._tgt_corpus_path, skip_empty=False)
        super(_TranslationDataset, self).__init__(src_corpus, tgt_corpus)

    def _fetch_data_path(self, file_name_hashs):
        archive_file_name, archive_hash = self._archive_file[self._pair_key]
        paths = []
        root = self._root
        for data_file_name, data_hash in file_name_hashs:
            path = os.path.join(root, data_file_name)
            if not os.path.exists(path) or not check_sha1(path, data_hash):
                downloaded_file_path = download(_get_repo_file_url(self._namespace,
                                                                   archive_file_name),
                                                path=root,
                                                sha1_hash=archive_hash)

                with zipfile.ZipFile(downloaded_file_path, 'r') as zf:
                    for member in zf.namelist():
                        filename = os.path.basename(member)
                        if filename:
                            dest = os.path.join(root, filename)
                            with zf.open(member) as source, \
                                    open(dest, "wb") as target:
                                shutil.copyfileobj(source, target)
            paths.append(path)
        return paths

    def _get_data(self):
        src_corpus_file_name, src_corpus_hash =\
            self._data_file[self._pair_key][self._segment + "_" + self._src_lang]
        tgt_corpus_file_name, tgt_corpus_hash =\
            self._data_file[self._pair_key][self._segment + "_" + self._tgt_lang]
        return self._fetch_data_path([(src_corpus_file_name, src_corpus_hash),
                                      (tgt_corpus_file_name, tgt_corpus_hash)])

    def get_vocab(self):
        """ Get the default vocabulary.

        Returns
        -------
        src_vocab : Vocab
            Source vocabulary along with the dataset
        tgt_vocab : Vocab
            Target vocabulary along with the dataset
        """
        src_vocab_file_name, src_vocab_hash =\
            self._data_file[self._pair_key]['vocab' + "_" + self._src_lang]
        tgt_vocab_file_name, tgt_vocab_hash = \
            self._data_file[self._pair_key]['vocab' + "_" + self._tgt_lang]
        [src_vocab_path, tgt_vocab_path] =\
            self._fetch_data_path([(src_vocab_file_name, src_vocab_hash),
                                   (tgt_vocab_file_name, tgt_vocab_hash)])
        return Vocab.from_json(io.open(src_vocab_path, 'r', encoding='utf-8').read()),\
               Vocab.from_json(io.open(tgt_vocab_path, 'r', encoding='utf-8').read())


class IWSLT2015(_TranslationDataset):
    def __init__(self, segment="train", src_lang="en", tgt_lang="vi",
                 root=os.path.join('~', '.mxnet', 'datasets', 'iwslt2015')):
        assert segment in ['train', 'val', 'test'],\
            "Only supports `train`, `val`, `test` for the segment. Received segment=%s" %segment
        self._archive_file = {_get_pair_key("en", "vi"):
                                  ('iwslt15.zip', '15a05df23caccb1db458fb3f9d156308b97a217b')}
        self._data_file = {_get_pair_key("en", "vi"):
                            {'train_en': ('train.en',
                                          '675d16d057f2b6268fb294124b1646d311477325'),
                             'train_vi': ('train.vi',
                                          'bb6e21d4b02b286f2a570374b0bf22fb070589fd'),
                             'val_en' : ('tst2012.en',
                                         'e381f782d637b8db827d7b4d8bb3494822ec935e'),
                             'val_vi' : ('tst2012.vi',
                                         '4511988ce67591dc8bcdbb999314715f21e5a1e1'),
                             'test_en' : ('tst2013.en',
                                          'd320db4c8127a85de81802f239a6e6b1af473c3d'),
                             'test_vi' : ('tst2013.vi',
                                          'af212c48a68465ceada9263a049f2331f8af6290'),
                             'vocab_en' : ('vocab.en.json',
                                           'b6f8e77a45f6dce648327409acd5d52b37a45d94'), # Word Number: 17191
                             'vocab_vi' : ('vocab.vi.json',
                                           '9be11a9edd8219647754d04e0793d2d8c19dc852')}} #7709
        super(IWSLT2015, self).__init__('iwslt2015', segment=segment, src_lang=src_lang,
                                        tgt_lang=tgt_lang, root=root)


class WMT2016(_TranslationDataset):
    def __init__(self, segment="train", src_lang="en", tgt_lang="de",
                 root=os.path.join('~', '.mxnet', 'datasets', 'wmt2016')):
        assert segment in ['train', 'val', 'test'], \
            "Only supports `train`, `val`, `test` for the segment. Received segment=%s" % segment
        self._archive_file = {_get_pair_key("de", "en"):
                                  ('wmt2016_de_en.zip',
                                   '8cf0dbf6a102381443a472bcf9f181299231b496')}
        self._data_file = {_get_pair_key("de", "en"):
                               {'train_en': ('train.tok.clean.bpe.32000.en',
                                             '56f37cb4d68c2f83efd6a0c555275d1fe09f36b5'),
                                'train_de': ('train.tok.clean.bpe.32000.de',
                                             '58f30a0ba7f80a8840a5cf3deff3c147de7d3f68'),
                                'val_en': ('newstest2013.tok.bpe.32000.en',
                                           'fa03fe189fe68cb25014c5e64096ac8daf2919fa'),
                                'val_de': ('newstest2013.tok.bpe.32000.de',
                                           '7d10a884499d352c2fea6f1badafb40473737640'),
                                'test_en': ('newstest2015.tok.bpe.32000.en',
                                            'ca335076f67b2f9b98848f8abc2cd424386f2309'),
                                'test_de': ('newstest2015.tok.bpe.32000.de',
                                            'e633a3fb74506eb498fcad654d82c9b1a0a347b3'),
                                'vocab_en': ('vocab.bpe.32000.json',
                                             '1c5aea0a77cad592c4e9c1136ec3b70ceeff4e8c'),
                                'vocab_de': ('vocab.bpe.32000.json',
                                             '1c5aea0a77cad592c4e9c1136ec3b70ceeff4e8c')}}  # 36548
        super(WMT2016, self).__init__('wmt2016', segment=segment, src_lang=src_lang,
                                      tgt_lang=tgt_lang,
                                      root=os.path.join(root, _get_pair_key(src_lang, tgt_lang)))
