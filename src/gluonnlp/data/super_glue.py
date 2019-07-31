
import zipfile
import os
import re

from .dataset import JsonlDataset
from .registry import register
from ..base import get_home_dir

from mxnet.gluon.utils import download, check_sha1, _get_repo_file_url


class _SuperGlueDataset(JsonlDataset):
    def __init__(self, root, data_file):
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            os.makedirs(root)
        segment, zip_hash, data_hash = data_file
        self._root = root
        filename = os.path.join(self._root, '%s.jsonl' % segment)
        self._get_data(segment, zip_hash, data_hash, filename)
        super(_SuperGlueDataset, self).__init__(filename)

    def _get_data(self, segment, zip_hash, data_hash, filename):
        data_filename = '%s-%s.zip' % (segment, data_hash[:8])
        if not os.path.exists(filename) or not check_sha1(filename, data_hash):
            download(_get_repo_file_url(self._repo_dir(), data_filename),
                     path=self._root, sha1_hash=zip_hash)
            # unzip
            downloaded_path = os.path.join(self._root, data_filename)
            with zipfile.ZipFile(downloaded_path, 'r') as zf:
                # skip dir structures in the zip
                for zip_info in zf.infolist():
                    if zip_info.filename[-1] == '/':
                        continue
                    zip_info.filename = os.path.basename(zip_info.filename)
                    zf.extract(zip_info, self._root)

    def _repo_dir(self):
        raise NotImplementedError


@register(segment=['train', 'val', 'test'])
class SuperGlueRTE(_SuperGlueDataset):
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_rte')):
        self._segment = segment
        self._data_file = {'train': ('train', 'a4471b47b23f6d8bc2e89b2ccdcf9a3a987c69a1',
                                     '01ebec38ff3d2fdd849d3b33c2a83154d1476690'),
                           'val': ('val', '17f23360f77f04d03aee6c42a27a61a6378f1fd9',
                                   '410f8607d9fc46572c03f5488387327b33589069'),
                           'test': ('test', 'ef2de5f8351ef80036c4aeff9f3b46106b4f2835',
                                    '69f9d9b4089d0db5f0605eeaebc1c7abc044336b')}
        data_file = self._data_file[segment]

        super(SuperGlueRTE, self).__init__(root, data_file)

    def read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/RTE'


@register(segment=['train', 'val', 'test'])
class SuperGlueCB(_SuperGlueDataset):
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_cb')):
        self._segment = segment
        self._data_file = {'train': ('train', '0b27cbdbbcdf2ba82da2f760e3ab40ed694bd2b9',
                                     '193bdb772d2fe77244e5a56b4d7ac298879ec529'),
                           'val': ('val', 'e1f9dc77327eba953eb41d5f9b402127d6954ae0',
                                   'd286ac7c9f722c2b660e764ec3be11bc1e1895f8'),
                           'test': ('test', '008f9afdc868b38fdd9f989babe034a3ac35dd06',
                                    'cca70739162d54f3cd671829d009a1ab4fd8ec6a')}
        data_file = self._data_file[segment]

        super(SuperGlueCB, self).__init__(root, data_file)

    def read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/CB'


@register(segment=['train', 'val', 'test'])
class SuperGlueWSC(_SuperGlueDataset):
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_wsc')):
        self._segment = segment
        self._data_file = {'train': ('train', 'ed0fe96914cfe1ae8eb9978877273f6baed621cf',
                                     'fa978f6ad4b014b5f5282dee4b6fdfdaeeb0d0df'),
                           'val': ('val', 'cebec2f5f00baa686573ae901bb4d919ca5d3483',
                                   'ea2413e4e6f628f2bb011c44e1d8bae301375211'),
                           'test': ('test', '3313896f315e0cb2bb1f24f3baecec7fc93124de',
                                    'a47024aa81a5e7c9bc6e957b36c97f1d1b5da2fd')}
        data_file = self._data_file[segment]

        super(SuperGlueWSC, self).__init__(root, data_file)

    def read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/WSC'


@register(segment=['train', 'val', 'test'])
class SuperGlueWiC(_SuperGlueDataset):
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_wic')):
        self._segment = segment
        self._data_file = {'train': ('train', 'ec1e265bbdcde1d8da0b56948ed30d86874b1f12',
                                     '831a58c553def448e1b1d0a8a36e2b987c81bc9c'),
                           'val': ('val', '2046c43e614d98d538a03924335daae7881f77cf',
                                   '73b71136a2dc2eeb3be7ab455a08f20b8dbe7526'),
                           'test': ('test', '77af78a49aac602b7bbf080a03b644167b781ba9',
                                    '1be93932d46c8f8dc665eb7af6703c56ca1b1e08')}
        data_file = self._data_file[segment]

        super(SuperGlueWiC, self).__init__(root, data_file)

    def read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/WiC'


@register(segment=['train', 'val', 'test'])
class SuperGlueCOPA(_SuperGlueDataset):
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_copa')):
        self._segment = segment
        self._data_file = {'train': ('train', '96d20163fa8371e2676a50469d186643a07c4e7b',
                                     '5bb9c8df7b165e831613c8606a20cbe5c9622cc3'),
                           'val': ('val', 'acc13ad855a1d2750a3b746fb0cfe3ca6e8b6615',
                                   'c8b908d880ffaf69bd897d6f2a1f23b8c3a732d4'),
                           'test': ('test', '89347d7884e71b49dd73c6bcc317aef64bb1bac8',
                                    '735f39f3d31409d83b16e56ad8aed7725ef5ddd5')}
        data_file = self._data_file[segment]

        super(SuperGlueCOPA, self).__init__(root, data_file)

    def read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/COPA'


@register(segment=['train', 'val', 'test'])
class SuperGlueMultiRC(_SuperGlueDataset):
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_multirc')):
        self._segment = segment
        self._data_file = {'train': ('train', '28d908566004fb84ff81828db8955f86fb771929',
                                     '2ef471a038f0b8116bf056da6440f290be7ab96e'),
                           'val': ('val', 'af93161bb987fbafe68111bce87dece4472b4ca0',
                                   '2364ed153f4f4e8cadde78680229a8544ba427db'),
                           'test': ('test', 'eabf1e8b426a8370cd3755a99412c7871a47ffa4',
                                    'd6d1107520d535332969ffe5f5b9bd7af2a33072')}
        data_file = self._data_file[segment]

        super(SuperGlueMultiRC, self).__init__(root, data_file)

    def read_samples(self, samples):
        for i, sample in enumerate(samples):
            paragraph = dict()
            text = sample['paragraph']['text']
            sentences = self.split_text(text)
            paragraph['text'] = sentences
            paragraph['questions'] = sample['paragraph']['questions']
            samples[i] = paragraph
        return samples

    def split_text(self, text):
        text = re.sub("<b>Sent .{1,2}: </b>", "", text)
        text = text.split('<br>')
        sents = [s for s in text if len(s) > 0]
        return sents

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/MultiRC'


@register(segment=['train', 'val', 'test'])
class SuperGlueBoolQ(_SuperGlueDataset):
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_boolq')):
        self._segment = segment
        self._data_file = {'train': ('train', '89507ff3015c3212b72318fb932cfb6d4e8417ef',
                                     'd5be523290f49fc0f21f4375900451fb803817c0'),
                           'val': ('val', 'fd39562fc2c9d0b2b8289d02a8cf82aa151d0ad4',
                                   '9b09ece2b1974e4da20f0173454ba82ff8ee1710'),
                           'test': ('test', 'a805d4bd03112366d548473a6848601c042667d3',
                                    '98c308620c6d6c0768ba093858c92e5a5550ce9b')}
        data_file = self._data_file[segment]

        super(SuperGlueBoolQ, self).__init__(root, data_file)

    def read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/BoolQ'


@register(segment=['train', 'val', 'test'])
class SuperGlueReCoRD(_SuperGlueDataset):
    def __init__(self, segment='train',
                 root=os.path.join(get_home_dir(), 'datasets', 'superglue_record')):
        self._segment = segment
        self._data_file = {'train': ('train', '047282c912535c9a3bcea519935fde882feb619d',
                                     '65592074cefde2ecd1b27ce7b35eb8beb86c691a'),
                           'val': ('val', '442d8470bff2c9295231cd10262a7abf401edc64',
                                   '9d1850e4dfe2eca3b71bfea191d5f4b412c65309'),
                           'test': ('test', 'fc639a18fa87befdc52f14c1092fb40475bf50d0',
                                    'b79b22f54b5a49f98fecd05751b122ccc6947c81')}
        data_file = self._data_file[segment]

        super(SuperGlueReCoRD, self).__init__(root, data_file)

    def read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/ReCoRD'


class SuperGlueAX_b(_SuperGlueDataset):
    def __init__(self, root=os.path.join(get_home_dir(), 'datasets', 'superglue_ax_b')):
        data_file = ('AX-b', '398c5a376eb436f790723cd217ac040334140000',
                     '50fd8ac409897b652daa4b246917097c3c394bc8')

        super(SuperGlueAX_b, self).__init__(root, data_file)

    def read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/AX-b'


class SuperGlueAX_g(_SuperGlueDataset):
    def __init__(self, root=os.path.join(get_home_dir(), 'datasets', 'superglue_ax_g')):
        data_file = ('AX-g', 'd8c92498496854807dfeacd344eddf466d7f468a',
                     '8a8cbfe00fd88776a2a2f20b477e5b0c6cc8ebae')

        super(SuperGlueAX_g, self).__init__(root, data_file)

    def read_samples(self, samples):
        return samples

    def _repo_dir(self):
        return 'gluon/dataset/SUPERGLUE/AX-g'
