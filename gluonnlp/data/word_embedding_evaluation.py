# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=too-many-lines
"""Word embedding evaluation datasets."""

import os
import tarfile
import zipfile

import pandas as pd
from mxnet.gluon.data.dataset import Dataset
from mxnet.gluon.utils import check_sha1, download

word_similarity_datasets = [
    'WordSim353', 'MEN', 'RadinskyMTurk', 'RareWords', 'SimLex999',
    'SimVerb3500', 'SemEval17Task2', 'BakerVerb143', 'YangPowersVerb130',
    'GoogleAnalogyTestSet', 'BiggerAnalogyTestSet'
]
__all__ = word_similarity_datasets


class _Dataset(Dataset):
    _url = None  # Dataset is retrieved from here if not cached
    _archive_file = None  # Archive name and checksum
    _checksums = None  # Checksum of archive contents

    _data = None  # Internal dataset representation

    def __init__(self, root):
        self.root = os.path.expanduser(root)
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        self._get_data()

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        raise NotImplementedError

    def _get_data(self):
        _, archive_hash = self._archive_file
        for name, checksum in self._checksums.items():
            path = os.path.join(self.root, name)
            if not os.path.exists(path) or not check_sha1(path, checksum):
                downloaded_file_path = download(
                    self._url, path=self.root, sha1_hash=archive_hash)

                if downloaded_file_path.lower().endswith('zip'):
                    with zipfile.ZipFile(downloaded_file_path, 'r') as zf:
                        zf.extractall(path=self.root)
                elif downloaded_file_path.lower().endswith('tar.gz'):
                    with tarfile.open(downloaded_file_path, 'r') as tf:
                        tf.extractall(path=self.root)
                elif len(self._checksums) > 1:
                    err = 'Failed retrieving {clsname}.'.format(
                        clsname=self.__class__.__name__)
                    err += (' Expecting multiple files, '
                            'but could not detect archive format.')
                    raise RuntimeError(err)


###############################################################################
# Word similarity and relatedness datasets
###############################################################################
class _WordSimilarityEvaluationDataset(_Dataset):
    word1 = 'word1'
    word2 = 'word2'
    score = 'score'

    def __getitem__(self, idx):
        row = self._data.iloc[idx]
        return row[[self.word1, self.word2, self.score]].values


class WordSim353(_WordSimilarityEvaluationDataset):
    """WordSim353 dataset.

    The dataset was collected by Finkelstein et al.
    (http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/). Agirre et
    al. proposed to split the collection into two datasets, one focused on
    measuring similarity, and the other one on relatedness
    (http://alfonseca.org/eng/research/wordsim353.html).

    - Finkelstein, L., Gabrilovich, E., Matias, Y., Rivlin, E., Solan, Z.,
      Wolfman, G., & Ruppin, E. (2002). Placing search in context: the concept
      revisited. ACM} Trans. Inf. Syst., 20(1), 116–131.
      http://dx.doi.org/10.1145/503104.503110
    - Agirre, E., Alfonseca, E., Hall, K. B., Kravalova, J., Pasca, M., & Soroa, A.
      (2009). A study on similarity and relatedness using distributional and
      wordnet-based approaches. In , Human Language Technologies: Conference of the
      North American Chapter of the Association of Computational Linguistics,
      Proceedings, May 31 - June 5, 2009, Boulder, Colorado, {USA (pp. 19–27). :
      The Association for Computational Linguistics.

    License: Creative Commons Attribution 4.0 International (CC BY 4.0)

    Parameters
    ----------
    segment : str
        'relatedness', 'similiarity' or 'all'
    root : str, default '~/.mxnet/datasets/wordsim353'
        Path to temp folder for storing data.

    """
    _url = 'http://alfonseca.org/pubs/ws353simrel.tar.gz'
    _archive_file = ('ws353simrel.tar.gz',
                     '1b9ca7f4d61682dea0004acbd48ce74275d5bfff')
    _checksums = {
        'wordsim353_sim_rel/wordsim353_agreed.txt':
        '1c9f77c9dd42bcc09092bd32adf0a1988d03ca80',
        'wordsim353_sim_rel/wordsim353_annotator1.txt':
        '674d5a9263d099a5128b4bf4beeaaceb80f71f4e',
        'wordsim353_sim_rel/wordsim353_annotator2.txt':
        '9b79a91861a4f1075183b93b89b73e1b470b94c1',
        'wordsim353_sim_rel/wordsim_relatedness_goldstandard.txt':
        'c36c5dc5ebea9964f4f43e2c294cd620471ab1b8',
        'wordsim353_sim_rel/wordsim_similarity_goldstandard.txt':
        '4845df518a83c8f7c527439590ed7e4c71916a99'
    }

    _data_file = {
        'relatedness': ('wordsim_relatedness_goldstandard.txt',
                        'c36c5dc5ebea9964f4f43e2c294cd620471ab1b8'),
        'similarity': ('wordsim_similarity_goldstandard.txt',
                       '4845df518a83c8f7c527439590ed7e4c71916a99')
    }

    min = 0
    max = 10

    def __init__(self,
                 segment='all',
                 root=os.path.join('~', '.mxnet', 'datasets', 'wordsim353')):
        super(WordSim353, self).__init__(root=root)
        self._segment = segment

        paths = []
        if segment == 'relatedness' or segment == 'all':
            paths.append(
                os.path.join(
                    self.root,
                    'wordsim353_sim_rel/wordsim_relatedness_goldstandard.txt'))
        if segment == 'similarity' or segment == 'all':
            paths.append(
                os.path.join(
                    self.root,
                    'wordsim353_sim_rel/wordsim_similarity_goldstandard.txt'))

        dfs = [
            pd.read_table(
                p,
                delimiter='\t',
                header=None,
                names=('word1', 'word2', 'score')) for p in paths
        ]
        df = pd.concat(dfs)

        self._data = df


class MEN(_WordSimilarityEvaluationDataset):
    """MEN dataset for word-similarity and relatedness.

    The dataset was collected by Bruni et al.
    (http://clic.cimec.unitn.it/~elia.bruni/MEN.html).

    - Bruni, E., Boleda, G., Baroni, M., & Nam-Khanh Tran (2012). Distributional
      semantics in technicolor. In , The 50th Annual Meeting of the Association for
      Computational Linguistics, Proceedings of the Conference, July 8-14, 2012,
      Jeju Island, Korea - Volume 1: Long Papers (pp. 136–145). : The Association
      for Computer Linguistics.

    License: Creative Commons Attribution 2.0 Generic (CC BY 2.0)

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/men'
        Path to temp folder for storing data.
    segment : str, default 'train'
        Dataset segment. Options are 'train', 'val', 'test'.

    """
    _url = 'http://clic.cimec.unitn.it/~elia.bruni/resources/MEN.tar.gz'
    _archive_file = ('MEN.tar.gz', '3c4af1b7009c1ad75e03562f7f7bc5f51ff3a31a')
    _checksums = {
        'MEN/MEN_dataset_lemma_form.dev':
        '55d2c9675f84dc661861172fc89db437cab2ed92',
        'MEN/MEN_dataset_lemma_form.test':
        'c003c9fddfe0ce1d38432cdb13863599d7a2d37d',
        'MEN/MEN_dataset_lemma_form_full':
        'e32e0a0fa09ccf95aa898bd42011e84419f7fafb',
        'MEN/MEN_dataset_natural_form_full':
        'af9c2ca0033e2561676872eed98e223ee6366b82',
        'MEN/agreement/agreement-score.txt':
        'bee1fe16ce63a198a12a924ceb50253c49c7b45c',
        'MEN/agreement/elias-men-ratings.txt':
        'd180252df271de96c8fbba6693eaa16793e0f7f0',
        'MEN/agreement/marcos-men-ratings.txt':
        'dbfceb7d88208c2733861f27d3d444c15db18519',
        'MEN/instructions.txt':
        'e6f69c7338246b404bafa6e24257fc4a5ba01baa',
        'MEN/licence.txt':
        'f57c6d61814a0895236ab99c06b61b2611430f92'
    }

    _segment_file = {
        'full': 'MEN/MEN_dataset_lemma_form_full',
        'dev': 'MEN/MEN_dataset_lemma_form.dev',
        'test': 'MEN/MEN_dataset_lemma_form.test',
    }

    min = 0
    max = 50

    def __init__(self,
                 segment='dev',
                 root=os.path.join('~', '.mxnet', 'datasets', 'men')):
        super(MEN, self).__init__(root=root)
        self._segment = segment

        datafilepath = os.path.join(self.root, self._segment_file[segment])
        df = pd.read_table(
            datafilepath,
            delimiter=' ',
            header=None,
            names=('word1', 'word2', 'score'))

        # Remove lemma information
        df['word1'].str.slice(0, -2)
        df['word2'].str.slice(0, -2)

        self._data = df


class RadinskyMTurk(_WordSimilarityEvaluationDataset):
    """MTurk dataset for word-similarity and relatedness by Radinsky et al..

    - Radinsky, K., Agichtein, E., Gabrilovich, E., & Markovitch, S. (2011). A word
      at a time: computing word relatedness using temporal semantic analysis. In S.
      Srinivasan, K. Ramamritham, A. Kumar, M. P. Ravindra, E. Bertino, & R. Kumar,
      Proceedings of the 20th International Conference on World Wide Web, {WWW}
      2011, Hyderabad, India, March 28 - April 1, 2011 (pp. 337–346). : ACM.

    License: Unspecified

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/men'
        Path to temp folder for storing data.

    """
    _url = 'http://www.kiraradinsky.com/files/Mtruk.csv'
    _archive_file = ('Mtruk.csv', '14959899c092148abba21401950d6957c787434c')
    _checksums = {'Mtruk.csv': '14959899c092148abba21401950d6957c787434c'}

    min = 1
    max = 5

    def __init__(self,
                 root=os.path.join('~', '.mxnet', 'datasets',
                                   'radinskymturk')):
        super(RadinskyMTurk, self).__init__(root=root)
        datafilepath = os.path.join(self.root, self._archive_file[0])
        df = pd.read_table(
            datafilepath,
            delimiter=',',
            header=None,
            names=('word1', 'word2', 'score'))
        self._data = df


class RareWords(_WordSimilarityEvaluationDataset):
    """Rare words dataset word-similarity and relatedness.

    - Luong, T., Socher, R., & Manning, C. D. (2013). Better word representations
      with recursive neural networks for morphology. In J. Hockenmaier, & S.
      Riedel, Proceedings of the Seventeenth Conference on Computational Natural
      Language Learning, CoNLL 2013, Sofia, Bulgaria, August 8-9, 2013 (pp.
      104–113). : ACL.

    License: Unspecified

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/rarewords'
        Path to temp folder for storing data.

    """
    _url = 'http://www-nlp.stanford.edu/~lmthang/morphoNLM/rw.zip'
    _archive_file = ('rw.zip', 'bf9c5959a0a2d7ed8e51d91433ac5ebf366d4fb9')
    _checksums = {'rw/rw.txt': 'bafc59f099f1798b47f5bed7b0ebbb933f6b309a'}

    min = 0
    max = 10

    def __init__(self,
                 root=os.path.join('~', '.mxnet', 'datasets', 'rarewords')):
        super(RareWords, self).__init__(root=root)
        datafilepath = os.path.join(self.root, 'rw/rw.txt')
        df = pd.read_table(
            datafilepath,
            delimiter='\t',
            header=None,
            usecols=(0, 1, 2),
            names=('word1', 'word2', 'score'))
        self._data = df


class SimLex999(_WordSimilarityEvaluationDataset):
    """SimLex999 dataset word-similarity.

    - Hill, F., Reichart, R., & Korhonen, A. (2015). Simlex-999: evaluating
      semantic models with (genuine) similarity estimation. Computational
      Linguistics, 41(4), 665–695. http://dx.doi.org/10.1162/COLI_a_00237

    License: Unspecified

    The dataset contains
    - word1: The first concept in the pair.
    - word2: The second concept in the pair. Note that the order is only
      relevant to the column Assoc(USF). These values (free association scores)
      are asymmetric. All other values are symmetric properties independent of
      the ordering word1, word2.
    - POS: The majority part-of-speech of the concept words, as determined by
      occurrence in the POS-tagged British National Corpus. Only pairs of
      matching POS are included in SimLex-999.
    - SimLex999: The SimLex999 similarity rating. Note that average annotator
      scores have been (linearly) mapped from the range [0,6] to the range
      [0,10] to match other datasets such as WordSim-353.
    - conc(w1): The concreteness rating of word1 on a scale of 1-7. Taken from
      the University of South Florida Free Association Norms database.
    - conc(w2): The concreteness rating of word2 on a scale of 1-7. Taken from
      the University of South Florida Free Association Norms database.
    - concQ: The quartile the pair occupies based on the two concreteness
      ratings. Used for some analyses in the above paper.
    - Assoc(USF): The strength of free association from word1 to word2. Values
      are taken from the University of South Florida Free Association Dataset.
    - SimAssoc333: Binary indicator of whether the pair is one of the 333 most
      associated in the dataset (according to Assoc(USF)). This subset of
      SimLex999 is often the hardest for computational models to capture
      because the noise from high association can confound the similarity
      rating. See the paper for more details.
    - SD(SimLex): The standard deviation of annotator scores when rating this
      pair. Low values indicate good agreement between the 15+ annotators on
      the similarity value SimLex999. Higher scores indicate less certainty.

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/simlex999'
        Path to temp folder for storing data.

    """
    _url = 'https://www.cl.cam.ac.uk/~fh295/SimLex-999.zip'
    _archive_file = ('SimLex-999.zip',
                     '0d3afe35b89d60acf11c28324ac7be10253fda39')
    _checksums = {
        'SimLex-999/README.txt': 'f54f4a93213b847eb93cc8952052d6b990df1bd1',
        'SimLex-999/SimLex-999.txt': '0496761e49015bc266908ea6f8e35a5ec77cb2ee'
    }

    min = 0
    max = 10

    score = 'SimLex999'

    def __init__(self,
                 root=os.path.join('~', '.mxnet', 'datasets', 'simlex999')):
        super(SimLex999, self).__init__(root=root)

        df = pd.read_table(
            os.path.join(self.root, 'SimLex-999/SimLex-999.txt'),
            delimiter='\t')
        self._data = df


class SimVerb3500(_WordSimilarityEvaluationDataset):
    """SimVerb3500 dataset word-similarity.

    - Hill, F., Reichart, R., & Korhonen, A. (2015). Simlex-999: evaluating
      semantic models with (genuine) similarity estimation. Computational
      Linguistics, 41(4), 665–695. http://dx.doi.org/10.1162/COLI_a_00237

    License: Unspecified

    The dataset contains

    - word1: The first verb of the pair.
    - word2: The second verb of the pair.
    - POS: The part-of-speech tag. Note that it is 'V' for all pairs, since the
      dataset exclusively contains verbs. We decided to include it nevertheless
      to make it compatible with SimLex-999.
    - score: The SimVerb-3500 similarity rating. Note that average annotator
      scores have been linearly mapped from the range [0,6] to the range [0,10]
      to match other datasets.
    - relation: the lexical relation of the pair. Possible values: 'SYNONYMS',
      'ANTONYMS', 'HYPER/HYPONYMS', 'COHYPONYMS', 'NONE'.

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/simverb3500'
        Path to temp folder for storing data.

    """
    _url = 'http://people.ds.cam.ac.uk/dsg40/paper/simverb/simverb-3500-data.zip'
    _archive_file = ('simverb-3500-data.zip',
                     '8c43b0a34823def29ad4c4f4d7e9d3b91acf8b8d')
    _checksums = {
        'data/README.txt':
        'fc2645b30a291a7486015c3e4b51d8eb599f7c7e',
        'data/SimVerb-3000-test.txt':
        '4cddf11f0fbbb3b94958e69b0614be5d125ec607',
        'data/SimVerb-3500-ratings.txt':
        '133d45daeb0e73b9da26930741455856887ac17b',
        'data/SimVerb-3500-stats.txt':
        '79a0fd7c6e03468742d276b127d70478a6995681',
        'data/SimVerb-3500.txt':
        '0e79af04fd42f44affc93004f2a02b62f155a9ae',
        'data/SimVerb-3520-annotator-ratings.csv':
        '9ff69cec9c93a1abba7be1404fc82d7f20e6633b',
        'data/SimVerb-500-dev.txt':
        '3ae184352ca2d9f855ca7cb099a65635d184f75a'
    }

    _segment_file = {
        'full': 'data/SimVerb-3500.txt',
        'test': 'data/SimVerb-3000-test.txt',
        'dev': 'data/SimVerb-500-dev.txt'
    }

    min = 0
    max = 10

    def __init__(self,
                 segment='full',
                 root=os.path.join('~', '.mxnet', 'datasets', 'simverb3500')):
        super(SimVerb3500, self).__init__(root=root)
        self._segment = segment

        df = pd.read_table(
            os.path.join(self.root, self._segment_file[segment]),
            header=None,
            delimiter='\t',
            names=('word1', 'word2', 'POS', 'score', 'relation'))
        self._data = df


class SemEval17Task2(_WordSimilarityEvaluationDataset):
    """SemEval17Task2 dataset for word-similarity.

    The dataset was collected by Finkelstein et al.
    (http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/). Agirre et
    al. proposed to split the collection into two datasets, one focused on
    measuring similarity, and the other one on relatedness
    (http://alfonseca.org/eng/research/wordsim353.html).

    - Finkelstein, L., Gabrilovich, E., Matias, Y., Rivlin, E., Solan, Z.,
      Wolfman, G., & Ruppin, E. (2002). Placing search in context: the concept
      revisited. ACM} Trans. Inf. Syst., 20(1), 116–131.
      http://dx.doi.org/10.1145/503104.503110
    - Agirre, E., Alfonseca, E., Hall, K. B., Kravalova, J., Pasca, M., & Soroa, A.
      (2009). A study on similarity and relatedness using distributional and
      wordnet-based approaches. In , Human Language Technologies: Conference of the
      North American Chapter of the Association of Computational Linguistics,
      Proceedings, May 31 - June 5, 2009, Boulder, Colorado, {USA (pp. 19–27). :
      The Association for Computational Linguistics.

    License: Unspecified

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/wordsim353'
        Path to temp folder for storing data.
    segment : str, default 'train'
        Dataset segment. Options are 'trial', 'test'.
    language : str, default 'en'
        Dataset language.

    """
    _url = 'http://alt.qcri.org/semeval2017/task2/data/uploads/semeval2017-task2.zip'
    _archive_file = ('semeval2017-task2.zip',
                     'b29860553f98b057303815817dfb60b9fe79cfba')
    _checksums = {
        'SemEval17-Task2/README.txt':
        'ad02d4c22fff8a39c9e89a92ba449ec78750af6b',
        'SemEval17-Task2/task2-scorer.jar':
        '145ef73ce955656d59e3b67b41f8152e8ee018d8',
        'SemEval17-Task2/test/subtask1-monolingual/data/de.test.data.txt':
        '6fc840f989d2274509549e472a68fb88dd2e149f',
        'SemEval17-Task2/test/subtask1-monolingual/data/en.test.data.txt':
        '05293fcbd80b2f4aad9b6518ce1a546ad8f61f33',
        'SemEval17-Task2/test/subtask1-monolingual/data/es.test.data.txt':
        '552904b5988f9951311290ca8fa0441dd4351d4b',
        'SemEval17-Task2/test/subtask1-monolingual/data/fa.test.data.txt':
        '29d5970feac5982961bd6ab621ba31f83d3bff77',
        'SemEval17-Task2/test/subtask1-monolingual/data/it.test.data.txt':
        'c95fe2be8fab37e9c70610117bdedc48a0a8e95c',
        'SemEval17-Task2/test/subtask1-monolingual/keys/de.test.gold.txt':
        'c51463460495a242cc726d41713c5e00b66fdd18',
        'SemEval17-Task2/test/subtask1-monolingual/keys/en.test.gold.txt':
        '2d2bb2ed41308cc60e7953cc9036f7dc89141b48',
        'SemEval17-Task2/test/subtask1-monolingual/keys/es.test.gold.txt':
        'a5842ff17fe3847d15414924826a8eb236018bcc',
        'SemEval17-Task2/test/subtask1-monolingual/keys/fa.test.gold.txt':
        '717bbe035d8ae2bad59416eb3dd4feb7238b97d4',
        'SemEval17-Task2/test/subtask1-monolingual/keys/it.test.gold.txt':
        'a342b950109c73afdc86a7829e17c1d8f7c482f0',
        'SemEval17-Task2/test/subtask2-crosslingual/data/de-es.test.data.txt':
        'ef92b1375762f68c700e050d214d3241ccde2319',
        'SemEval17-Task2/test/subtask2-crosslingual/data/de-fa.test.data.txt':
        '17aa103981f3193960309bb9b4cc151acaf8136c',
        'SemEval17-Task2/test/subtask2-crosslingual/data/de-it.test.data.txt':
        'eced15e8565689dd67605a82a782d19ee846222a',
        'SemEval17-Task2/test/subtask2-crosslingual/data/en-de.test.data.txt':
        '5cb69370a46385a7a3d37cdf2018744be77203a0',
        'SemEval17-Task2/test/subtask2-crosslingual/data/en-es.test.data.txt':
        '402f7fed52b60e915fb1be49f935395488cf7a7b',
        'SemEval17-Task2/test/subtask2-crosslingual/data/en-fa.test.data.txt':
        '9bdddbbde3da755f2a700bddfc3ed1cd9324ad48',
        'SemEval17-Task2/test/subtask2-crosslingual/data/en-it.test.data.txt':
        'd3b37aac79ca10311352309ef9b172f686ecbb80',
        'SemEval17-Task2/test/subtask2-crosslingual/data/es-fa.test.data.txt':
        'a2959aec346c26475a4a6ad4d950ee0545f2381e',
        'SemEval17-Task2/test/subtask2-crosslingual/data/es-it.test.data.txt':
        'ca627c30143d9f82a37a8776fabf2cee226dd35c',
        'SemEval17-Task2/test/subtask2-crosslingual/data/it-fa.test.data.txt':
        'a03d79a6ce7b798356b53b4e85dbe828247b97ef',
        'SemEval17-Task2/test/subtask2-crosslingual/keys/de-es.test.gold.txt':
        '7564130011d38daad582b83135010a2a58796df6',
        'SemEval17-Task2/test/subtask2-crosslingual/keys/de-fa.test.gold.txt':
        'c9e23c2e5e970e7f95550fbac3362d85b82cc569',
        'SemEval17-Task2/test/subtask2-crosslingual/keys/de-it.test.gold.txt':
        'b74cc2609b2bd2ceb5e076f504882a2e0a996a3c',
        'SemEval17-Task2/test/subtask2-crosslingual/keys/en-de.test.gold.txt':
        '428dfdad2a144642c13c24b845e6b7de6bf5f663',
        'SemEval17-Task2/test/subtask2-crosslingual/keys/en-es.test.gold.txt':
        '1dd7ab08a10552486299151cdd32ed19b56db682',
        'SemEval17-Task2/test/subtask2-crosslingual/keys/en-fa.test.gold.txt':
        '17451ac2165aa9b695dae9b1aba20eb8609fb400',
        'SemEval17-Task2/test/subtask2-crosslingual/keys/en-it.test.gold.txt':
        '5041c0b84a603ed85aa0a5cbe4b1c34f69a2fa7c',
        'SemEval17-Task2/test/subtask2-crosslingual/keys/es-fa.test.gold.txt':
        '8c09a219670dc32ab3864078bf0c28a287accabc',
        'SemEval17-Task2/test/subtask2-crosslingual/keys/es-it.test.gold.txt':
        'b1cdd13209354cc2fc2f4226c80aaa85558daf4a',
        'SemEval17-Task2/test/subtask2-crosslingual/keys/it-fa.test.gold.txt':
        'e0b560bb1d2db39ce45e841c8aad611734dc94f1',
        'SemEval17-Task2/trial/subtask1-monolingual/data/de.trial.data.txt':
        'dd071fd90f59bec8d271a447d86ee2e462941f52',
        'SemEval17-Task2/trial/subtask1-monolingual/data/en.trial.data.txt':
        'e8e5add0850b3dec07f102be26b8791a5e9bbbcf',
        'SemEval17-Task2/trial/subtask1-monolingual/data/es.trial.data.txt':
        '8956c78ff9ceae1d923a57816e55392c6a7dfc49',
        'SemEval17-Task2/trial/subtask1-monolingual/data/fa.trial.data.txt':
        '2f7c4247cde0d918b3508e90f6b49a1f5031c81b',
        'SemEval17-Task2/trial/subtask1-monolingual/data/it.trial.data.txt':
        'c11e0b5b55f94fc97c7b11fa455e71b071be879f',
        'SemEval17-Task2/trial/subtask1-monolingual/keys/de.trial.gold.txt':
        'ce5567b1accf3eb07da53229dfcb2a8a1dfac380',
        'SemEval17-Task2/trial/subtask1-monolingual/keys/en.trial.gold.txt':
        '693cb5928e807c79e39136dc0981dadca7832ae6',
        'SemEval17-Task2/trial/subtask1-monolingual/keys/es.trial.gold.txt':
        '8241ca66bf5ba55f77607e9bcfae8e34902715d8',
        'SemEval17-Task2/trial/subtask1-monolingual/keys/fa.trial.gold.txt':
        'd30701a93c8c5500b82ac2334ed8410f9a23864b',
        'SemEval17-Task2/trial/subtask1-monolingual/keys/it.trial.gold.txt':
        'bad225573e1216ba8b35429e9fa520a20e8ce031',
        'SemEval17-Task2/trial/subtask1-monolingual/output/de.trial.sample.output.txt':
        'f85cba9f6690d61736623c16e620826b09384aa5',
        'SemEval17-Task2/trial/subtask1-monolingual/output/en.trial.sample.output.txt':
        'f85cba9f6690d61736623c16e620826b09384aa5',
        'SemEval17-Task2/trial/subtask1-monolingual/output/es.trial.sample.output.txt':
        'f85cba9f6690d61736623c16e620826b09384aa5',
        'SemEval17-Task2/trial/subtask1-monolingual/output/fa.trial.sample.output.txt':
        'f85cba9f6690d61736623c16e620826b09384aa5',
        'SemEval17-Task2/trial/subtask1-monolingual/output/it.trial.sample.output.txt':
        'f85cba9f6690d61736623c16e620826b09384aa5',
        'SemEval17-Task2/trial/subtask2-crosslingual/data/de-es.trial.data.txt':
        'c27c8977d8d4434fdc3e59a7b0121d87e0a03237',
        'SemEval17-Task2/trial/subtask2-crosslingual/data/de-fa.trial.data.txt':
        '88a6f6dd1bba309f7cae7281405e37f442782983',
        'SemEval17-Task2/trial/subtask2-crosslingual/data/de-it.trial.data.txt':
        'ebdab0859f3b349fa0120fc8ab98be3394f0d73d',
        'SemEval17-Task2/trial/subtask2-crosslingual/data/en-de.trial.data.txt':
        '128d1a460fe9836b66f0fcdf59455b02edb9f258',
        'SemEval17-Task2/trial/subtask2-crosslingual/data/en-es.trial.data.txt':
        '508c5dde8ffcc32ee3009a0d020c7c96a338e1d1',
        'SemEval17-Task2/trial/subtask2-crosslingual/data/en-fa.trial.data.txt':
        '1a3640eb5facfe15b1e23a07183a2e62ed80c7d9',
        'SemEval17-Task2/trial/subtask2-crosslingual/data/en-it.trial.data.txt':
        '141c83d591b0292016583d9c23a2cc5514a006aa',
        'SemEval17-Task2/trial/subtask2-crosslingual/data/es-fa.trial.data.txt':
        'a0a548cd698c389ee80c34d6ec72abed5f1625e5',
        'SemEval17-Task2/trial/subtask2-crosslingual/data/es-it.trial.data.txt':
        '8d42bed8a43ff93d26ca95794758d9392ca707ed',
        'SemEval17-Task2/trial/subtask2-crosslingual/data/it-fa.trial.data.txt':
        '9c85223f1f734de61c28157df0ce417bb0537803',
        'SemEval17-Task2/trial/subtask2-crosslingual/keys/de-es.trial.gold.txt':
        '126c92b2fb3b8f2784dd4ae2a4c52b02a87a8196',
        'SemEval17-Task2/trial/subtask2-crosslingual/keys/de-fa.trial.gold.txt':
        '1db6201c2c8f19744c39dbde8bd4a803859d64c1',
        'SemEval17-Task2/trial/subtask2-crosslingual/keys/de-it.trial.gold.txt':
        '5300bf2ead163ff3981fb41ec5d0e291c287c9e0',
        'SemEval17-Task2/trial/subtask2-crosslingual/keys/en-de.trial.gold.txt':
        'd4f5205de929bb0c4020e1502a3f2204b5accd51',
        'SemEval17-Task2/trial/subtask2-crosslingual/keys/en-es.trial.gold.txt':
        '3237e11c3a0d9c0f5d583f8dc1d025b97a1f8bfe',
        'SemEval17-Task2/trial/subtask2-crosslingual/keys/en-fa.trial.gold.txt':
        'c14de7bf326907336a02d499c9b92ab229f3f4f8',
        'SemEval17-Task2/trial/subtask2-crosslingual/keys/en-it.trial.gold.txt':
        '3c0276c4b4e7a6d8a618bbe1ab0f30ad7b07929c',
        'SemEval17-Task2/trial/subtask2-crosslingual/keys/es-fa.trial.gold.txt':
        '359f69e9dfd6411a936baa3392b8f05c398a7707',
        'SemEval17-Task2/trial/subtask2-crosslingual/keys/es-it.trial.gold.txt':
        '44090607fabe5a26926a384e521ef1317f6f00d0',
        'SemEval17-Task2/trial/subtask2-crosslingual/keys/it-fa.trial.gold.txt':
        '97b09ffa11803023c2143fd4a4ac4bbc9775e645',
        'SemEval17-Task2/trial/subtask2-crosslingual/output/de-es.trial.sample.output.txt':
        'a0735361a692be357963959728dacef85ea08240',
        'SemEval17-Task2/trial/subtask2-crosslingual/output/de-fa.trial.sample.output.txt':
        'b71166d8615e921ee689cefc81419398d341167f',
        'SemEval17-Task2/trial/subtask2-crosslingual/output/de-it.trial.sample.output.txt':
        'b71166d8615e921ee689cefc81419398d341167f',
        'SemEval17-Task2/trial/subtask2-crosslingual/output/en-de.trial.sample.output.txt':
        'b71166d8615e921ee689cefc81419398d341167f',
        'SemEval17-Task2/trial/subtask2-crosslingual/output/en-es.trial.sample.output.txt':
        'b71166d8615e921ee689cefc81419398d341167f',
        'SemEval17-Task2/trial/subtask2-crosslingual/output/en-fa.trial.sample.output.txt':
        'a0735361a692be357963959728dacef85ea08240',
        'SemEval17-Task2/trial/subtask2-crosslingual/output/en-it.trial.sample.output.txt':
        'a0735361a692be357963959728dacef85ea08240',
        'SemEval17-Task2/trial/subtask2-crosslingual/output/es-fa.trial.sample.output.txt':
        'b71166d8615e921ee689cefc81419398d341167f',
        'SemEval17-Task2/trial/subtask2-crosslingual/output/es-it.trial.sample.output.txt':
        'b71166d8615e921ee689cefc81419398d341167f',
        'SemEval17-Task2/trial/subtask2-crosslingual/output/it-fa.trial.sample.output.txt':
        'a0735361a692be357963959728dacef85ea08240'
    }

    _segment_file = {
        'trial': 'sts-train.csv',
        'dev': 'sts-dev.csv',
        'test': 'sts-test.csv'
    }

    _datatemplate = ('SemEval17-Task2/{segment}/subtask1-monolingual/data/'
                     '{language}.{segment}.data.txt')
    _keytemplate = ('SemEval17-Task2/{segment}/subtask1-monolingual/keys/'
                    '{language}.{segment}.gold.txt')

    min = 0
    max = 5
    segments = ('trial', 'test')
    languages = ('en', 'es', 'de', 'it', 'fa')

    def __init__(self,
                 segment='trial',
                 language='en',
                 root=os.path.join('~', '.mxnet', 'datasets',
                                   'semeval17task2')):
        assert segment in self.segments
        assert language in self.languages
        super(SemEval17Task2, self).__init__(root=root)

        self._language = language
        self._segment = segment

        data = self._datatemplate.format(
            segment=self._segment, language=self._language)
        data = os.path.join(self.root, data)
        keys = self._keytemplate.format(
            segment=self._segment, language=self._language)
        keys = os.path.join(self.root, keys)
        df_data = pd.read_table(data, header=None)
        df_keys = pd.read_table(keys, header=None)
        df = pd.concat([df_data, df_keys], axis=1)
        df.columns = ['word1', 'word2', 'score']
        self._data = df


class BakerVerb143(_WordSimilarityEvaluationDataset):
    """Verb143 dataset.

    - Baker, S., Reichart, R., & Korhonen, A. (2014). An unsupervised model for
      instance level subcategorization acquisition. In A. Moschitti, B. Pang, &
      W. Daelemans, Proceedings of the 2014 Conference on Empirical Methods in
      Natural Language Processing, {EMNLP} 2014, October 25-29, 2014, Doha,
      Qatar, {A} meeting of SIGDAT, a Special Interest Group of the {ACL (pp.
      278–289). : ACL.

    143 pairs of verbs annotated by 10 annotators following the WS-353
    guidelines.

    License: unspecified

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/wordsim353'
        Path to temp folder for storing data.

    """
    _url = 'https://ie.technion.ac.il/~roiri/papers/EMNLP14.zip'
    _archive_file = ('EMNLP14.zip', '1862e52af784e76e83d472532a75eb797fb8b807')
    _checksums = {
        'verb_similarity dataset.txt':
        'd7e4820c7504cbae56898353e4d94e6408c330fc'
    }

    min = 0
    max = 10

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets',
                                         'verb143')):
        super(BakerVerb143, self).__init__(root=root)

        path = os.path.join(self.root, 'verb_similarity dataset.txt')
        df = pd.read_table(
            path,
            delimiter='\t',
            header=None,
            names=('word1', 'word2', 'score'))

        self._data = df


class YangPowersVerb130(_WordSimilarityEvaluationDataset):
    """Verb-130 dataset.

    - Yang, D., & Powers, D. M. (2006). Verb similarity on the taxonomy of
      wordnet. In The Third International WordNet Conference: GWC 2006

    License: Unspecified

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/wordsim353'
        Path to temp folder for storing data.

    """

    _words1 = [
        'brag', 'concoct', 'divide', 'build', 'end', 'accentuate',
        'demonstrate', 'solve', 'consume', 'position', 'swear', 'furnish',
        'merit', 'submit', 'seize', 'spin', 'enlarge', 'swing', 'circulate',
        'recognize', 'resolve', 'prolong', 'tap', 'block', 'arrange', 'twist',
        'hail', 'dissipate', 'approve', 'impose', 'hasten', 'rap', 'lean',
        'make', 'show', 'sell', 'weave', 'refer', 'distribute', 'twist',
        'drain', 'depict', 'build', 'hail', 'call', 'swing', 'yield', 'split',
        'challenge', 'hinder', 'welcome', 'need', 'refer', 'finance', 'expect',
        'terminate', 'yell', 'swell', 'rotate', 'seize', 'approve', 'supply',
        'clip', 'divide', 'advise', 'complain', 'want', 'twist', 'swing',
        'make', 'hinder', 'build', 'express', 'resolve', 'bruise', 'swing',
        'catch', 'swear', 'request', 'arrange', 'relieve', 'move', 'weave',
        'swear', 'forget', 'supervise', 'situate', 'explain', 'ache',
        'evaluate', 'recognize', 'dilute', 'hasten', 'scorn', 'swear',
        'arrange', 'discard', 'list', 'stamp', 'market', 'boil', 'sustain',
        'resolve', 'dissipate', 'anger', 'approve', 'research', 'request',
        'boast', 'furnish', 'refine', 'acknowledge', 'clean', 'lean',
        'postpone', 'hail', 'remember', 'scrape', 'sweat', 'highlight',
        'seize', 'levy', 'alter', 'refer', 'empty', 'flush', 'shake',
        'imitate', 'correlate', 'refer'
    ]
    _words2 = [
        'boast', 'devise', 'split', 'construct', 'terminate', 'highlight',
        'show', 'figure', 'eat', 'situate', 'vow', 'supply', 'deserve',
        'yield', 'take', 'twirl', 'swell', 'sway', 'distribute', 'acknowledge',
        'settle', 'sustain', 'knock', 'hinder', 'plan', 'curl', 'acclaim',
        'disperse', 'support', 'levy', 'accelerate', 'tap', 'rest', 'earn',
        'publish', 'market', 'intertwine', 'direct', 'commercialize',
        'intertwine', 'tap', 'recognize', 'organize', 'address', 'refer',
        'bounce', 'seize', 'crush', 'yield', 'assist', 'recognize', 'deserve',
        'explain', 'build', 'deserve', 'postpone', 'boast', 'curl', 'situate',
        'request', 'scorn', 'consume', 'twist', 'figure', 'furnish', 'boast',
        'deserve', 'fasten', 'crash', 'trade', 'yield', 'propose', 'figure',
        'examine', 'split', 'break', 'consume', 'explain', 'levy', 'study',
        'hinder', 'swell', 'print', 'think', 'resolve', 'concoct', 'isolate',
        'boast', 'spin', 'terminate', 'succeed', 'market', 'permit', 'yield',
        'describe', 'explain', 'arrange', 'figure', 'weave', 'sweeten', 'tap',
        'lower', 'publicize', 'isolate', 'approve', 'boast', 'distribute',
        'concoct', 'yield', 'impress', 'sustain', 'distribute', 'concoct',
        'grate', 'show', 'judge', 'hail', 'lean', 'spin', 'restore', 'refer',
        'believe', 'highlight', 'carry', 'situate', 'spin', 'swell',
        'highlight', 'levy', 'lean'
    ]

    _url = ('https://dspace2.flinders.edu.au/xmlui/bitstream/handle/'
            '2328/9557/Yang%20Verb.pdf?sequence=1')

    min = 0
    max = 4

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets',
                                         'verb130')):
        super(YangPowersVerb130, self).__init__(root=root)
        df = pd.DataFrame.from_records(
            dict(word1=self._words1, word2=self._words2))
        df['score'] = 0
        for i, score in enumerate(range(4, 0, -1)):
            df['score'].iloc[i * 26:(i + 1) * 26] = score
        self._data = df

    def _get_data(self):
        pass


###############################################################################
# Word analogy datasets
###############################################################################
class _WordAnalogyEvaluationDataset(_Dataset):
    word1 = 'word1'
    word2 = 'word2'
    word3 = 'word3'
    word4 = 'word4'

    def __getitem__(self, idx):
        row = self._data.iloc[idx]
        return row[[self.word1, self.word2, self.word3, self.word4]].values


class GoogleAnalogyTestSet(_WordAnalogyEvaluationDataset):
    """Google analogy test set

    - Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient
      estimation of word representations in vector space. In Proceedings of
      the International Conference on Learning Representations (ICLR).

    License: Unspecified

    """

    _archive_file = ('questions-words.txt',
                     'fa92df4bbe788f2d51827c762c63bd8e470edf31')
    _checksums = {
        'questions-words.txt': 'fa92df4bbe788f2d51827c762c63bd8e470edf31'
    }
    _url = 'http://download.tensorflow.org/data/questions-words.txt'

    def __init__(self,
                 root=os.path.join('~', '.mxnet', 'datasets',
                                   'google_analogy')):
        super(GoogleAnalogyTestSet, self).__init__(root=root)

        words = []
        with open(os.path.join(self.root, self._archive_file[0])) as f:
            for line in f:
                if line.startswith(':'):
                    category = line.split()[1]
                    if 'gram' in category:
                        group = 'syntactic'
                    else:
                        group = 'semantic'
                else:
                    words.append(line.split() + [group, category])

        df = pd.DataFrame(
            words,
            columns=['word1', 'word2', 'word3', 'word4', 'group', 'category'])

        self._data = df


class BiggerAnalogyTestSet(_WordAnalogyEvaluationDataset):
    """SimVerb3500 dataset word-similarity.

    - Hill, F., Reichart, R., & Korhonen, A. (2015). Simlex-999: evaluating
      semantic models with (genuine) similarity estimation. Computational
      Linguistics, 41(4), 665–695. http://dx.doi.org/10.1162/COLI_a_00237

    License: Unspecified

    The dataset contains

    - word1: The first verb of the pair.
    - word2: The second verb of the pair.
    - POS: The part-of-speech tag. Note that it is 'V' for all pairs, since the
      dataset exclusively contains verbs. We decided to include it nevertheless
      to make it compatible with SimLex-999.
    - score: The SimVerb-3500 similarity rating. Note that average annotator
      scores have been linearly mapped from the range [0,6] to the range [0,10]
      to match other datasets.
    - relation: the lexical relation of the pair. Possible values: 'SYNONYMS',
      'ANTONYMS', 'HYPER/HYPONYMS', 'COHYPONYMS', 'NONE'.

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/simverb3500'
        Path to temp folder for storing data.

    """
    _archive_file = ('BATS_3.0.zip',
                     'bf94d47884be9ea83af369beeea7499ed25dcf0d')
    _checksums = {
        'BATS_3.0/1_Inflectional_morphology/I01 [noun - plural_reg].txt':
        'cfcba2835edf81abf11b84defd2f4daa3ca0b0bf',
        'BATS_3.0/1_Inflectional_morphology/I02 [noun - plural_irreg].txt':
        '44dbc56432b79ff5ce2ef80b6840a8aa916524f9',
        'BATS_3.0/1_Inflectional_morphology/I03 [adj - comparative].txt':
        'dc530918e98b467b8102a7dab772a66d3db32a73',
        'BATS_3.0/1_Inflectional_morphology/I04 [adj - superlative].txt':
        '6c6fdfb6c733bc9b298d95013765163f42faf6fb',
        'BATS_3.0/1_Inflectional_morphology/I05 [verb_inf - 3pSg].txt':
        '39fa47ec7238ddb3f9818bc586f23f55b55418d8',
        'BATS_3.0/1_Inflectional_morphology/I06 [verb_inf - Ving].txt':
        '8fabeb9f5af6c3e7154a220b7034bbe5b900c36f',
        'BATS_3.0/1_Inflectional_morphology/I07 [verb_inf - Ved].txt':
        'aa04df95aa2edb436cbcc03c7b15bc492ece52d6',
        'BATS_3.0/1_Inflectional_morphology/I08 [verb_Ving - 3pSg].txt':
        '5f22d8121a5043ce76d3b6b53a49a7bb3fe33920',
        'BATS_3.0/1_Inflectional_morphology/I09 [verb_Ving - Ved].txt':
        '377777c1e793c638e72c010228156d01f916708e',
        'BATS_3.0/1_Inflectional_morphology/I10 [verb_3pSg - Ved].txt':
        '051c0c3c633e10900f827991dac14cf76da7f022',
        'BATS_3.0/2_Derivational_morphology/D01 [noun+less_reg].txt':
        '5d6839e9d34ee1e9fddb5bbf6516cf6420b85d8d',
        'BATS_3.0/2_Derivational_morphology/D02 [un+adj_reg].txt':
        '80b82227a0d5f7377f1e8cebe28c582bfeb1afb5',
        'BATS_3.0/2_Derivational_morphology/D03 [adj+ly_reg].txt':
        '223e120bd61b3116298a253f392654c15ad5a39a',
        'BATS_3.0/2_Derivational_morphology/D04 [over+adj_reg].txt':
        'a56f8685af489bcd09c36f864eba1657ce0a7c28',
        'BATS_3.0/2_Derivational_morphology/D05 [adj+ness_reg].txt':
        '5da99b1f1781ecfb4a1a7448c715abf07451917b',
        'BATS_3.0/2_Derivational_morphology/D06 [re+verb_reg].txt':
        '4c5e1796091fade503fbf0bfc2fae2c7f98b5dd2',
        'BATS_3.0/2_Derivational_morphology/D07 [verb+able_reg].txt':
        'a6218162bc257d98e875fc667c23edfac59e19fd',
        'BATS_3.0/2_Derivational_morphology/D08 [verb+er_irreg].txt':
        '9a4236c3bbc23903e101a42fb5ad6e15e552fadf',
        'BATS_3.0/2_Derivational_morphology/D09 [verb+tion_irreg].txt':
        '3ab0153926d5cf890cf08a4077da6d9946133874',
        'BATS_3.0/2_Derivational_morphology/D10 [verb+ment_irreg].txt':
        '2a012b87a9a60e128e064c5fe24b60f99e16ddce',
        'BATS_3.0/3_Encyclopedic_semantics/E01 [country - capital].txt':
        '9890315d3c4e6a38b8ae5fc441858564be3d3dc4',
        'BATS_3.0/3_Encyclopedic_semantics/E02 [country - language].txt':
        'ef08a00e8ff7802811ace8f00fabac41b5d03678',
        'BATS_3.0/3_Encyclopedic_semantics/E03 [UK_city - county].txt':
        '754957101c93a25b438785bd4458404cd9010259',
        'BATS_3.0/3_Encyclopedic_semantics/E04 [name - nationality].txt':
        '71a6562c34fb6154992a7c3e499375fcc3529c96',
        'BATS_3.0/3_Encyclopedic_semantics/E05 [name - occupation].txt':
        'a9a6f9f1af959aef83106f3dbd6bed16dfe9a3ea',
        'BATS_3.0/3_Encyclopedic_semantics/E06 [animal - young].txt':
        '12d5b51c7b76b9136eadc719abc8cf4806c67b73',
        'BATS_3.0/3_Encyclopedic_semantics/E07 [animal - sound].txt':
        '91991b007a35f45bd42bd7d0d465c6f8311df911',
        'BATS_3.0/3_Encyclopedic_semantics/E08 [animal - shelter].txt':
        'e5af11e216db392986ba0cbb597d861066c29adb',
        'BATS_3.0/3_Encyclopedic_semantics/E09 [things - color].txt':
        'd30b2eb2fc7a60f19afda7c54582e30f6fe28f51',
        'BATS_3.0/3_Encyclopedic_semantics/E10 [male - female].txt':
        '247a588671bc1da8f615e14076bd42573d24b4b3',
        'BATS_3.0/4_Lexicographic_semantics/L01 [hypernyms - animals].txt':
        '4b5c4dabe2c9c038fafee85d8d3958f1b1dec987',
        'BATS_3.0/4_Lexicographic_semantics/L02 [hypernyms - misc].txt':
        '83d5ecad78d9de28fd70347731c7ee5918ba43c9',
        'BATS_3.0/4_Lexicographic_semantics/L03 [hyponyms - misc].txt':
        'a8319856ae2f76b4d4c030ac7e899bb3a06a9a48',
        'BATS_3.0/4_Lexicographic_semantics/L04 [meronyms - substance].txt':
        'c081e1104e1b40725063f4b39d13d1ec12496bfd',
        'BATS_3.0/4_Lexicographic_semantics/L05 [meronyms - member].txt':
        'bcbf05f3be76cef990a74674a9999a0bb9790a07',
        'BATS_3.0/4_Lexicographic_semantics/L06 [meronyms - part].txt':
        '2f9bdcc74b881e1c54b391c9a6e7ea6243b3accc',
        'BATS_3.0/4_Lexicographic_semantics/L07 [synonyms - intensity].txt':
        '8fa287860b096bef004fe0f6557e4f686e3da81a',
        'BATS_3.0/4_Lexicographic_semantics/L08 [synonyms - exact].txt':
        'a17c591961bddefd97ae5df71f9d1559ce7900f4',
        'BATS_3.0/4_Lexicographic_semantics/L09 [antonyms - gradable].txt':
        '117fbb86504c192b33a5469f2f282e741d9c016d',
        'BATS_3.0/4_Lexicographic_semantics/L10 [antonyms - binary].txt':
        '3cde2f2c2a0606777b8d7d11d099f316416a7224'
    }
    _url = 'https://s3.amazonaws.com/blackbirdprojects/tut_vsm/BATS_3.0.zip'
    _category_group_map = {
        'I': '1_Inflectional_morphology',
        'D': '2_Derivational_morphology',
        'E': '3_Encyclopedic_semantics',
        'L': '4_Lexicographic_semantics'
    }
    _categories = {
        'I01': '[noun - plural_reg]',
        'I02': '[noun - plural_irreg]',
        'I03': '[adj - comparative]',
        'I04': '[adj - superlative]',
        'I05': '[verb_inf - 3pSg]',
        'I06': '[verb_inf - Ving]',
        'I07': '[verb_inf - Ved]',
        'I08': '[verb_Ving - 3pSg]',
        'I09': '[verb_Ving - Ved]',
        'I10': '[verb_3pSg - Ved]',
        'D01': '[noun+less_reg]',
        'D02': '[un+adj_reg]',
        'D03': '[adj+ly_reg]',
        'D04': '[over+adj_reg]',
        'D05': '[adj+ness_reg]',
        'D06': '[re+verb_reg]',
        'D07': '[verb+able_reg]',
        'D08': '[verb+er_irreg]',
        'D09': '[verb+tion_irreg]',
        'D10': '[verb+ment_irreg]',
        'E01': '[country - capital]',
        'E02': '[country - language]',
        'E03': '[UK_city - county]',
        'E04': '[name - nationality]',
        'E05': '[name - occupation]',
        'E06': '[animal - young]',
        'E07': '[animal - sound]',
        'E08': '[animal - shelter]',
        'E09': '[things - color]',
        'E10': '[male - female]',
        'L01': '[hypernyms - animals]',
        'L02': '[hypernyms - misc]',
        'L03': '[hyponyms - misc]',
        'L04': '[meronyms - substance]',
        'L05': '[meronyms - member]',
        'L06': '[meronyms - part]',
        'L07': '[synonyms - intensity]',
        'L08': '[synonyms - exact]',
        'L09': '[antonyms - gradable]',
        'L10': '[antonyms - binary]'
    }

    def __init__(self,
                 form_analogy_pairs=True,
                 drop_alternative_solutions=True,
                 root=os.path.join('~', '.mxnet', 'datasets', 'simverb3500')):
        super(BiggerAnalogyTestSet, self).__init__(root=root)

        dfs = []
        for category in self._categories:
            group = self._category_group_map[category[0]]
            category_name = self._categories[category]
            path = os.path.join(
                self.root,
                'BATS_3.0/{group}/{category} {category_name}.txt'.format(
                    group=group,
                    category=category,
                    category_name=category_name))
            df = pd.read_table(
                path, header=None, delimiter='\t', names=('word1', 'word2'))
            df['word2'] = df['word2'].str.split('/')
            # Drop alternative solutions seperated by '/' from word2 column
            if drop_alternative_solutions:
                df['word2'] = df['word2'].str[0]

            # Final dataset consists of all analogy pairs per category
            if form_analogy_pairs:
                df['_joinc'] = 1
                df = pd.merge(df.reset_index(), df.reset_index(), on='_joinc')
                df = df[df.index_x !=
                        df.index_y]  # Drop A : B = A : B analogies
                df = df[['word1_x', 'word2_x', 'word1_y', 'word2_y']]
                df.columns = ['word1', 'word2', 'word3', 'word4']

            df['group'] = group
            df['category'] = category
            df['category_name'] = category_name
            dfs.append(df)

        df = pd.concat(dfs)
        self._data = df
