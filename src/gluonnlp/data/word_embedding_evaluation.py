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

from mxnet.gluon.data.dataset import SimpleDataset
from mxnet.gluon.utils import check_sha1, _get_repo_file_url, download

from .. import _constants as C
from .dataset import CorpusDataset
from .registry import register
from .utils import _get_home_dir

base_datasets = [
    'WordSimilarityEvaluationDataset', 'WordAnalogyEvaluationDataset'
]
word_similarity_datasets = [
    'WordSim353', 'MEN', 'RadinskyMTurk', 'RareWords', 'SimLex999',
    'SimVerb3500', 'SemEval17Task2', 'BakerVerb143', 'YangPowersVerb130'
]
word_analogy_datasets = ['GoogleAnalogyTestSet', 'BiggerAnalogyTestSet']
__all__ = base_datasets + word_similarity_datasets + word_analogy_datasets


class _Dataset(SimpleDataset):
    _url = None  # Dataset is retrieved from here if not cached
    _archive_file = (None, None)  # Archive name and checksum
    _checksums = None  # Checksum of archive contents
    _verify_ssl = True  # Verify SSL certificates when downloading from self._url
    _namespace = None  # Contains S3 namespace for self-hosted datasets

    def __init__(self, root):
        self.root = os.path.expanduser(root)
        if not os.path.isdir(self.root):
            os.makedirs(self.root)
        self._download_data()
        super(_Dataset, self).__init__(self._get_data())

    def _download_data(self):
        _, archive_hash = self._archive_file
        for name, checksum in self._checksums.items():
            name = name.split('/')
            path = os.path.join(self.root, *name)
            if not os.path.exists(path) or not check_sha1(path, checksum):
                if self._namespace is not None:
                    url = _get_repo_file_url(self._namespace,
                                             self._archive_file[0])
                else:
                    url = self._url
                downloaded_file_path = download(url, path=self.root,
                                                sha1_hash=archive_hash,
                                                verify_ssl=self._verify_ssl)

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

    def _get_data(self):
        raise NotImplementedError


###############################################################################
# Word similarity and relatedness datasets
###############################################################################
class WordSimilarityEvaluationDataset(_Dataset):
    """Base class for word similarity or relatedness task datasets.

    Inheriting classes are assumed to implement datasets of the form ['word1',
    'word2', score] where score is a numerical similarity or relatedness score
    with respect to 'word1' and 'word2'.

    """

    def __init__(self, root):
        super(WordSimilarityEvaluationDataset, self).__init__(root=root)
        self._cast_score_to_float()

    def _get_data(self):
        raise NotImplementedError

    def _cast_score_to_float(self):
        self._data = [[row[0], row[1], float(row[2])] for row in self._data]


@register(segment=['all', 'similarity', 'relatedness'])
class WordSim353(WordSimilarityEvaluationDataset):
    """WordSim353 dataset.

    The dataset was collected by Finkelstein et al.
    (http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/). Agirre et
    al. proposed to split the collection into two datasets, one focused on
    measuring similarity, and the other one on relatedness
    (http://alfonseca.org/eng/research/wordsim353.html).

    - Finkelstein, L., Gabrilovich, E., Matias, Y., Rivlin, E., Solan, Z.,
      Wolfman, G., & Ruppin, E. (2002). Placing search in context: the concept
      revisited. ACM} Trans. Inf. Syst., 20(1), 116–131.
      https://dl.acm.org/citation.cfm?id=372094
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
    root : str, default '$MXNET_HOME/datasets/wordsim353'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
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

    def __init__(self, segment='all', root=os.path.join(
            _get_home_dir(), 'datasets', 'wordsim353')):
        if segment is not None:
            assert segment in ['all', 'relatedness', 'similarity']

        self.segment = segment
        super(WordSim353, self).__init__(root=root)

    def _get_data(self):
        paths = []
        if self.segment == 'relatedness' or self.segment == 'all':
            paths.append(
                os.path.join(
                    self.root,
                    'wordsim353_sim_rel/wordsim_relatedness_goldstandard.txt'))
        if self.segment == 'similarity' or self.segment == 'all':
            paths.append(
                os.path.join(
                    self.root,
                    'wordsim353_sim_rel/wordsim_similarity_goldstandard.txt'))

        return list({tuple(row) for row in CorpusDataset(paths)})


@register(segment=['full', 'dev', 'test'])
class MEN(WordSimilarityEvaluationDataset):
    """MEN dataset for word-similarity and relatedness.

    The dataset was collected by Bruni et al.
    (https://staff.fnwi.uva.nl/e.bruni/MEN).

    - Bruni, E., Boleda, G., Baroni, M., & Nam-Khanh Tran (2012). Distributional
      semantics in technicolor. In , The 50th Annual Meeting of the Association for
      Computational Linguistics, Proceedings of the Conference, July 8-14, 2012,
      Jeju Island, Korea - Volume 1: Long Papers (pp. 136–145). : The Association
      for Computer Linguistics.

    License: Creative Commons Attribution 2.0 Generic (CC BY 2.0)

    Parameters
    ----------
    root : str, default '$MXNET_HOME/datasets/men'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    segment : str, default 'train'
        Dataset segment. Options are 'train', 'dev', 'test'.

    """
    _url = 'https://staff.fnwi.uva.nl/e.bruni/resources/MEN.tar.gz'
    _namespace = 'gluon/dataset/men'
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

    def __init__(self, segment='dev', root=os.path.join(
            _get_home_dir(), 'datasets', 'men')):
        self.segment = segment
        super(MEN, self).__init__(root=root)

    def _get_data(self):
        datafilepath = os.path.join(
            self.root, *self._segment_file[self.segment].split('/'))
        dataset = CorpusDataset(datafilepath)

        # Remove lemma information
        return [[row[0][:-2], row[1][:-2], row[2]] for row in dataset]


@register
class RadinskyMTurk(WordSimilarityEvaluationDataset):
    """MTurk dataset for word-similarity and relatedness by Radinsky et al..

    - Radinsky, K., Agichtein, E., Gabrilovich, E., & Markovitch, S. (2011). A word
      at a time: computing word relatedness using temporal semantic analysis. In S.
      Srinivasan, K. Ramamritham, A. Kumar, M. P. Ravindra, E. Bertino, & R. Kumar,
      Proceedings of the 20th International Conference on World Wide Web, {WWW}
      2011, Hyderabad, India, March 28 - April 1, 2011 (pp. 337–346). : ACM.

    License: Unspecified

    Parameters
    ----------
    root : str, default '$MXNET_HOME/datasets/radinskymturk'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

    """
    _url = 'http://www.kiraradinsky.com/files/Mtruk.csv'
    _archive_file = ('Mtruk.csv', '14959899c092148abba21401950d6957c787434c')
    _checksums = {'Mtruk.csv': '14959899c092148abba21401950d6957c787434c'}

    min = 1
    max = 5

    def __init__(self, root=os.path.join(_get_home_dir(), 'datasets',
                                         'radinskymturk')):
        super(RadinskyMTurk, self).__init__(root=root)

    def _get_data(self):
        datafilepath = os.path.join(self.root, self._archive_file[0])

        dataset = CorpusDataset(datafilepath, tokenizer=lambda x: x.split(','))
        return [row for row in dataset]


@register
class RareWords(WordSimilarityEvaluationDataset):
    """Rare words dataset word-similarity and relatedness.

    - Luong, T., Socher, R., & Manning, C. D. (2013). Better word representations
      with recursive neural networks for morphology. In J. Hockenmaier, & S.
      Riedel, Proceedings of the Seventeenth Conference on Computational Natural
      Language Learning, CoNLL 2013, Sofia, Bulgaria, August 8-9, 2013 (pp.
      104–113). : ACL.

    License: Unspecified

    Parameters
    ----------
    root : str, default '$MXNET_HOME/datasets/rarewords',
        MXNET_HOME defaults to '~/.mxnet'.
        Path to temp folder for storing data.

    """
    _url = 'http://www-nlp.stanford.edu/~lmthang/morphoNLM/rw.zip'
    _archive_file = ('rw.zip', 'bf9c5959a0a2d7ed8e51d91433ac5ebf366d4fb9')
    _checksums = {'rw/rw.txt': 'bafc59f099f1798b47f5bed7b0ebbb933f6b309a'}

    min = 0
    max = 10

    def __init__(self, root=os.path.join(_get_home_dir(), 'datasets',
                                         'rarewords')):
        super(RareWords, self).__init__(root=root)

    def _get_data(self):
        datafilepath = os.path.join(self.root, 'rw', 'rw.txt')
        dataset = CorpusDataset(datafilepath)
        return [[row[0], row[1], row[2]] for row in dataset]


@register
class SimLex999(WordSimilarityEvaluationDataset):
    """SimLex999 dataset word-similarity.

    - Hill, F., Reichart, R., & Korhonen, A. (2015). Simlex-999: evaluating
      semantic models with (genuine) similarity estimation. Computational
      Linguistics, 41(4), 665–695. https://arxiv.org/abs/1408.3456

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
    root : str, default '$MXNET_HOME/datasets/simlex999'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

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

    def __init__(self, root=os.path.join(_get_home_dir(), 'datasets',
                                         'simlex999')):
        super(SimLex999, self).__init__(root=root)

    def _get_data(self):
        dataset = CorpusDataset(
            os.path.join(self.root, 'SimLex-999', 'SimLex-999.txt'))
        return [[row[0], row[1], row[3]] for i, row in enumerate(dataset)
                if i != 0]  # Throw away header


@register
class SimVerb3500(WordSimilarityEvaluationDataset):
    """SimVerb3500 dataset word-similarity.

    - Hill, F., Reichart, R., & Korhonen, A. (2015). Simlex-999: evaluating
      semantic models with (genuine) similarity estimation. Computational
      Linguistics, 41(4), 665–695. https://arxiv.org/abs/1408.3456

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
    root : str, default '$MXNET_HOME/datasets/verb3500'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

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

    def __init__(self, segment='full', root=os.path.join(
            _get_home_dir(), 'datasets', 'simverb3500')):
        self.segment = segment
        super(SimVerb3500, self).__init__(root=root)

    def _get_data(self):
        dataset = CorpusDataset(
            os.path.join(self.root,
                         *self._segment_file[self.segment].split('/')))
        return [[row[0], row[1], row[3]] for row in dataset]


@register(segment=['trial', 'test'])
class SemEval17Task2(WordSimilarityEvaluationDataset):
    """SemEval17Task2 dataset for word-similarity.

    The dataset was collected by Finkelstein et al.
    (http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/). Agirre et
    al. proposed to split the collection into two datasets, one focused on
    measuring similarity, and the other one on relatedness
    (http://alfonseca.org/eng/research/wordsim353.html).

    - Finkelstein, L., Gabrilovich, E., Matias, Y., Rivlin, E., Solan, Z.,
      Wolfman, G., & Ruppin, E. (2002). Placing search in context: the concept
      revisited. ACM} Trans. Inf. Syst., 20(1), 116–131.
      https://dl.acm.org/citation.cfm?id=372094
    - Agirre, E., Alfonseca, E., Hall, K. B., Kravalova, J., Pasca, M., & Soroa, A.
      (2009). A study on similarity and relatedness using distributional and
      wordnet-based approaches. In , Human Language Technologies: Conference of the
      North American Chapter of the Association of Computational Linguistics,
      Proceedings, May 31 - June 5, 2009, Boulder, Colorado, {USA (pp. 19–27). :
      The Association for Computational Linguistics.

    License: Unspecified

    Parameters
    ----------
    root : str, default '$MXNET_HOME/datasets/semeval17task2'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.
    segment : str, default 'train'
        Dataset segment. Options are 'trial', 'test'.
    language : str, default 'en'
        Dataset language.

    """
    _url = 'http://alt.qcri.org/semeval2017/task2/data/uploads/semeval2017-task2.zip'
    _archive_file = ('semeval2017-task2.zip',
                     'b29860553f98b057303815817dfb60b9fe79cfba')
    _checksums = C.SEMEVAL17_CHECKSUMS

    _datatemplate = ('SemEval17-Task2/{segment}/subtask1-monolingual/data/'
                     '{language}.{segment}.data.txt')
    _keytemplate = ('SemEval17-Task2/{segment}/subtask1-monolingual/keys/'
                    '{language}.{segment}.gold.txt')

    min = 0
    max = 5
    segments = ('trial', 'test')
    languages = ('en', 'es', 'de', 'it', 'fa')

    def __init__(self, segment='trial', language='en', root=os.path.join(
            _get_home_dir(), 'datasets', 'semeval17task2')):
        assert segment in self.segments
        assert language in self.languages
        self.language = language
        self.segment = segment
        super(SemEval17Task2, self).__init__(root=root)

    def _get_data(self):
        data = self._datatemplate.format(segment=self.segment,
                                         language=self.language)
        data = os.path.join(self.root, *data.split('/'))
        keys = self._keytemplate.format(segment=self.segment,
                                        language=self.language)
        keys = os.path.join(self.root, *keys.split('/'))

        data_dataset = CorpusDataset(data)
        keys_dataset = CorpusDataset(keys)
        return [[d[0], d[1], k[0]] for d, k in zip(data_dataset, keys_dataset)]


@register
class BakerVerb143(WordSimilarityEvaluationDataset):
    """Verb143 dataset.

    - Baker, S., Reichart, R., & Korhonen, A. (2014). An unsupervised model for
      instance level subcategorization acquisition. In A. Moschitti, B. Pang, &
      W. Daelemans, Proceedings of the 2014 Conference on Empirical Methods in
      Natural Language Processing, {EMNLP} 2014, October 25-29, 2014, Doha,
      Qatar, {A} meeting of SIGDAT, a Special Interest Group of the {ACL (pp.
      278–289). : ACL.

    144 pairs of verbs annotated by 10 annotators following the WS-353
    guidelines.

    License: unspecified

    Parameters
    ----------
    root : str, default '$MXNET_HOME/datasets/verb143'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

    """
    _url = 'https://ie.technion.ac.il/~roiri/papers/EMNLP14.zip'
    _archive_file = ('EMNLP14.zip', '1862e52af784e76e83d472532a75eb797fb8b807')
    _checksums = {
        'verb_similarity dataset.txt':
        'd7e4820c7504cbae56898353e4d94e6408c330fc'
    }
    _verify_ssl = False  # ie.technion.ac.il serves an invalid cert as of 2018-04-16

    min = 0
    max = 10

    def __init__(self, root=os.path.join(_get_home_dir(), 'datasets',
                                         'verb143')):
        super(BakerVerb143, self).__init__(root=root)

    def _get_data(self):
        path = os.path.join(self.root, 'verb_similarity dataset.txt')

        dataset = CorpusDataset(path)
        return [[row[0], row[1], row[12]] for row in dataset]


@register
class YangPowersVerb130(WordSimilarityEvaluationDataset):
    """Verb-130 dataset.

    - Yang, D., & Powers, D. M. (2006). Verb similarity on the taxonomy of
      wordnet. In The Third International WordNet Conference: GWC 2006

    License: Unspecified

    Parameters
    ----------
    root : str, default '$MXNET_HOME/datasets/verb130'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

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

    def _get_data(self):
        scores = [4] * 26 + [3] * 26 + [2] * 26 + [1] * 26 + [0] * 26
        return list(zip(self._words1, self._words2, scores))

    def _download_data(self):
        # Overwrite download method as this dataset is self-contained
        pass


###############################################################################
# Word analogy datasets
###############################################################################
class WordAnalogyEvaluationDataset(_Dataset):
    """Base class for word analogy task datasets.

    Inheriting classes are assumed to implement datasets of the form ['word1',
    'word2', 'word3', 'word4'] or ['word1', [ 'word2a', 'word2b', ... ],
    'word3', [ 'word4a', 'word4b', ... ]].

    """

    def _get_data(self):
        raise NotImplementedError


@register(category=C.GOOGLEANALOGY_CATEGORIES)
class GoogleAnalogyTestSet(WordAnalogyEvaluationDataset):
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

    groups = ['syntactic', 'semantic']
    categories = C.GOOGLEANALOGY_CATEGORIES

    def __init__(self, group=None,
                 category=None, lowercase=True, root=os.path.join(
                     _get_home_dir(), 'datasets', 'google_analogy')):

        assert group is None or group in self.groups
        assert category is None or category in self.categories
        self.category = category
        self.group = group
        self.lowercase = lowercase
        super(GoogleAnalogyTestSet, self).__init__(root=root)

    def _get_data(self):
        words = []
        with open(os.path.join(self.root, self._archive_file[0])) as f:
            for line in f:
                if line.startswith(':'):
                    current_category = line.split()[1]
                    if 'gram' in current_category:
                        current_group = 'syntactic'
                    else:
                        current_group = 'semantic'
                else:
                    if self.group is not None and self.group != current_group:
                        continue
                    if self.category is not None and self.category != current_category:
                        continue

                    if self.lowercase:
                        line = line.lower()

                    words.append(line.split())

        return words


@register(category=list(C.BATS_CATEGORIES.keys()))
class BiggerAnalogyTestSet(WordAnalogyEvaluationDataset):
    """Bigger analogy test set

    - Gladkova, A., Drozd, A., & Matsuoka, S. (2016). Analogy-based detection
      of morphological and semantic relations with word embeddings: what works
      and what doesn’t. In Proceedings of the NAACL-HLT SRW (pp. 47–54). San
      Diego, California, June 12-17, 2016: ACL. Retrieved from
      https://www.aclweb.org/anthology/N/N16/N16-2002.pdf

    License: Unspecified

    Parameters
    ----------
    root : str, default '$MXNET_HOME/datasets/bats'
        Path to temp folder for storing data.
        MXNET_HOME defaults to '~/.mxnet'.

    """
    _archive_file = ('BATS_3.0.zip',
                     'bf94d47884be9ea83af369beeea7499ed25dcf0d')
    _checksums = C.BATS_CHECKSUMS
    _url = 'https://s3.amazonaws.com/blackbirdprojects/tut_vsm/BATS_3.0.zip'
    _category_group_map = {
        'I': '1_Inflectional_morphology',
        'D': '2_Derivational_morphology',
        'E': '3_Encyclopedic_semantics',
        'L': '4_Lexicographic_semantics'
    }
    _categories = C.BATS_CATEGORIES

    def __init__(self, category=None, form_analogy_pairs=True,
                 drop_alternative_solutions=True, root=os.path.join(
                     _get_home_dir(), 'datasets', 'bigger_analogy')):
        self.form_analogy_pairs = form_analogy_pairs
        self.drop_alternative_solutions = drop_alternative_solutions
        self.category = category

        if self.category is not None:
            assert self.category in self._categories.keys()

        super(BiggerAnalogyTestSet, self).__init__(root=root)

    def _get_data(self):
        if self.category is not None:
            categories = [self.category]
        else:
            categories = self._categories.keys()

        datasets = []
        for category in categories:
            group = self._category_group_map[category[0]]
            category_name = self._categories[category]
            path = os.path.join(
                self.root,
                *('BATS_3.0/{group}/{category} {category_name}.txt'.format(
                    group=group, category=category,
                    category_name=category_name).split('/')))
            dataset = CorpusDataset(path)
            dataset = [[row[0], row[1].split('/')] for row in dataset]
            # Drop alternative solutions seperated by '/' from word2 column
            if self.drop_alternative_solutions:
                dataset = [[row[0], row[1][0]] for row in dataset]

            # Final dataset consists of all analogy pairs per category
            if self.form_analogy_pairs:
                dataset = [[arow[0], arow[1], brow[0], brow[1]]
                           for arow in dataset for brow in dataset
                           if arow != brow]
            datasets += dataset
        return datasets
