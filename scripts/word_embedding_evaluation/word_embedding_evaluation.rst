Word Embedding Evaluation
-------------------------

This script can be used to evaluate pretrained word embeddings included in the
gluon NLP toolkit.

The download link below contains a notebook with extended results comparing the
different included pretrained embeddings on all Word Embedding Evaluation
datasets included in the toolkit, providing detailed information per category in
the respective datasets.

We include a `run_all.sh` script to reproduce the results.

To evaluate a specific embedding on one or multiple datasets you can use the
included `word_embedding_evaluation.py` as follows:

.. code-block:: bash

   $ python word_embedding_evaluation.py --help

    usage: word_embedding_evaluation.py [-h] [--embedding-name EMBEDDING_NAME]
                                        [--embedding-source EMBEDDING_SOURCE]
                                        [--list-embedding-sources] [--ignore-oov]
                                        [--similarity-datasets [SIMILARITY_DATASETS [SIMILARITY_DATASETS ...]]]
                                        [--similarity-functions SIMILARITY_FUNCTIONS [SIMILARITY_FUNCTIONS ...]]
                                        [--analogy-datasets [ANALOGY_DATASETS [ANALOGY_DATASETS ...]]]
                                        [--analogy-functions ANALOGY_FUNCTIONS [ANALOGY_FUNCTIONS ...]]
                                        [--analogy-dont-exclude-question-words]
                                        [--analogy-max-vocab ANALOGY_MAX_VOCAB]
                                        [--batch-size BATCH_SIZE] [--gpu GPU]
                                        [--dont-hybridize] [--log LOG]
    
    Word embedding training with Gluon.
    
    optional arguments:
      -h, --help            show this help message and exit
    
    Embedding arguments:
      --embedding-name EMBEDDING_NAME
                            Name of embedding type to load. Valid entries: glove,
                            fasttext (default: fasttext)
      --embedding-source EMBEDDING_SOURCE
                            Source from which to initialize the embedding.Pass
                            --list-embedding-sources to get a list of valid
                            sources for a given --embedding-name. (default:
                            wiki.simple)
      --list-embedding-sources
    
    Evaluation arguments:
      --ignore-oov          Drop OOV words from evaluation datasets. (default:
                            False)
      --similarity-datasets [SIMILARITY_DATASETS [SIMILARITY_DATASETS ...]]
                            Word similarity datasets to use for intrinsic
                            evaluation. (default: ['WordSim353', 'MEN',
                            'RadinskyMTurk', 'RareWords', 'SimLex999',
                            'SimVerb3500', 'SemEval17Task2', 'BakerVerb143',
                            'YangPowersVerb130'])
      --similarity-functions SIMILARITY_FUNCTIONS [SIMILARITY_FUNCTIONS ...]
                            Word similarity functions to use for intrinsic
                            evaluation. (default: ['cosinesimilarity'])
      --analogy-datasets [ANALOGY_DATASETS [ANALOGY_DATASETS ...]]
                            Word similarity datasets to use for intrinsic
                            evaluation. (default: ['GoogleAnalogyTestSet',
                            'BiggerAnalogyTestSet'])
      --analogy-functions ANALOGY_FUNCTIONS [ANALOGY_FUNCTIONS ...]
                            Word analogy functions to use for intrinsic
                            evaluation. (default: ['threecosmul', 'threecosadd'])
      --analogy-dont-exclude-question-words
                            Exclude input words from valid output analogies.The
                            performance of word embeddings on the analogy task is
                            around 0{'option_strings': ['--analogy-dont-exclude-
                            question-words'], 'dest':
                            'analogy_dont_exclude_question_words', 'nargs': 0,
                            'const': True, 'default': False, 'type': None,
                            'choices': None, 'required': False, 'help': 'Exclude
                            input words from valid output analogies.The
                            performance of word embeddings on the analogy task is
                            around 0% accuracy if input words are not excluded.',
                            'metavar': None, 'container': <argparse._ArgumentGroup
                            object at 0x1a0e658400>, 'prog':
                            'word_embedding_evaluation.py'}ccuracy if input words
                            are not excluded. (default: False)
      --analogy-max-vocab ANALOGY_MAX_VOCAB
                            Only retain the X first tokens from the pretrained
                            embedding. The tokens are ordererd by decreasing
                            frequency.As the analogy task takes the whole
                            vocabulary into account, removing very infrequent
                            words improves performance. (default: None)
    
    Computation arguments:
      --batch-size BATCH_SIZE
                            Batch size to use on analogy task.Decrease batch size
                            if evaluation crashes. (default: 32)
      --gpu GPU             Number (index) of GPU to run on, e.g. 0. If not
                            specified, uses CPU. (default: None)
      --dont-hybridize      Disable hybridization of gluon HybridBlocks. (default:
                            False)
    
    Logging arguments:
      --log LOG             Path to logfile.Results of evaluation runs are written
                            to there in a CSV format. (default: results.csv)
    

