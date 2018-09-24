r"""
This file contains all hyperparameters.
"""
import mxnet as mx

DATA_PATH = '../data/'
RAW_TRAIN_FILE = 'train-v1.1.json'
RAW_DEV_FILE = 'dev-v1.1.json'
GLOVE_FILE_NAME = 'glove.840B.300d.txt'
TRAIN_FILE_NAME = 'train_sorted.json'
DEV_FILE_NAME = 'dev_sorted.json'
WORD_EMB_FILE_NAME = 'word_emb.json'
CHAR_EMB_FILE_NAME = 'char_emb.json'

ACCUM_AVG_TRAIN_CROSS_ENTROPY = '../logs/accum_avg_train_cross_entropy.json'
BATCH_TRAIN_CROSS_ENTROPY = '../logs/batch_train_cross_entropy.json'
TRAIN_F1 = '../logs/train_f1.json'
TRAIN_EM = '../logs/train_em.json'
DEV_CROSS_ENTROPY = '../logs/dev_cross_entropy.json'
DEV_F1 = '../logs/dev_f1.json'
DEV_EM = '../logs/dev_em.json'

EVALUATE_INTERVAL = 5000

TRAIN_FLAG = True
CTX = [mx.gpu(1)]

EPOCHS = 60
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 16

# model save & load
SAVE_MODEL_PREFIX_NAME = 'model_'
SAVE_TRAINER_PREFIX_NAME = 'trainer_'
NEED_LOAD_TRAINED_MODEL = False
TARGET_MODEL_FILE_NAME = 'model_Wed Jul 25 18:12:54 2018'
TARGET_TRAINER_FILE_NAME = ''
LAST_GLOBAL_STEP = 0

# dropout
LAYERS_DROPOUT = 0.1
p_L = 0.9
WORD_EMBEDDING_DROPOUT = 0.1
CHAR_EMBEDDING_DROPOUT = 0.05
HIGHWAY_LAYERS_DROPOUT = 0.1

# padding parameter
MAX_CONTEXT_SENTENCE_LEN = 400
MAX_QUESTION_SENTENCE_LEN = 50
MAX_CHARACTER_PER_WORD = 16

# embedding parameter
UNK = 1
PAD = 0

DIM_WORD_EMBED = 300
DIM_CHAR_EMBED = 200

CORPUS_CHARACTERS = 1372 + 2
CORPUS_WORDS = 86822 + 2

NUM_HIGHWAY_LAYERS = 2

# embedding encoder parameter
EMB_ENCODER_CONV_CHANNELS = 128
EMB_ENCODER_CONV_KERNEL_SIZE = 7
EMB_ENCODER_NUM_CONV_LAYERS = 4
EMB_ENCODER_NUM_HEAD = 1
EMB_ENCODER_NUM_BLOCK = 1

# model encoder parameter
MODEL_ENCODER_CONV_KERNEL_SIZE = 5
MODEL_ENCODER_CONV_CHANNELS = 128
MODEL_ENCODER_NUM_CONV_LAYERS = 2
MODEL_ENCODER_NUM_HEAD = 1
MODEL_ENCODER_NUM_BLOCK = 7

# opt parameter
INIT_LEARNING_RATE = 0.001
CLIP_GRADIENT = 5
WEIGHT_DECAY = 3e-7
BETA1 = 0.8
BETA2 = 0.999
EPSILON = 1e-7
EXPONENTIAL_MOVING_AVERAGE_DECAY = 0.9999
WARM_UP_STEPS = 1000

# evaluate
MAX_ANSWER_LENS = 30
