'''
Created on 21 nov. 2017

@author: lvanni
'''
import numpy as np

# Model Hyperparameters

# TOKENIZER : KEEP ONLY MOST FREQUENT WORD (NONE == ALL)
MAX_NB_WORDS = None

# ------------- EMBEDDING -------------
EMBEDDING_DIM = 400

# float >= 0. 0 for no negative (=random) samples. 1 for same number as positive samples. etc.
NEGATIVE_SAMPLES = 5.

# Window size for finding collocation (cooccurrence calcul)
WINDOW_SIZE = 10

# ------------- CNN -------------
# Size of the validation dataset (0.1 => 10%)
VALIDATION_SPLIT = 0.2

# SIZE OF SENTENCE (NUMBER OF WORDS)
MAX_SEQUENCE_LENGTH = 15

NB_FILTERS = 50

# 3 filtres:
# 1) taille 3
# 2) taille 4
# 3) taille 5
#FILTER_SIZES = [3,4,5]
FILTER_SIZES = [3,4,5]
# 3 filtres de taille 2 a chaque fois pour le maxpooling
#FILTER_POOL_LENGTHS = [3,3,3]

DROPOUT_VAL = 0.2

DENSE_LAYER_SIZE = 10

NUM_EPOCHS = 5
BATCH_SIZE = 10

# label delimiter
LABEL_MARK = "__"

# Label Dictionary:
LABEL_DIC_2={
        "will":0,
        "do":1,
        "shall":2,
        "may":3,
        "would":4,
        "can":5,
        "should":6,
        "must":7,
        "doth":8,
        "did":9,
        "could":10,
        "might":11,
        "wouldst":12,
        "didst":13,
        "dost":14,
        "wilt":15,
        "mayst":16,
        "canst":17,
        "shouldst":18,
        "couldst":19,
        "mightst":20,
	"shalt":21,
#	"does":22
        }
LABEL_DIC={"LEXICAL":0,"MODAL":1}



# indexsystem
#INSY_PATH = "/home/sara/sentenceSplit/src/split/indexsystem_bi"
INSY_PATH = "indexsystemwg_bi"

# OOV vector
#OOV_VECTOR = np.load("/home/sara/sentenceSplit/src/lstms/oov.npy")
OOV_VECTOR = "/scratch/antwerpen/205/vsc20501/oov.npy"

# Train embedding weights
TRAIN_EMBEDDING_WEIGHTS = True

MASKER = "target"
