import os 
import torch 

# PATHS 
PATH_LINCE_DATASET = os.environ.get("PATH_LINCE_DATASET", "./data/lince/ner")
PATH_GLUECOS_LID = os.environ.get("PATH_GLUECOS_LID_DATASET", "./data/GLUECoS/LID/Romanized")
PATH_GLUECOS_NER = os.environ.get("PATH_GLUECOS_NER_DATASET", "./data/GLUECoS/NER/Romanized")
PATH_BASE_MODELS = os.environ.get("PATH_BASE_MODELS", "./base_models")
PATH_CACHE_DATASET = os.environ.get("PATH_CACHE_DATASET", "./data/cache")

PATH_EXPERIMENTS = os.environ.get("PATH_EXPERIMENTS", "./runs")

# HARDWARE 
NUM_WORKERS = min(4, int(os.cpu_count() / 2))
AVAIL_GPUS = min(1, torch.cuda.device_count())

# HYPERPARAMS 
GLOBAL_SEED = 42

MAX_EPOCHS = 30
BATCH_SIZE = 32

LEARNING_RATE = 3e-5
WARM_RESTARTS = 50          # Restart after 50 epochs

WEIGHT_DECAY = 0
DROPOUT_RATE = 1e-3

BASE_MODEL = "bert-base-multilingual-cased"       # mBERT
MAX_SEQUENCE_LENGTH = 64
PADDING = "max_length"

LABEL2ID = {
    "O": 0, 
    "B-PERSON": 1, 
    "I-PERSON": 2, 
    "B-ORGANISATION": 3, 
    "I-ORGANISATION": 4, 
    "B-PLACE": 5, 
    "I-PLACE": 6,
}

LID2ID = {
    "hi": 0, 
    "en": 1, 
    "rest": 2
}

GLC_NER_LABEL2ID = {
    "Other": 0,
    "B-Per": 1,
    "I-Per": 2,
    "B-Org": 3,
    "I-Org": 4,
    "B-Loc": 5,
    "I-Loc": 6
}

GLC_LID_LABEL2ID = {
    "EN": 0,
    "HI": 1,
    "OTHER": 2
}

GLC_POS_LABEL2ID = {
    "PRON": 0,
    "VERB": 1,
    "ADJ": 2,
    "ADP": 3,
    "AUX": 4,
    "CCONJ": 5,
    "DET": 6,
    "PUNCT": 7,
    "SYM": 8,
    "NOUN": 9,
    "PART": 10,
    "PROPN": 11,
    "ADV": 12,
    "SCONJ": 13,
    "INTJ": 14,
    "NUM": 15,
    "X": 16
}

K_CROSSFOLD_VALIDATION_SPLITS = 10

# PROJECT CONFIGURATION
PROJECT_NAME = "MetaLearning-CodeMix"