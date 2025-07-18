import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
# Default paths (can be overridden via CLI)
FP_DIR   = os.path.join(DATA_DIR, 'fingerprints')
LBL_DIR  = os.path.join(DATA_DIR, 'labels')

# Algorithms to train
MODELS = ['svm', 'rf', 'xgb']

# Grid search settings
CV = 5 # number of folds for CV
SCORING_METRICS = ['accuracy', 'precision', 'recall', 'f1']
REFIT_METRIC = 'f1'
RANDOM_STATE = 42