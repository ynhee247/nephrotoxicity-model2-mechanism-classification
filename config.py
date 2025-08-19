import os
import subprocess
from sklearn.model_selection import StratifiedKFold


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
# Default paths (can be overridden via CLI)
FP_DIR   = os.path.join(DATA_DIR, 'fingerprints')
LBL_DIR  = os.path.join(DATA_DIR, 'labels')

# Algorithms to train
MODELS = ['svm', 'rf', 'xgb']

# Reproducibility
RANDOM_STATE = 42

# Cross-validation settings
CV_N_SPLITS = 5 # number of folds for CV
CV_SPLITTER = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
SCORING_METRICS = ['accuracy', 'precision', 'recall', 'f1']


def gpu_available() -> bool:
    """Return True if a CUDA-capable GPU is detected."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


# Use GPU when available
DEVICE = "cuda" if gpu_available() else "cpu"