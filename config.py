import os
import subprocess

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