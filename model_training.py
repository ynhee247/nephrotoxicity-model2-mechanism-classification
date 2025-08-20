import numpy as np
import pandas as pd
import cupy as cp
import cudf

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from config import CV_SPLITTER, SCORING_METRICS, RANDOM_STATE, DEVICE


# Hyperparameter grids (Bảng 2.4 KL)
PARAM_GRIDS = {
    # SVM: tách linear vs rbf (gamma chỉ áp dụng cho rbf)
    'svm_linear': {
        'clf__kernel': ['linear'],
        'clf__C': [0.1, 1, 10, 100, 1000],
    },
    'svm_rbf': {
        'clf__kernel': ['rbf'],
        'clf__C': [0.1, 1, 10, 100, 1000],
        'clf__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    },
    'rf': {
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [5, 10, 15, 20, 25],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 2, 4]
    },
    'xgb': {
        'clf__max_depth': [3, 5, 7],
        'clf__learning_rate': [0.01, 0.1, 0.2],
        'clf__n_estimators': [50, 100, 200, 300],
        'clf__min_child_weight': [1, 5, 10],
        'clf__gamma': [0, 0.1, 0.5]
    }
}


def _to_gpu_array(X, y):
    """
    Cast numpy/pandas -> cupy/cudf if DEVICE = 'cuda'
    """
    # Cast X
    if isinstance(X, np.ndarray):
        X_gpu = cp.asarray(X)
    elif isinstance(X, pd.DataFrame):
        X_gpu = cudf.DataFrame.from_pandas(X)
    else:
        X_gpu = X  # giả sử đã là cupy/cudf
    
    # Cast y
    if isinstance(y, np.ndarray):
        y_gpu = cp.asarray(y)
    elif isinstance(y, pd.Series):
        y_gpu = cudf.Series(y)
    else:
        y_gpu = y
    
    return X_gpu, y_gpu


def _build_estimator(model_name: str):
    if model_name == 'svm':
        return SVC(random_state=RANDOM_STATE, probability=True)
    
    elif model_name == 'rf':
        return RandomForestClassifier(random_state=RANDOM_STATE)
    
    elif model_name == 'xgb':
        if DEVICE == 'cuda':
            try:
                return XGBClassifier(
                    eval_metric='logloss',
                    random_state=RANDOM_STATE,
                    tree_method='hist',
                    device='cuda',
                    predictor='gpu_predictor'
                )
            except TypeError:
                # Fallback cho phiên bản xgboost cũ (không có tham số 'device')
                return XGBClassifier(
                    eval_metric='logloss',
                    random_state=RANDOM_STATE,
                    tree_method='gpu_hist',
                    predictor='gpu_predictor'
                )
        # CPU fallback
        return XGBClassifier(
            eval_metric='logloss',
            random_state=RANDOM_STATE,
            tree_method='hist'
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")
    

def build_pipeline(model_name: str, sampler=None):
    if sampler:
        return ImbPipeline([
            ('sampler', sampler),
            ('clf', _build_estimator(model_name))
        ])
    else:
        return ImbPipeline([
            ('clf', _build_estimator(model_name))
        ])


def _scorers():
    return {
        'accuracy' : make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, zero_division=0), # Thêm zero_division=0 tránh warning
        'recall'   : make_scorer(recall_score, zero_division=0),
        'f1'       : make_scorer(f1_score, zero_division=0),
    }


def train_model(X, y, model_name: str, refit_metric=None, params=None, sampler=None):
    """
    Hyperparameter tuning with GridSearchCV using multiple scoring metrics.
    Optionally refits on the provided metric or fits directly with supplied params.
    """

    # Inform the user which device will be used for training
    print(f"Using device: {DEVICE}")

    # Nếu dùng GPU thì cast sang cupy/cudf
    if DEVICE == 'cuda' and model_name == 'xgb':
        X, y = _to_gpu_array(X, y)

    pipe = build_pipeline(model_name, sampler)

    # Fit trực tiếp để retrain final
    if params:
        mapped = { (k if str(k).startswith('clf__') else f'clf__{k}') : v for k, v in params.items() }
        pipe.set_params(**mapped)
        return pipe.fit(X, y), None, None
    
    # Grid Search
    param_grid = [PARAM_GRIDS['svm_linear'], PARAM_GRIDS['svm_rbf']] if model_name == 'svm' else PARAM_GRIDS[model_name]
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=CV_SPLITTER, # StratifiedKFold(shuffle=True, random_state=42)
        scoring=_scorers(),
        refit=refit_metric or False,
        n_jobs=-1
    )
    grid.fit(X, y)
    if refit_metric:
        best_estimator = grid.best_estimator_
        best_score = grid.best_score_
    else:
        best_estimator = None
        best_score = None
    
    return best_estimator, best_score, grid.cv_results_