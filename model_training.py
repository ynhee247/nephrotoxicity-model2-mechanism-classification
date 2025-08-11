from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from config import CV, SCORING_METRICS, REFIT_METRIC, RANDOM_STATE, DEVICE

# Hyperparameter grids (Bảng 2.4 KL)
PARAM_GRIDS = {
    'svm': {
        'kernel': ['linear', 'rbf'],
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
    },
    'rf': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, 20, 25],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'xgb': {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200, 300],
        'min_child_weight': [1, 5, 10],
        'gamma': [0, 0.1, 0.5]
    }
}

def train_model(X, y, model_name: str, refit_metric=None):
    """
    Hyperparameter tuning with GridSearchCV using multiple scoring metrics.
    Optionally refits on the provided metric to obtain a trained model.
    Returns best_estimator_ (or None), best_score_ (or None), cv_results_
    """

    # Inform the user which device will be used for training
    print(f"Using device: {DEVICE}")
    
    if model_name == 'svm':
        estimator = SVC(random_state=RANDOM_STATE, probability=True)
    elif model_name == 'rf':
        estimator = RandomForestClassifier(random_state=RANDOM_STATE)
    elif model_name == 'xgb':
        if DEVICE == 'cuda':
            estimator = XGBClassifier(
                # use_label_encoder=False,
                eval_metric='logloss',
                random_state=RANDOM_STATE,
                tree_method='hist',
                # predictor='gpu_predictor',
                # gpu_id=0,
            )
        else:
            estimator = XGBClassifier(
                # use_label_encoder=False,
                eval_metric='logloss',
                random_state=RANDOM_STATE,
            )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Create scoring dict for 4 metrics
    SCORERS = {
        'accuracy' : make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, zero_division=0), # Thêm zero_division=0 tránh warning
        'recall'   : make_scorer(recall_score, zero_division=0),
        'f1'       : make_scorer(f1_score, zero_division=0),
    }

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=PARAM_GRIDS[model_name],
        cv=CV,
        scoring=SCORERS,
        refit=refit_metric or False,
        n_jobs=-1
    )
    grid.fit(X,y)
    
    if refit_metric:
        best_estimator = grid.best_estimator_
        best_score = grid.best_score_
    else:
        best_estimator = None
        best_score = None

    return best_estimator, best_score, grid.cv_results_