import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone


def evaluate_model(model, X_test, y_test):
    """
    Compute test-set metrics: report, ROC-AUC, confusion matrix.
    Returns dict of metrics.
    """
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'): # check xem model có predict_proba không (RF, XGB có, SVM ko -> dùng decision_function)
        y_proba = model.predict_proba(X_test)[:,1]
    else:
        y_proba = model.decision_function(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    return {
        'report': report,
        'roc_auc': auc,
        'confusion_matrix': cm # [[TN, FP], [FN, TP]]
    }


def plot_confusion_matrix(cm, labels=None, cmap='Blues', filename=None, title=None, xlabel=None, ylabel=None):
    """Plot and optionally save a confusion matrix."""
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels or ['0', '1'])
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    disp.plot(cmap=cmap, ax=ax, colorbar=False)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close(fig)


def oof_eval_for_params(estimator_pipeline, X, y, params, cv_splitter):
    """
    TÍNH TOÁN OOF (out-of-fold) CHO MỘT BỘ THAM SỐ CỤ THỂ:
      - set_params(**params) vào pipeline (có sampler bên trong)
      - cross_val_predict để lấy OOF predictions (+ OOF proba/scores)
      - trả về: cm(2x2), accuracy, precision, recall, f1, roc_auc, report(dict)
    """

    est = clone(estimator_pipeline)
    est.set_params(**params)

    # OOF predicted labels
    y_pred_oof = cross_val_predict(est, X, y, cv=cv_splitter, method='predict', n_jobs=-1)

    # OOF scores/proba cho AUC
    y_score_oof = None
    # Ưu tiên predict_proba nếu có
    try:
        proba = cross_val_predict(est, X, y, cv=cv_splitter, method='predict_proba', n_jobs=-1)
        y_score_oof = proba[:, 1]
    except Exception:
        try:
            y_score_oof = cross_val_predict(est, X, y, cv=cv_splitter, method='decision_function', n_jobs=-1)
        except Exception:
            y_score_oof = y_pred_oof # fallback

    cm = confusion_matrix(y, y_pred_oof, labels=[0, 1])
    
    return {
        'cm': cm,
        'accuracy': accuracy_score(y, y_pred_oof),
        'precision': precision_score(y, y_pred_oof, zero_division=0),
        'recall': recall_score(y, y_pred_oof, zero_division=0),
        'f1': f1_score(y, y_pred_oof, zero_division=0),
        'roc_auc': roc_auc_score(y, y_score_oof) if len(set(y)) > 1 else float('nan'),
        'report': classification_report(y, y_pred_oof, output_dict=True, zero_division=0)
    }