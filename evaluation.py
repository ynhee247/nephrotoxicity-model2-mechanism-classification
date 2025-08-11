from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    """
    Compute metrics: report, ROC-AUC, confusion matrix.
    Returns dict of metrics.
    """
    y_pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'): # check xem model có predict_proba không (RF, XGB có, SVM ko -> dùng decision_function)
        y_proba = model.predict_proba(X_test)[:,1]
    else:
        y_proba = model.decision_function(X_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    return {
        'report': report,
        'roc_auc': auc,
        'confusion_matrix': cm # [[TN, FP], [FN, TP]]
    }


def plot_confusion_matrix(cm, labels=None, cmap='Blues', filename=None):
    """Plot and optionally save a confusion matrix."""
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(cmap=cmap, ax=ax, colorbar=False)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close(fig)