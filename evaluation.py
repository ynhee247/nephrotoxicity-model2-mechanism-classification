from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

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