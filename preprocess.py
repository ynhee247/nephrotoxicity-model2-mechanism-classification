from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from config import RANDOM_STATE

def resample_data(X, y, method: str = 'smote'):
    """
    Always perform resampling for both methods 'smote' and 'rus'.
    Print counts and distribution before and after resampling.
    method: 'smote' or 'rus'
    Returns X_res, y_res
    """

    # Before resampling
    total_before = len(y)
    counts_before = Counter(y)
    ratio_before_0 = counts_before[0] / total_before
    ratio_before_1 = counts_before[1] / total_before
    print(f"Before {method}: total={total_before}, 0 -> {counts_before[0]} ({ratio_before_0:.2%}), 1 -> {counts_before[1]} ({ratio_before_1:.2%})")

    # Xử lý mất cân bằng dữ liệu
    method = method.lower()
    if method == 'smote':
        sampler = SMOTE(random_state=RANDOM_STATE)
    elif method == 'rus':
        sampler = RandomUnderSampler(random_state=RANDOM_STATE)
    else:
        raise ValueError("method must be 'smote' or 'rus'")

    # Resample
    X_res, y_res = sampler.fit_resample(X, y)

    # After resampling
    total_after = len(y_res)
    counts_after = Counter(y_res)
    ratio_after_0 = counts_after[0] / total_after
    ratio_after_1 = counts_after[1] / total_after
    print(f"After {method}: total={total_after}, 0 -> {counts_after[0]} ({ratio_after_0:.2%}), 1 -> {counts_after[1]} ({ratio_after_1:.2%})")

    return X_res, y_res