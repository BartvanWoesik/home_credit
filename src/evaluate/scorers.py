from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
import numpy as np


def gini_score(y_true, y_pred):
    """
    Calculate the Gini score for binary classification.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: Gini score.
    """
    return 2 * roc_auc_score(y_true, y_pred) - 1


def kaggle_score(y_true, y_pred):
    """
    Calculate the Kaggle score for a given set of true and predicted values.

    Parameters:
    - y_true: array-like, true values
    - y_pred: array-like, predicted values

    Returns:
    - Kaggle score: float
    """
    return gini_score(y_true, y_pred) - 0.5 * np.std(y_pred)


# Dictionary of custom scorers
custom_scorers = {
    "gini": make_scorer(gini_score, greater_is_better=True),
    "kaggle": make_scorer(kaggle_score, greater_is_better=True),
}
