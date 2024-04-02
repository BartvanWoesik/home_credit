import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from typing import List
import pandas as pd


def gini_score(y_true: List[int], y_pred: list[int]) -> float:
    """
    Calculate the Gini score for binary classification.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.

    Returns:
        float: Gini score.
    """
    check_scorer_input(y_true, y_pred)
    return 2 * roc_auc_score(y_true, y_pred) - 1


def kaggle_score(y_true: np.array, y_pred: np.array) -> float:
    """
    Calculate the Kaggle score for a given set of true and predicted values.

    Parameters:
    - y_true: array-like, true values
    - y_pred: array-like, predicted values

    Returns:
    - Kaggle score: float
    """
    check_scorer_input(y_true, y_pred)
    return gini_score(y_true, y_pred) - 0.5 * np.std(y_true - y_pred)


def check_scorer_input(y_true: list[int], y_pred: list[int]) -> None:
    if not isinstance(y_true, (np.ndarray, list, pd.Series)) or not isinstance(
        y_pred, (np.ndarray, list, pd.Series)
    ):
        raise TypeError("Both y_true and y_pred must be numpy arrays or lists")
    assert len(y_true) == len(y_pred)
    assert len(y_true) > 0


# Dictionary of custom scorers
custom_scorers = {
    "gini": make_scorer(
        gini_score, greater_is_better=True, response_method="predict_proba"
    ),
    "kaggle": make_scorer(
        kaggle_score, greater_is_better=True, response_method="predict_proba"
    ),
}
