from sklearn.model_selection import cross_validate
from sklearn.metrics import get_scorer
import numpy as np
from typing import Dict, Any, List


class ModelEvaluator:
    """
    A class for evaluating machine learning models using cross-validation and multiple metrics.

    Attributes:
    - model: The machine learning model to be evaluated.
    - metrics: A list of metrics to be used for evaluation.

    Methods:
    - evaluate(X, y, cv=5): Evaluate the model using cross-validation and multiple metrics.

    """

    def __init__(self, model: Any, metrics: List[str]) -> None:
        self.model = model
        self.metrics = metrics

    def evaluate(self, X: np.array, y: np.array, cv: int = 5) -> Dict[str, Any]:
        """
        Evaluate the model using cross-validation and multiple metrics.

        Parameters:
        - X (array-like): The input features.
        - y (array-like): The target variable.
        - cv (int, optional): The number of cross-validation folds. Default is 5.

        Returns:
        - results (dict): A dictionary containing the evaluation results for each metric.

        """
        scoring = {}
        for metric in self.metrics:
            scoring[metric] = get_scorer(metric)

        results = cross_validate(self.model, X, y, scoring=scoring, cv=cv)
        return results
