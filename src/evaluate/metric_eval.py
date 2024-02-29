from sklearn.model_selection import cross_validate
from sklearn.metrics import get_scorer
from sklearn.metrics import get_scorer_names
import numpy as np
from typing import Dict, Any, List
from evaluate.scorers import custom_scorers
from my_logger.custom_logger import logger


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
            if metric in get_scorer_names():
                scoring[metric] = get_scorer(metric)
            elif metric in list(custom_scorers.keys()):
                scoring[metric] = custom_scorers[metric]
            else:
                logger.info(f"Metric {metric} not recognized. Ignoring.")

        results = cross_validate(self.model, X, y, scoring=scoring, cv=cv)
        return results
