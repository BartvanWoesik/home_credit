from sklearn.metrics import get_scorer

from evaluate.scorers import custom_scorers


SKLEARN_METRICS = {metric: get_scorer(metric) for metric in ["roc_auc"]}
METRICS = dict(custom_scorers, **SKLEARN_METRICS)
