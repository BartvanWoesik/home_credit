from sklearn.base import BaseEstimator, TransformerMixin
from my_logger.custom_logger import logger
from optuna import Trial


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features: list[str] = []) -> None:
        self.features = set(features)

    def transform(self, X):
        logger.info('Selecting features from Dataset')
        return X[list(self.features)]

    def fit_transform(self, X, *fit_args, **fit_kwargs):
        return self.transform(X)

    def fit(self, X, *fit_args, **fit_kwargs):
        return self
    
    @classmethod
    def instantiate_trial(cls, cfg, trial: Trial):
        return cls(features=cfg.features.default)