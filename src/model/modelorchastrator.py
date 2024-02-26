from sklearn.pipeline import Pipeline
from hydra.utils import instantiate
from sklearn.base import BaseEstimator, TransformerMixin
from optbinning import BinningProcess


class ModelOrchestrator:
    def __init__(self, cfg):
        self.modelpipeline = self.create_model(cfg.model_steps)

    def create_model(self, modelsteps: list):
        pipeline = Pipeline(steps=[])
        pipeline_list = []
        for i, step in enumerate(modelsteps):
            pipeline_list.append((str(i), instantiate(step)))
        pipeline = Pipeline(steps=pipeline_list)
        return CustomModelPipeline(pipeline)


class CustomModelPipeline:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, X, y):
        # Add your code here to fit the model
        self.pipeline.fit(X, y)

    def predict(self, X):
        # Add your code here to make predictions
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        # Add your code here to make probability predictions
        return self.pipeline.predict_proba(X)


class OptBinningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = list(column)

    def fit(self, X, y=None):
        # Fit OptBinning on the specified column
        self.binning_process = BinningProcess(variable_names=self.column)
        self.binning_process.fit(X[self.column].values, y)
        return self

    def transform(self, X):
        # Transform the specified column using OptBinning
        X_transformed = X.copy()
        transformed_column = self.binning_process.transform(X[self.column].values)
        X_transformed[self.column] = transformed_column
        return X_transformed
