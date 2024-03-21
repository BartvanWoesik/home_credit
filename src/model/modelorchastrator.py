from sklearn.pipeline import Pipeline
from hydra.utils import instantiate
from sklearn.base import BaseEstimator, TransformerMixin
from optbinning import BinningProcess
from omegaconf import DictConfig, OmegaConf


class ModelOrchestrator:
    def __init__(self, cfg):
        self.modelpipeline = CustomModelPipeline.create_from_config(cfg)

  


class CustomModelPipeline(Pipeline):
   

   @classmethod
   def create_from_config(cls, cfg: DictConfig | OmegaConf) -> "CustomModelPipeline":
        # First create list of tuples from the modelsteps list
        pipeline_list = []
        for i, step in enumerate(cfg.model.model_steps):
            pipeline_list.append((str(i), instantiate(step)))

        # Create instance of cls
        custom_pipeline = cls(steps=pipeline_list)
        return custom_pipeline
   
   def transform_without_predictor(self, X):
        # Add your code here to transform the data
        return self[:-1].transform(X)



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
