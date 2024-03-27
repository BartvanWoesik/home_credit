from sklearn.pipeline import Pipeline
from hydra.utils import instantiate
from sklearn.base import BaseEstimator, TransformerMixin
from optbinning import BinningProcess
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import HistGradientBoostingClassifier


class ModelOrchestrator:
    """
    Class for orchestrating the model pipeline and tuning.

    Args:
        cfg (dict): Configuration parameters for the model pipeline.
        trial (optuna.Trial, optional): Optuna trial object for hyperparameter tuning. Defaults to None.
    """

    def __init__(self, cfg, trial=None):
        self.trial = trial
        self.cfg = cfg
        self.tuning_params = None
        if trial:
            self.tuning_params = self.create_tuning_params(cfg)
        self.modelpipeline = CustomModelPipeline.create_from_config(cfg)

    def create_tuning_pipeline(self):
        """
        Create a tuning pipeline based on the model pipeline and configuration.

        Returns:
            sklearn.pipeline.Pipeline: Tuning pipeline.
        """
        return self.modelpipeline.create_from_config(self.cfg, self.tuning_params)
    

    def create_tuning_params(self, cfg):
        """
        Create tuning parameters based on the provided configuration.

        Args:
            cfg (object): The configuration object containing hyperparameters.

        Returns:
            dict: A dictionary containing the tuning parameters for each step.

        """
        params = {}
        for step in cfg.hyperparameters:
            params_step = {}
            for parameter in cfg.hyperparameters[step]:
                short_cfg = cfg.hyperparameters[step]
                if 'type' in short_cfg[parameter]:
                    if short_cfg[parameter].type == "float":
                        params_step[parameter] = self.trial.suggest_float(parameter, short_cfg[parameter].min, short_cfg[parameter].max)
                    elif short_cfg[parameter].type == "int":
                        params_step[parameter] = self.trial.suggest_int(parameter, short_cfg[parameter].min, short_cfg[parameter].max)
                    else:
                        params_step[parameter] = short_cfg[parameter].default
                else:
                    # Handle the case when 'type' key is missing in cfg[parameter]
                    params_step[parameter] = short_cfg[parameter].default
            params[step] = params_step
        return params
        

  


class CustomModelPipeline(Pipeline):
    """
    Custom pipeline class for defining a model pipeline.
    """

    def transform_without_predictor(self, X):
        """
        Transform the data without using the predictor step.

        Args:
            X (array-like): The input data to be transformed.

        Returns:
            array-like: The transformed data.
        """
        # Add your code here to transform the data
        return self[:-1].transform(X)

    @classmethod
    def create_from_config(cls, cfg: DictConfig | OmegaConf, params = None) -> "CustomModelPipeline":
        """
        Create a custom model pipeline from the provided configuration.

        Args:
            cfg (DictConfig | OmegaConf): The configuration object.

        Returns:
            CustomModelPipeline: The created custom model pipeline.
        """
        # First create list of tuples from the modelsteps list
        pipeline_list = []
        for i, step in enumerate(cfg.model.model_steps):
            _step_dict = next(iter(step.items()))
            if params is not None:
                pipeline_list.append((str(i), instantiate(_step_dict[1],  **(params[_step_dict[0]]))))
            else:
                pipeline_list.append((str(i), instantiate(_step_dict[1])))

        # Create instance of cls
        custom_pipeline = cls(steps=pipeline_list)
        return custom_pipeline

        



class OptBinningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column = [], max_n_prebins = 100, min_prebin_size = 0.1):
        self.column = list(column)
        self.max_n_prebins = max_n_prebins
        self.min_prebin_size = min_prebin_size

    def fit(self, X, y=None):
        # Fit OptBinning on the specified column
        self.binning_process = BinningProcess(variable_names=self.column, max_n_prebins=self.max_n_prebins, min_prebin_size=self.min_prebin_size )
        self.binning_process.fit(X[self.column].values, y)
        return self

    def transform(self, X):
        # Transform the specified column using OptBinning
        X_transformed = X.copy()
        transformed_column = self.binning_process.transform(X[self.column].values)
        X_transformed[self.column] = transformed_column
        return X_transformed
    
    @classmethod
    def instantiate_trial(cls, cfg, trial):
        return cls(column=cfg.column.default, 
                    max_n_prebins=trial.suggest_int("max_n_prebins", cfg.max_n_prebins.min, cfg.max_n_prebins.max),
                    min_prebin_size=cfg.min_prebin_size.default)

class HistBooster(HistGradientBoostingClassifier):

    pass
