from typing import Iterable
from numpy import ndarray
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from hydra.utils import instantiate
from sklearn.base import BaseEstimator, TransformerMixin
from optbinning import BinningProcess
from omegaconf import DictConfig, OmegaConf


class ModelOrchestrator:
    """
    Class for orchestrating the model pipeline and tuning.

    Args:
        cfg (dict): Configuration parameters for the model pipeline.
        trial (optuna.Trial, optional): Optuna trial object for hyperparameter tuning. Defaults to None.
    """

    def __init__(self, cfg, trial=None, tuning_params=None):
        self.trial = trial
        self.cfg = cfg
        self.tuning_params = tuning_params
        if trial:
            self.tuning_params = self.create_tuning_params(cfg)

    def create_tuning_pipeline(self):
        """
        Create a tuning pipeline based on the model pipeline and configuration.

        Returns:
            sklearn.pipeline.Pipeline: Tuning pipeline.
        """
        return self.modelpipeline.create_from_config(self.cfg, self.tuning_params)

    def create_pipe_line(self):
        """
        Create a pipeline based on the model pipeline and configuration.

        Returns:
            sklearn.pipeline.Pipeline: Tuning pipeline.
        """
        return CustomModelPipeline.create_from_config(self.cfg)

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
                if "type" in short_cfg[parameter]:
                    if short_cfg[parameter].type == "float":
                        params_step[parameter] = self.trial.suggest_float(
                            parameter,
                            short_cfg[parameter].min,
                            short_cfg[parameter].max,
                        )
                    elif short_cfg[parameter].type == "int":
                        params_step[parameter] = self.trial.suggest_int(
                            parameter,
                            short_cfg[parameter].min,
                            short_cfg[parameter].max,
                        )
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

    def predict(
        self, X: list[str] | ndarray | Iterable | DataFrame, **predict_params
    ) -> ndarray | tuple[ndarray, ndarray]:
        return super().predict_proba(X, **predict_params).T[1]

    @classmethod
    def create_from_config(
        cls, cfg: DictConfig | OmegaConf, params=None
    ) -> "CustomModelPipeline":
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
                pipeline_list.append(
                    (str(i), instantiate(_step_dict[1], **(params[_step_dict[0]])))
                )
            else:
                pipeline_list.append((str(i), instantiate(_step_dict[1])))

        # Create instance of cls
        custom_pipeline = cls(steps=pipeline_list)
        return custom_pipeline


class OptBinningTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        variable_names=None,
        max_n_prebins=100,
        min_prebin_size=0.1,
        random_state=None,
    ):
        self.variable_names = variable_names if variable_names is not None else []
        self.max_n_prebins = max_n_prebins
        self.min_prebin_size = min_prebin_size
        self.random_state = random_state
        self.binning_process = BinningProcess(
            variable_names=list(self.variable_names),
            max_n_prebins=self.max_n_prebins,
            min_prebin_size=self.min_prebin_size,
        )

    def fit(self, X, y=None):
        # Fit OptBinning on the specified column
        self.binning_process.fit(X[self.variable_names].values, y)
        return self

    def transform(self, X):
        # Transform the specified column using OptBinning
        X_transformed = X.copy()
        transformed_column = self.binning_process.transform(
            X[list(self.variable_names)].values
        )
        X_transformed[list(self.variable_names)] = transformed_column
        return X_transformed
