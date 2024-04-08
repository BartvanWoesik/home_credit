"""
Implemented a FactoryDesign pattern to create a pipeline based on the configuration provided.
Implemented pipelines:
1. ModelPipeline
2. TuningPipeline
"""


from abc import ABC, abstractmethod

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from sklearn.pipeline import Pipeline


class Orchestartor(ABC):
    def __init__(self, cfg) -> None:
        self.cfg = cfg

    @abstractmethod
    def create_pipeline(self):
        ...

    @abstractmethod
    def features_in(self, cfg):
        ...


class ModelOrchestrator(Orchestartor):
    def create_pipeline(self):
        return ModelPipeline.create_from_config(self.cfg)

    def features_in(self, cfg):
        return cfg.features


class TuningOrchestrator(Orchestartor):
    def __init__(self, cfg, trial) -> None:
        super().__init__(cfg)
        self.trial = trial

    def create_pipeline(self):
        return TuningPipeline.create_from_config(self.cfg, self.trial)

    def features_in(self, cfg):
        return cfg.features


class CustomPipeline(ABC, Pipeline):
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

    @staticmethod
    def create_from_config():
        ...


class TuningPipeline(CustomPipeline):
    @classmethod
    def create_from_config(cls, cfg: DictConfig | OmegaConf, trial) -> "TuningPipeline":
        """
        Create a custom model pipeline from the provided configuration.

        Args:
            cfg (DictConfig | OmegaConf): The configuration object.

        Returns:
            TuningPipeline: The created custom model pipeline.
        """
        # First create list of tuples from the modelsteps list
        params = cls.create_tuning_params(cfg, trial)
        pipeline_list = []
        for i, step in enumerate(cfg.model.model_steps):
            _step_dict = next(iter(step.items()))

            pipeline_list.append(
                (str(i), instantiate(_step_dict[1], **(params[_step_dict[0]])))
            )

        # Create instance of cls
        custom_pipeline = cls(steps=pipeline_list)
        return custom_pipeline

    @classmethod
    def create_tuning_params(cls, cfg, trial):
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
                        params_step[parameter] = trial.suggest_float(
                            parameter,
                            short_cfg[parameter].min,
                            short_cfg[parameter].max,
                        )
                    elif short_cfg[parameter].type == "int":
                        params_step[parameter] = trial.suggest_int(
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


class ModelPipeline(CustomPipeline):
    """
    Custom pipeline class for defining a model pipeline.
    """

    @classmethod
    def create_from_config(cls, cfg: DictConfig | OmegaConf) -> "ModelPipeline":
        """
        Create a custom model pipeline from the provided configuration.

        Args:
            cfg (DictConfig | OmegaConf): The configuration object.

        Returns:
            ModelPipeline: The created custom model pipeline.
        """
        # First create list of tuples from the modelsteps list
        pipeline_list = []
        for i, step in enumerate(cfg.model.model_steps):
            _step_dict = next(iter(step.items()))
            pipeline_list.append((str(i), instantiate(_step_dict[1])))

        # Create instance of cls
        custom_pipeline = cls(steps=pipeline_list)
        return custom_pipeline
