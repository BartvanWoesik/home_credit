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
    """
    Abstract base class for orchestrating the model pipeline.

    Attributes:
        cfg (dict): Configuration parameters for the orchestrator.

    Methods:
        create_pipeline(): Abstract method for creating the model pipeline.
        features_in(cfg): Abstract method for processing the input features.

    """

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    @abstractmethod
    def create_pipeline(self):
        ...

    @abstractmethod
    def features_in(self, cfg):
        ...


class ModelOrchestrator(Orchestartor):
    """
    Class representing the model orchestrator.

    This class is responsible for creating a model pipeline and managing the features.

    Attributes:
        cfg (Config): The configuration object.

    Methods:
        create_pipeline: Creates a model pipeline based on the configuration.
        features_in: Returns the features specified in the configuration.

    """

    def create_pipeline(self):
        """
        Creates a model pipeline based on the configuration.

        Returns:
            ModelPipeline: The created model pipeline.

        """
        return ModelPipeline.create_from_config(self.cfg)

    def features_in(self, cfg):
        """
        Returns the features specified in the configuration.

        Args:
            cfg (Config): The configuration object.

        Returns:
            list: The list of features.

        """
        return cfg.features


class TuningOrchestrator(Orchestartor):
    """
    Class representing the orchestrator for model tuning.
    """

    def __init__(self, cfg, trial) -> None:
        super().__init__(cfg)
        self.trial = trial

    def create_pipeline(self):
        """
        Creates a tuning pipeline based on the configuration and trial.
        """
        return TuningPipeline.create_from_config(self.cfg, self.trial)

    def features_in(self, cfg):
        """
        Returns the features specified in the given configuration.
        """
        return cfg.features


class TuningParameter:
    def __init__(
        self,
        parameter_name: str,
        trial_type: str,
        search_space: list = None,
        default_value: list | float = None,
    ) -> None:
        self.parameter_name = parameter_name
        self.search_space = search_space
        self.default_value = default_value
        self.trial_type = trial_type
        self._check_input()

    def _check_input(self):
        if self.trial_type in ("float", "int") and len(self.search_space) != 2:
            raise ValueError("search_space must have 2 values for float and int types")
        if self.trial_type == "categorical" and len(self.search_space) == 0:
            raise ValueError(
                "search_space must have at least 1 value for categorical type"
            )
        if self.trial_type not in ("float", "int", "categorical", "default"):
            raise ValueError("Invalid trial_type. Choose float, int or categorical.")
        if (
            self.trial_type in ("float", "int")
            and self.search_space[0] > self.search_space[1]
        ):
            raise ValueError(
                "Invalid search_space. First value must be less than the second value for float and int types"
            )

    def create_range(self, trial):
        suggest_methods = {
            "float": trial.suggest_float,
            "int": trial.suggest_int,
            "categorical": trial.suggest_categorical,
        }

        suggest_method = suggest_methods.get(self.trial_type)
        if self.trial_type == "categorical":
            return suggest_method(self.parameter_name, self.search_space)
        elif self.trial_type == "default":
            return self.default_value
        return suggest_method(
            self.parameter_name, self.search_space[0], self.search_space[1]
        )


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
            params_steps = {}
            for parameters in cfg.hyperparameters[step]:
                parameter_trial = instantiate(parameters)
                params_steps[
                    parameter_trial.parameter_name
                ] = parameter_trial.create_range(trial)
            params[step] = params_steps
        return params

        # params_step = {}
        # short_cfg = cfg.hyperparameters[step]
        # for parameter in cfg.hyperparameters[step]:
        #     if "type" in short_cfg[parameter]:
        #         if short_cfg[parameter].type == "float":
        #             params_step[parameter] = trial.suggest_float(
        #                 parameter,
        #                 short_cfg[parameter].min,
        #                 short_cfg[parameter].max,
        #             )
        #         elif short_cfg[parameter].type == "int":
        #             params_step[parameter] = trial.suggest_int(
        #                 parameter,
        #                 short_cfg[parameter].min,
        #                 short_cfg[parameter].max,
        #             )
        #         else:
        #             params_step[parameter] = short_cfg[parameter].default
        #     else:
        #         # Handle the case when 'type' key is missing in cfg[parameter]
        #         params_step[parameter] = short_cfg[parameter].default
        # params[step] = params_step


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
