import os
import subprocess
import hydra
import mlflow
import optuna

from pathlib import Path
import numpy as np
from functools import partial

from hydra.utils import instantiate
from optuna import create_study
from evaluate.metric_eval import ModelEvaluator

from data_pipeline.pipelinesteps import load_data, data_splitter
from model.modelorchastrator import ModelOrchestrator
from data_pipeline.dataset import Dataset

mlflow.set_tracking_uri("http://localhost:5000")
base_path = Path(os.getcwd())
commit_message = (
    subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).decode().strip()
)


def mlflow_decorator(func):
    def wrapper(trial):
        with mlflow.start_run(experiment_id="107882983913598320", nested=True):
            result = func(trial)
            mlflow.log_params(trial.params)
        return result

    return wrapper


@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def start_tunning(cfg):
    with mlflow.start_run(experiment_id="107882983913598320", run_name=commit_message):
        dataset = Dataset.create_from_pipeline(
            partial(load_data, base_path),
            instantiate(cfg.data_pipeline),
            data_splitter=data_splitter,
            target_column="target",
        )

        @mlflow_decorator
        def objective(trial):
            model_orchestrator = ModelOrchestrator(cfg, trial)
            model = model_orchestrator.create_tuning_pipeline()
            # model.fit(dataset.X_train, dataset.y_train)
            metrics = ["roc_auc", "gini", "kaggle"]
            cv_eval = ModelEvaluator(model, metrics)
            cv_eval_results = cv_eval.evaluate(dataset.X_train, dataset.y_train)
            for metric, values in cv_eval_results.items():
                if (m := metric.removeprefix("test_")) in metrics:
                    mlflow.log_metric(f"cv_{m}", np.mean(values))

            return cv_eval_results["roc_auc"].mean()

        # with mlflow.start_run(experiment_id="264751226398184019", run_name=commit_message, nested=True) as parent_run:
        study = create_study(study_name="optimization", direction="maximize")
        study.optimize(lambda trial: objective(trial), n_trials=40)
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(
            Path(os.getcwd()) / "artifact_storage" / "tune" / "tunehistory.png"
        )

        mlflow.log_artifact(Path(os.getcwd()) / "artifact_storage/tune")
        mlflow.log_metric("roc_auc", study.best_value)
        mlflow.log_params(study.best_params)


if __name__ == "__main__":
    start_tunning()
