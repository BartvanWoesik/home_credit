import os
import subprocess
import hydra
import mlflow
import optuna

from pathlib import Path
from functools import partial

from hydra.utils import instantiate
from optuna import samplers, create_study
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path

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
                    mlflow.log_metric('AUC', result)
                return result
            return wrapper


@hydra.main(version_base= None, config_path="../conf", config_name="config.yaml")
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
            pipe = model_orchestrator.create_tuning_pipeline()
            pipe.fit(dataset.X_train, dataset.y_train)
            y_pred = pipe.predict_proba(dataset.X_test)[:,1]
            auc = roc_auc_score(dataset.y_test, y_pred)
            return auc



        # with mlflow.start_run(experiment_id="264751226398184019", run_name=commit_message, nested=True) as parent_run:
        study = create_study(study_name='optimization', direction='maximize' )
        study.optimize(lambda trial: objective(trial), n_trials=40)
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(Path(os.getcwd()) / "artifact_storage" / "tune" / "tunehistory.png")

        mlflow.log_artifact(Path(os.getcwd())/ "artifact_storage/tune")
        mlflow.log_metric('AUC', study.best_value)
        mlflow.log_params(study.best_params)

if __name__ == "__main__":
   start_tunning()