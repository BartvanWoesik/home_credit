import os
import subprocess
import shutil

from pathlib import Path
from functools import partial

import hydra
import mlflow
import mlflow.sklearn
import numpy as np

from hydra.utils import instantiate

from my_logger.custom_logger import logger

from model_forge.data.dataset import Dataset
from model_forge.model.model_evaluator import ModelEvaluator
from model_forge.model.model_orchastrator import ModelOrchestrator

from evaluate.shap_eval import ShapEval
from data_pipeline.pipelinesteps import data_splitter, load_data
from visualisation_forge.make_plots import MakePlots

from constants import METRICS

mlflow.set_tracking_uri("http://127.0.0.1:5000/")


@hydra.main(config_path="../conf", config_name="config.yaml", version_base=None)
def main(cfg):
    # Get the relative path of the file
    file_path = Path(os.path.abspath(__file__))
    base_path = file_path.parent.parent

    # Start an MLflow run
    commit_message = (
        subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).decode().strip()
    )

    with mlflow.start_run(experiment_id="107882983913598320", run_name=commit_message):
        # Create dataset
        dataset = Dataset.create_from_pipeline(
            partial(load_data, base_path),
            instantiate(cfg.data_pipeline),
            data_splitter=data_splitter,
            target_column="target",
            splits_columns=["train", "test", "OOT"],
        )

        model_orchestrator = ModelOrchestrator(cfg)
        model = model_orchestrator.create_pipeline()
        # Remove the directory with everything in it
        if os.path.exists("model"):
            shutil.rmtree("model")
        mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.save_model(model, "model")

        # Use cross evaluation to evaluate the model on training data
        logger.info("Creating a cross-validation evaluator for the model.")

        cv_eval = ModelEvaluator(metrics=METRICS, cv=5)
        cv_eval_results = cv_eval.evaluate(model, dataset.X_train, dataset.y_train)
        for metric, values in cv_eval_results.items():
            if (m := metric.removeprefix("test_")) in METRICS:
                mlflow.log_metric(f"cv_{m}", np.mean(values))

        logger.info("Creating a shap plots for the model.")
        model.fit(dataset.X_train, dataset.y_train)
        # Create a SHAP explainer
        shap_eval = ShapEval(
            model[-1],
            model.transform_without_predictor(dataset.X_test[:1000]),
            base_path,
            num_of_features=10,
        )
        shap_eval.create_shap_insights()

        mlflow.log_artifact(base_path / "artifact_storage/model_evaluation/")

        folder = "Plots"
        plot_maker = MakePlots()
        for split_name, (X, y) in dataset:
            plot_maker.make_and_write_plots(
                folder=folder,
                X=X,
                y=y,
                pred=model.predict(X),
                pred_proba=model.predict_proba(X),
                split_name=split_name,
            )
        mlflow.log_artifact(folder)


if __name__ == "__main__":
    main()
