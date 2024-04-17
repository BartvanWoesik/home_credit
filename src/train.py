import os
import mlflow
import mlflow.sklearn
import numpy as np
from model_forge.data.dataset import Dataset
from data_pipeline.pipelinesteps import data_splitter
from hydra.utils import instantiate
import hydra
import subprocess
from pathlib import Path
from model_forge.model.model_orchastrator import ModelOrchestrator
from model_forge.model.model_evaluator import ModelEvaluator

from evaluate.shap_eval import ShapEval
from my_logger.custom_logger import logger

from data_pipeline.pipelinesteps import load_data
from evaluate.plots.density import plot_density
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt
from functools import partial
from evaluate.scorers import custom_scorers

mlflow.set_tracking_uri("http://127.0.0.1:5000/")


@hydra.main(config_path="../conf", config_name="config.yaml")
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

        mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.save_model(model, "model")

        # Use cross evaluation to evaluate the model on training data
        logger.info("Creating a cross-validation evaluator for the model.")
        metrics = ["roc_auc"]
        cv_eval = ModelEvaluator(
            sklearn_metrics=metrics, custom_scorers=custom_scorers, cv=5
        )
        cv_eval_results = cv_eval.evaluate(model, dataset.X_train, dataset.y_train)
        for metric, values in cv_eval_results.items():
            if (m := metric.removeprefix("test_")) in metrics:
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

        dens_path = "Density/"
        pr_path = "Precision-Recall/"
        for split_name, (X, y) in dataset.splits.items():
            # Create Precision-Recall curve
            display = PrecisionRecallDisplay.from_estimator(
                model, X, y, plot_chance_level=True
            )
            _ = display.ax_.set_title(f"2-class Precision-Recall curve - {split_name}")
            plt.savefig(
                base_path
                / "artifact_storage"
                / "plot"
                / pr_path
                / f"pr-{split_name}.jpeg"
            )
            mlflow.log_artifact(
                base_path
                / "artifact_storage"
                / "plot"
                / pr_path
                / f"pr-{split_name}.jpeg",
                artifact_path=pr_path[:-1],
            )

            plot_density(
                model.predict_proba(X),
                y,
                base_path / "artifact_storage" / "plot" / dens_path,
                f"density-{split_name}.jpeg",
                threshold=0.5,
            )
            mlflow.log_artifact(
                base_path
                / "artifact_storage"
                / "plot"
                / dens_path
                / f"density-{split_name}.jpeg",
                artifact_path=dens_path[:-1],
            )


if __name__ == "__main__":
    main()
