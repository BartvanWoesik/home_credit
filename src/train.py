import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from data_pipeline.dataset import Dataset
from data_pipeline.pipelinesteps import data_splitter
from hydra.utils import instantiate
import hydra
import subprocess
from pathlib import Path
from model.modelorchastrator import ModelOrchestrator
from evaluate.metric_eval import ModelEvaluator

from evaluate.shap_eval import ShapEval
from my_logger.custom_logger import logger

from data_pipeline.pipelinesteps import load_data

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
from functools import partial


@hydra.main(config_path="../", config_name="config.yaml")
def main(cfg):

    # Get the relative path of the file
    file_path = Path(os.path.abspath(__file__))
    base_path = file_path.parent.parent

    # Start an MLflow run
    commit_message = (
        subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).decode().strip()
    )

    with mlflow.start_run(experiment_id="264751226398184019", run_name=commit_message):
        
        # Create dataset
        dataset = Dataset.create_from_pipeline(
                partial(load_data, base_path), 
                instantiate(cfg.data_pipeline),
                data_splitter=data_splitter,
                target_column="target",
        )

        model_orchestrator = ModelOrchestrator(cfg)
        model = model_orchestrator.modelpipeline

        # Train your model
        model.fit(dataset.X_train, dataset.y_train)

        mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.save_model(model, "model")

        # Use cross evaluation to evaluate the model on training data
        logger.info("Creating a cross-validation evaluator for the model.")
        metrics = ["roc_auc_ovr", "f1", "gini", "kaggle"]
        cv_eval = ModelEvaluator(model[-1], metrics)
        cv_eval_results = cv_eval.evaluate(
            model.transform_without_predictor(dataset.X), dataset.y
        )
        for metric, values in cv_eval_results.items():
            print(metric)
            if metric.removeprefix("test_") in metrics:
                mlflow.log_metric(f"cv_{metric}", np.mean(values))

        output_dir = base_path / "artifact_storage/predictions"
        os.makedirs(output_dir, exist_ok=True)
        test_data = pd.read_feather(
            base_path / "data/parquet_files/test/processed_test.feather"
        )


        predictions = model.predict_proba(test_data.reset_index())
        df_predictions = pd.DataFrame(
            {"case_id": test_data["case_id"], "predictions": predictions.T[1]}
        )
        df_predictions.to_csv(output_dir / "predictions.csv", index=False)

        # Create a SHAP explainer
        shap_eval = ShapEval(
            model[-1],
            model.transform_without_predictor(dataset.X_train[:1000]),
            base_path,
            num_of_features=10,
        )
        shap_eval.create_shap_insights()

        mlflow.log_artifact(output_dir / "predictions.csv")
        mlflow.log_artifact(base_path / "artifact_storage/model_evaluation/")

        # Print the run ID
        print("MLflow run ID:", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()
