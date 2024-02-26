import os
import mlflow
import mlflow.sklearn
import pandas as pd
from data_pipeline.dataset import Dataset
from data_pipeline.pipelinesteps import data_splitter
from hydra.utils import instantiate
import hydra
import subprocess
from pathlib import Path
from model.modelorchastrator import ModelOrchestrator

mlflow.set_tracking_uri("http://127.0.0.1:5000/")


@hydra.main(config_path="../", config_name="config.yaml")
def main(cfg):
    # Get the relative path of the file
    file_path = Path(os.path.abspath(__file__))
    base_path = file_path.parent.parent
    print("Base path:", base_path)
    # Start an MLflow run
    commit_message = (
        subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).decode().strip()
    )
    with mlflow.start_run(experiment_id="264751226398184019", run_name=commit_message):
        # Create dataset
        data_pipeline = instantiate(cfg.data_pipeline)

        df = pd.read_feather(
            base_path / "data/parquet_files/train/processed_train.feather"
        )
        df = data_pipeline.apply(df)

        dataset = Dataset(data=df, data_splitter=data_splitter, target_column="target")

        model_orchestrator = ModelOrchestrator(cfg.model)

        model = model_orchestrator.modelpipeline

        # Train your model
        model.fit(dataset.X_train, dataset.y_train)

        mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.save_model(model, "model")

        predictions = model.predict_proba(dataset.X_test)
        df_predictions = pd.DataFrame(
            {"case_id": dataset.X_test["case_id"], "predictions": predictions.T[1]}
        )
        output_dir = base_path / "artifact_storage/predictions"
        os.makedirs(output_dir, exist_ok=True)

        df_predictions.to_csv(output_dir / "predictions.csv")

        mlflow.log_artifact(output_dir / "predictions.csv")

        # Print the run ID
        print("MLflow run ID:", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    main()
