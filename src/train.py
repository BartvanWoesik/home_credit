import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from data_pipeline.dataset import Dataset
from data_pipeline.pipelinesteps import data_splitter

@hydra.main(config_path="", config_name="config.yaml")
def main():


    # Start an MLflow run
    with mlflow.start_run():


        # Create dataset
        data_pipeline = instantiate(cfg.data_pipeline)
        df = pd.read_feather("../../data/parquet_files/train/processed_train.feather")
        df = data_pipeline.apply(df)

        dataset = Dataset(data=df, data_splitter=data_splitter, target_column='target')





        # Train your model
        model = HistGradientBoostingClassifier()
        model.fit(dataset['X_train'], dataset['y_train'])

        # Log the model
        mlflow.sklearn.log_model(model, 'model')

 

        # Save the model locally
        mlflow.sklearn.save_model(model, 'model')

        # Print the run ID
        print("MLflow run ID:", mlflow.active_run().info.run_id)

if __name__ == "__main__":
    main()