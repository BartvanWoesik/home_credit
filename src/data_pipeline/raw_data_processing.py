from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
import os
import hydra
from hydra.utils import instantiate
from my_logger.custom_logger import logger


DATA_PATH = Path()
DATA_LOCATION = ""
BASE_FILE = ""
ID = "case_id"
TARGET = "target"
DATE_DECISION = "date_decision"


def create_agg_specs(cfg: dict) -> dict:
    """
    Create aggregation specifications based on the given configuration.

    Args:
        cfg (dict): Configuration dictionary containing aggregation specifications.

    Returns:
        dict: Aggregation specifications.

    """
    aggregation_specs = {}
    for column in cfg:
        aggregation_specs[f"{column.base_feature_name}"] = (
            column.name,
            instantiate(column.aggregation),
        )
    return aggregation_specs


def create_aggration_dataframe(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Create an aggregated dataframe based on the given configuration and input dataframe.

    Args:
        cfg (dict): Configuration dictionary containing aggregation specifications.
        df (pd.DataFrame): Input dataframe to be aggregated.

    Returns:
        pd.DataFrame: Aggregated dataframe.

    """
    # Remove rows where event is after the decision
    if len(cfg.time_col) > 0:
        df = df[df[cfg.time_col[0]] < df[DATE_DECISION]]

    # Create aggregation specifications
    aggregation_specs = create_agg_specs(cfg.agg_columns)

    # Perform the aggregation
    logger.info("Start the aggregation")
    return df.groupby(ID).agg(**aggregation_specs).reset_index()


def get_orderd_data_files(data_path: Path, all_file_sources):
    file_dict = []
    all_files = os.listdir(data_path)
    file_dict = [
        {source: [file for file in all_files if source in file and file is not None]}
        for source in all_file_sources
    ]

    return file_dict


@hydra.main(config_path="../../", config_name="config.yaml")
def create_dataframe(cfg: DictConfig) -> pd.DataFrame:
    # all_sources we want to use
    all_file_sources = list(cfg.data.keys())
    print(type(all_file_sources))
    data_structure = get_orderd_data_files(DATA_PATH, all_file_sources=all_file_sources)
    logger.info(f"Data structure: {data_structure}")
    # Read base file where we can join the other files on
    df_base = pd.read_parquet(DATA_PATH / BASE_FILE)

    # Only keep the ID and TARGET column
    if TARGET in df_base.columns and ID in df_base.columns:
        df = pd.DataFrame(df_base[[ID, TARGET]])
    else:
        df = pd.DataFrame(df_base[[ID]])

    # Loop over all the files and join them to the base file
    for source_file in data_structure:
        logger.info(f"source_file: {source_file}")
        source_name = list(source_file.keys())[0]
        files = list(source_file.values())[0]
        cfg_columns = cfg.data[source_name]
        df_all_files = pd.DataFrame()
        for file in files:
            print(file)
            unique_cols = list(set([col.name for col in cfg_columns.agg_columns]))
            if len(cfg_columns.time_col) > 0:
                df_file = pd.read_parquet(
                    DATA_PATH / file,
                    columns=[ID] + unique_cols + [cfg_columns.time_col[0]],
                )
            else:
                df_file = pd.read_parquet(DATA_PATH / file, columns=[ID] + unique_cols)
            df_all_files = pd.concat([df_all_files, df_file])
        df_combined = df_base.merge(
            df_all_files, on=ID, how="left", validate="one_to_many"
        )
        logger.info(f"columns in combined dataset: {df_combined.columns}")
        df_agg = create_aggration_dataframe(cfg_columns, df_combined)
        logger.info(f"columns in base df: {df.columns}")
        df = df.merge(df_agg, on="case_id", how="left", validate="one_to_many")

    # Write the dataframe to feather
    print("Write df to feather")
    df.to_feather(DATA_PATH / DATA_LOCATION)


def main():
    global DATA_PATH, DATA_LOCATION, BASE_FILE
    for split in ["train", "test"]:
        logger.info(f"Processing {split} data")
        DATA_PATH = Path(os.getcwd()) / f"data/parquet_files/{split}"
        DATA_LOCATION = f"processed_{split}.feather"
        BASE_FILE = f"{split}_base.parquet"
        create_dataframe()


if __name__ == "__main__":
    main()
