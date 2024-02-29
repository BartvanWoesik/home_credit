from omegaconf import DictConfig
from pathlib import Path
import pandas as pd
import os
import hydra
from hydra.utils import instantiate
from typing import List, Dict
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


def get_orderd_data_files(data_path: Path, all_file_sources) -> Dict[str, List[str]]:
    """
    Retrieve a dictionary of ordered data files based on the given data path and file sources.

    Args:
        data_path (Path): The path to the directory containing the data files.
        all_file_sources (list): A list of file sources to filter the data files.

    Returns:
        list: A list of dictionaries, where each dictionary contains a file source as the key and a list of corresponding
        data files as the value.
    """
    all_files = os.listdir(data_path)
    return {
        source: [file for file in all_files if source in file and file is not None]
        for source in all_file_sources
    }


def read_file(cfg_columns: List, unique_cols: List, file: str) -> pd.DataFrame:
    """
    Read a parquet file and return a DataFrame.

    Args:
        cfg_columns (List): List of configuration columns.
        unique_cols (List): List of unique columns.
        file (str): Path to the parquet file.

    Returns:
        pd.DataFrame: DataFrame containing the data from the parquet file.
    """
    if len(cfg_columns.time_col) > 0:
        return pd.read_parquet(
            DATA_PATH / file,
            columns=[ID] + unique_cols + [cfg_columns.time_col[0]],
        )
    else:
        return pd.read_parquet(DATA_PATH / file, columns=[ID] + unique_cols)


@hydra.main(config_path="../../", config_name="config.yaml")
def create_dataframe(cfg: DictConfig, split: str) -> pd.DataFrame:
    """
    Create a dataframe by reading and joining multiple files based on the provided configuration.

    Args:
        cfg (DictConfig): Configuration object containing data sources and columns.

    Returns:
        pd.DataFrame: The resulting dataframe.
    """

    # all_sources we want to use
    all_file_sources = list(cfg.data.keys())
    data_structure = get_orderd_data_files(DATA_PATH, all_file_sources=all_file_sources)

    # Only keep the ID and TARGET column
    if split == "train":
        df_base = pd.read_parquet(
            DATA_PATH / BASE_FILE, columns=[ID, TARGET, DATE_DECISION]
        )
    else:
        df_base = pd.read_parquet(DATA_PATH / BASE_FILE, columns=[ID, DATE_DECISION])
    df = df_base.copy()

    # Loop over all the files and join them to the base file
    for source, files in data_structure.items():
        cfg_columns = cfg.data[source]
        df_all_files = pd.DataFrame()
        unique_cols = list(set(col.name for col in cfg_columns.agg_columns))
        # Read splitted files and join them to the base file
        for file in files:
            df_all_files = pd.concat(
                [df_all_files, read_file(cfg_columns, unique_cols, file)]
            )

        df_combined = df_base.merge(
            df_all_files, on=ID, how="left", validate="one_to_many"
        )
        df = df.merge(
            create_aggration_dataframe(cfg_columns, df_combined),
            on="case_id",
            how="left",
            validate="one_to_many",
        )

    # Write the dataframe to feather
    logger.info("Write df to feather")
    df.to_feather(DATA_PATH / DATA_LOCATION)


def main():
    """
    This function is the entry point of the raw data processing pipeline.
    It processes the train and test data by creating dataframes and saving them in the specified locations.
    """
    global DATA_PATH, DATA_LOCATION, BASE_FILE
    for split in ["train", "test"]:
        logger.info(f"Processing {split} data")
        DATA_PATH = Path(os.getcwd()) / f"data/parquet_files/{split}"
        DATA_LOCATION = f"processed_{split}.feather"
        BASE_FILE = f"{split}_base.parquet"
        create_dataframe(split)


if __name__ == "__main__":
    main()
