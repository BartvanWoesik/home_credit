import os

from pathlib import Path
from typing import List, Dict

import hydra
import polars as pl

from omegaconf import DictConfig
from hydra.utils import instantiate

from my_logger.custom_logger import logger


DATA_PATH = Path()
DATA_LOCATION = ""
BASE_FILE = ""
ID = "case_id"
TARGET = "target"
DATE_DECISION = "date_decision"


def create_agg_specs(cfg: list) -> dict:
    """
    Create aggregation specifications based on the given configuration.

    Args:
        cfg (dict): Configuration dictionary containing aggregation specifications.

    Returns:
        dict: Aggregation specifications.

    """
    aggregation_specs = {}
    for column in cfg:
        aggregation_specs[f"{column.base_feature_name}"] = instantiate(
            column.aggregation
        )(column.name)
    return aggregation_specs


def create_aggration_dataframe(cfg: dict, df: pl.DataFrame) -> pl.DataFrame:
    """
    Create an aggregated dataframe based on the given configuration and input dataframe.

    Args:
        cfg (dict): Configuration dictionary containing aggregation specifications.
        df (pl.DataFrame): Input dataframe to be aggregated.

    Returns:
        pl.DataFrame: Aggregated dataframe.

    """
    # Remove rows where event is after the decision
    if len(cfg.time_col) > 0:
        df = df.filter(pl.col(cfg.time_col[0]) < pl.col(DATE_DECISION))

    # Check if each column in cfg.agg_columns contains only null values
    cols_all_null = [
        col.base_feature_name
        for col in cfg.agg_columns
        if len(df[col.name].drop_nulls()) == 0
    ]

    agg_columns = [
        col for col in cfg.agg_columns if col.base_feature_name not in cols_all_null
    ]

    # Create aggregation specifications
    aggregation_specs = create_agg_specs(agg_columns)

    # Perform the aggregation
    logger.info("Start the aggregation")
    df_agg = df.group_by(ID).agg(**aggregation_specs)
    del df
    for col in cols_all_null:
        df_agg = df_agg.with_columns(placeholder=None)
        df_agg = df_agg.rename({"placeholder": col})
    return df_agg


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
    logger.info(f"Get ordered data files from {data_path}")
    all_files = os.listdir(data_path)
    return {
        source: [file for file in all_files if source in file and file is not None]
        for source in all_file_sources
    }


def read_file(cfg_columns: List, unique_cols: List, file: str) -> pl.DataFrame:
    """
    Read a parquet file and return a DataFrame.

    Args:
        cfg_columns (List): List of configuration columns.
        unique_cols (List): List of unique columns.
        file (str): Path to the parquet file.

    Returns:
        pl.DataFrame: DataFrame containing the data from the parquet file.
    """
    if len(cfg_columns.time_col) > 0:
        return pl.read_parquet(
            DATA_PATH / file,
            columns=[ID] + unique_cols + [cfg_columns.time_col[0]],
        )
    else:
        return pl.read_parquet(DATA_PATH / file, columns=[ID] + unique_cols)


def create_dataframe(cfg: DictConfig, split: str) -> pl.DataFrame:
    """
    Create a dataframe by reading and joining multiple files based on the provided configuration.

    Args:
        cfg (DictConfig): Configuration object containing data sources and columns.

    Returns:
        pl.DataFrame: The resulting dataframe.
    """

    # all_sources we want to use
    all_file_sources = list(cfg.data.keys())
    data_structure = get_orderd_data_files(DATA_PATH, all_file_sources=all_file_sources)

    # Only keep the ID and TARGET column
    if split == "train":
        df_base = pl.read_parquet(
            DATA_PATH / BASE_FILE, columns=[ID, TARGET, DATE_DECISION]
        )
    else:
        df_base = pl.read_parquet(DATA_PATH / BASE_FILE, columns=[ID, DATE_DECISION])
    df = df_base.clone()

    # Loop over all the files and join them to the base file
    for source, files in data_structure.items():
        cfg_columns = cfg.data[source]
        df_all_files = pl.DataFrame()
        unique_cols = list(set(col.name for col in cfg_columns.agg_columns))
        # Read splitted files and join them to the base file
        for file in files:
            df_all_files = pl.concat(
                [df_all_files, read_file(cfg_columns, unique_cols, file)]
            )

        df_combined = df_base.join(df_all_files, on=ID, how="left", validate="1:m")
        df = df.join(
            create_aggration_dataframe(cfg_columns, df_combined),
            on="case_id",
            how="left",
            validate="1:m",
        )

    # Write the dataframe to feather
    logger.info("Write df to feather")
    df.write_ipc(DATA_PATH / DATA_LOCATION)


@hydra.main(
    version_base=None, config_path="../../conf", config_name="raw_data_conf.yaml"
)
def main(cfg):
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
        create_dataframe(cfg, split)


if __name__ == "__main__":
    main()
