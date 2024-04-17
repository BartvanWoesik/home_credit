from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from datetime import datetime

from sklearn.model_selection import ShuffleSplit


def read_data(file_path: str):
    df = pd.read_feather(file_path)
    df = df.set_index("case_id")
    return df


def drop_redundant_columns(df: pd.DataFrame, columns: list):
    return df.drop(columns=columns)


def get_age(df: pd.DataFrame, column: str):
    """
    Calculates the age based on a given date column in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the date column.
        column (str): The name of the date column.

    Returns:
        pd.DataFrame: The DataFrame with an additional 'Age' column representing the age in days.
    """
    df[column] = df[column].fillna("1970-01-01")
    df[column] = pd.to_datetime(df[column])
    df["Age"] = (pd.to_datetime("today") - df[column]).dt.days
    return df


def change_missing(df: pd.DataFrame, columns: list):
    for column in columns:
        df[column] = df[column].fillna("missing")
    return df


def load_data(file_path: Path) -> pd.DataFrame:
    return pd.read_feather(
        file_path / "data/parquet_files/train/processed_train.feather"
    )


def data_splitter(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits=2,
    random_state: int = 36,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Split the data into train, test, and out-of-time (OOT) sets.

    Parameters:
        X (pd.DataFrame): The input features.
        y (pd.DataFrame): The target variable.
        n_splits (int): The number of splits for cross-validation. Default is 2.
        random_state (int): The random seed for reproducibility. Default is 36.

    Returns:
        dict: A dictionary containing the train, test, and OOT sets.
            - "train": A tuple of the training features and target.
            - "test": A tuple of the testing features and target.
            - "oot": A tuple of the out-of-time features and target.
    """
    X = X.reset_index()
    oot_range = list(X[round(0.85 * X.shape[0]) : X.shape[0]].index)

    splitter = ShuffleSplit(test_size=0.2, n_splits=n_splits, random_state=random_state)
    row_in_train_test = ~X.index.isin(oot_range)
    split = splitter.split(X[row_in_train_test], y[row_in_train_test])
    train_ind, test_ind = next(split)

    return {
        "train": (X.iloc[train_ind], y.iloc[train_ind]),
        "test": (X.iloc[test_ind], y.iloc[test_ind]),
        "oot": (X.iloc[oot_range], y.iloc[oot_range]),
    }


def add_oot_indixes(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an Out-of-Time (OOT) indicator column to the input DataFrame.

    Parameters:
    X (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The input DataFrame with an additional 'OOT' column.

    """
    split_date = datetime.strptime("2020-04-01", "%Y-%m-%d")
    X["date_decision"] = pd.to_datetime(X["date_decision"])
    X["OOT"] = [1 if d > split_date else 0 for d in X["date_decision"]]
    return X


def add_train_test_indixes(
    X: pd.DataFrame, random_state: int = 42, test_size: int = 0.2
) -> pd.DataFrame:
    """
    Adds train and test indexes to the given DataFrame.

    Parameters:
    - X (pd.DataFrame): The input DataFrame.
    - random_state (int): Random seed for reproducibility. Default is 42.
    - test_size (int): Proportion of the data to be used for testing. Default is 0.2.

    Returns:
    - pd.DataFrame: The input DataFrame with train and test indexes added.
    """

    splitter = ShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
    split = splitter.split(X)
    train_ind, test_ind = next(split)
    X.loc[train_ind, "train"] = 1
    X.loc[test_ind, "test"] = 1

    return X
