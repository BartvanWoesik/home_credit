from sklearn.model_selection import ShuffleSplit
from typing import Dict, Tuple
import pandas as pd
from pathlib import Path

def read_data(file_path: str):
    df = pd.read_feather(file_path)
    df = df.set_index('case_id')
    return df


def drop_redundant_columns(df: pd.DataFrame, columns: list):
    return df.drop(columns=columns)


def load_data(file_path: Path) -> pd.DataFrame:
    return pd.read_feather(file_path / "data/parquet_files/train/processed_train.feather")

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
