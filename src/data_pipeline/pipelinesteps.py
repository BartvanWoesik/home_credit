from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

def create_one_hot_encoder(df: pd.DataFrame, columns):
    return pd.get_dummies(df, columns=columns)

def read_data(file_path: str):
    return pd.read_feather(file_path)

def drop_redundant_columns(df: pd.DataFrame, columns: list):
    return df.drop(columns=columns)

from sklearn.model_selection import ShuffleSplit
from typing import Dict, Tuple


def data_splitter(
    X: pd.DataFrame,
    y: pd.DataFrame,
    n_splits=2,
    random_state: int = 36,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
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