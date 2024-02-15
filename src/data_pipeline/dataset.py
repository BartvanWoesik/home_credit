import pandas as pd
import numpy as np
from typing import Any, Optional, Callable, Union

class Dataset(dict):
    def __init__(
        self,
        data: pd.DataFrame,
        data_splitter=None,
        target_column: str = "y",
        name: str = "dataset",
    ) -> None:
        self.data_splitter = data_splitter
        self.target_column = target_column
        self.name = name
        self._is_data_splitted = False
        self.data = data
        self._X = self.data.drop(columns=[self.target_column])
        self._y = self.data[self.target_column]
        self._split_data()
        super().__init__(self.splits)

    @property
    def X(self) -> pd.DataFrame:
        return self._X

    @property
    def y(self) -> np.array:
        return self._y

    @property
    def columns(self):
        return list(self.splits.values())[0][0].columns.tolist()

    def _split_data(self) -> None:
        self._is_data_splitted = True
        if self.data_splitter is None:
            self.splits = {"all_data": (self.X, self.y)}
        else:
            self.splits = self.data_splitter(self.X, self.y)
        self._run_checks()

    def _run_checks(self) -> None:
        for split_name, (X, y) in self.splits.items():
            assert X is not None
            # check_empty_df(X)
            assert (
                X.columns.tolist() == self.columns
            ), f"Columns of split '{split_name}' do not match columns of split"

    def __getattr__(self, __name: str) -> Any:
        if __name.startswith(("X_", "y_")):
            _, split_name = __name.split("_", 1)
            if split_name in self.splits.keys():
                return (
                    self.splits[split_name][0]
                    if __name.startswith("X_")
                    else self.splits[split_name][1]
                )
        raise AttributeError(f"Attribute '{__name}' not found")

    def load_split(
        self,
        split: str,
        return_X_y: bool = False,
        sample_n_rows: Optional[int] = None,
        random_state: int = 36,
    ) -> Union[tuple[pd.DataFrame, np.array], pd.DataFrame]:
        if not self._is_data_splitted:
            self._split_data()
        if split not in self.splits.keys():
            raise ValueError(
                f"Invalid Split: You requested split '{split}'. Valid splits are: {*list(self.splits.keys()),} "
            )
        X, y = self.splits[split][0], self.split[split][1]
        if sample_n_rows is not None:
            X = X.sample(sample_n_rows, random_state=random_state)
            y = y[X.index]

        if return_X_y:
            return X, y.rename(self.target_column)
        else:
            return X.assign(**{self.target_column: y})

    def load_train_test(
        self,
        train_split: str = "train",
        test_split: str = "test",
        sample_n_rows: Optional[int] = None,
        random_state: int = 36,
    ):
        X_train, y_train = self.load_split(
            split=train_split,
            return_X_y=True,
            sample_n_rows=sample_n_rows,
            random_state=random_state,
        )
        X_test, y_test = self.load_split(split=test_split, return_X_y=True)
        return X_train, X_test, y_train, y_test

    @classmethod
    def create_from_pipeline(
        cls,
        data_loading_function: Callable[[], pd.DataFrame],
        data_pipeline=None,
        data_splitter=None,
        target_column="y",
        name: str = "dataset",
    ):
        data = data_loading_function()
        if data_pipeline:
            data = data_pipeline.apply(data)
        return cls(
            data=data,
            data_splitter=data_splitter,
            target_column=target_column,
            name=name,
        )

    @classmethod
    def create_from_splits(
        cls,
        splits: dict[str, tuple[pd.DataFrame, np.array]],
        name: str = "dataset",
        target_column: str = "y"
    ):
        Xs = []
        for split_name, (X, y) in splits.items():
            assert (
                target_column not in X.columns
            ), f"Split {split_name} already has a target column ({target_column}), please drop or rename"
            Xs.append(X.assign(y=y))
        fullX = pd.concat(Xs, ignore_index=True)

        dataset = cls(
            data=fullX, data_splitter=None, target_column=target_column, name=name
        )
        dataset._is_data_splitted = True
        dataset.splits = splits
        dataset._run_checks()
        return dataset