from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
import pandas as pd


class FeatureFilter(BaseEstimator, TransformerMixin):
    """
    A feature filter that selects only the specified features from a dataset.
    """

    def __init__(self, features=None) -> None:
        """
        Initialize the FeatureFilter.

        Parameters:
        features (list, optional): The list of features to select. Defaults to None.
        """
        self.features = features if features is not None else []

    def transform(self, X):
        """
        Transforms the input dataset by selecting only the features specified in the feature selector.

        Parameters:
        X (pandas.DataFrame): The input dataset to be transformed.

        Returns:
        pandas.DataFrame: The transformed dataset with only the selected features.
        """
        return X[list(self.features)]

    def fit_transform(self, X, *fit_args, **fit_kwargs):
        """
        Fit the feature selector on the training data and transform the input data.

        Parameters:
        X (array-like): The input data to be transformed.

        Returns:
        array-like: The transformed data.
        """
        return self.transform(X)

    def fit(self, X, *fit_args, **fit_kwargs):
        """
        Fits the feature selector to the input data.

        Parameters:
        X (array-like): The input data to fit the feature selector on.
        *fit_args: Additional positional arguments to be passed to the fit method.
        **fit_kwargs: Additional keyword arguments to be passed to the fit method.

        Returns:
        self: The fitted feature selector object.
        """
        return self


class FeatureSelector(SelectFromModel):
    def transform(self, X):
        """
        Transforms the input data using the feature selector.

        Parameters:
        X (array-like): The input data to be transformed.

        Returns:
        DataFrame: The transformed data as a pandas DataFrame.
        """
        X = super().transform(X)
        df = pd.DataFrame(X, columns=self.get_feature_names_out())
        return df
