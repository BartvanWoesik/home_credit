from optbinning import BinningProcess
from sklearn.base import BaseEstimator, TransformerMixin


class OptBinningTransformer(BaseEstimator, TransformerMixin):
    """
    OptBinningTransformer applies optimal binning to the specified columns in a dataset.

    Parameters:
    ----------
    variable_names : list, optional
        List of column names to apply optimal binning to. If not provided, all columns will be considered.
    max_n_prebins : int, optional
        Maximum number of prebins to consider during the binning process. Default is 100.
    min_prebin_size : float, optional
        Minimum prebin size as a fraction of the total number of samples. Default is 0.1.
    random_state : int or None, optional
        Random seed for reproducibility. Default is None.

    Attributes:
    ----------
    variable_names : list
        List of column names to apply optimal binning to.
    max_n_prebins : int
        Maximum number of prebins to consider during the binning process.
    min_prebin_size : float
        Minimum prebin size as a fraction of the total number of samples.
    random_state : int or None
        Random seed for reproducibility.
    binning_process : BinningProcess
        Optimal binning process object.

    Methods:
    -------
    fit(X, y=None)
        Fit OptBinning on the specified columns.
    transform(X)
        Transform the specified columns using OptBinning.
    """

    def __init__(
        self,
        variable_names=None,
        max_n_prebins=100,
        min_prebin_size=0.1,
        random_state=None,
    ):
        self.variable_names = variable_names if variable_names is not None else []
        self.max_n_prebins = max_n_prebins
        self.min_prebin_size = min_prebin_size
        self.random_state = random_state
        self.binning_process = BinningProcess(
            variable_names=list(self.variable_names),
            max_n_prebins=self.max_n_prebins,
            min_prebin_size=self.min_prebin_size,
        )

    def fit(self, X, y=None):
        """
        Fit OptBinning on the specified columns.

        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.
        y : array-like, shape (n_samples,), optional
            Target variable. Default is None.

        Returns:
        -------
        self : OptBinningTransformer
            Returns self.
        """
        self.binning_process.fit(X[self.variable_names].values, y)
        return self

    def transform(self, X):
        """
        Transform the specified columns using OptBinning.

        Parameters:
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data.

        Returns:
        -------
        X_transformed : array-like, shape (n_samples, n_features)
            Transformed data.
        """
        X_transformed = X.copy()
        transformed_column = self.binning_process.transform(
            X[list(self.variable_names)].values
        )
        X_transformed[list(self.variable_names)] = transformed_column
        return X_transformed
