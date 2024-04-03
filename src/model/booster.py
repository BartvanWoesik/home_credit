from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance


class HistBooster(HistGradientBoostingClassifier):
    def fit(self, X, y):
        super().fit(X, y)
        self.feature_importances_ = self._calculate_feature_importances(X, y)

    def _calculate_feature_importances(self, X, y):
        results = permutation_importance(
            self, X, y, n_repeats=1, random_state=42, scoring="roc_auc", max_samples=0.5
        )

        return results.importances_mean
