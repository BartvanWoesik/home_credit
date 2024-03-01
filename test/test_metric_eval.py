import numpy as np
import unittest
from sklearn.linear_model import LogisticRegression

from evaluate.metric_eval import ModelEvaluator


class ModelEvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 1, 1], [1, 1, 1], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
        self.y = np.array([0, 0, 1, 1, 1])
        self.model = LogisticRegression()
        self.metrics = ["accuracy", "precision", "recall"]
        self.evaluator = ModelEvaluator(self.model, self.metrics)

    def test_evaluate(self):
        expected_results = {
            "accuracy": [1.0, 1.0],
            "precision": [1.0, 1.0],
            "recall": [1.0, 1.0],
            "gini": [1.0, 1.0],
        }

        results = self.evaluator.evaluate(self.X, self.y, cv=2)
        print(results)

        self.assertEqual(
            results["test_accuracy"].tolist(), expected_results["accuracy"]
        )
        self.assertEqual(
            results["test_precision"].tolist(), expected_results["precision"]
        )
        self.assertEqual(results["test_recall"].tolist(), expected_results["recall"])
        self.assertEqual(results["test_gini"].tolist(), expected_results["gini"])


if __name__ == "__main__":
    unittest.main()
