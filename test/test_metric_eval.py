import numpy as np
import unittest
from sklearn.linear_model import LogisticRegression

from evaluate.metric_eval import ModelEvaluator


class ModelEvaluatorTest(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.y = np.array([0, 1, 0])
        self.model = LogisticRegression()
        self.metrics = ["accuracy", "precision", "recall"]
        self.evaluator = ModelEvaluator(self.model, self.metrics)

    def test_evaluate(self):
        expected_results = {
            "accuracy": [1.0, 1.0, 1.0, 1.0, 1.0],
            "precision": [1.0, 1.0, 1.0, 1.0, 1.0],
            "recall": [1.0, 1.0, 1.0, 1.0, 1.0],
        }

        results = self.evaluator.evaluate(self.X, self.y, cv=5)

        self.assertEqual(set(results.keys()), set(self.metrics))
        for metric in self.metrics:
            self.assertEqual(results[metric].tolist(), expected_results[metric])


if __name__ == "__main__":
    unittest.main()
