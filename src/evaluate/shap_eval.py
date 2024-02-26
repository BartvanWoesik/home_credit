import shap
import matplotlib.pyplot as plt
import os
from sklearn.pipeline import Pipeline

from random import randint
from pathlib import Path


class ShapEval:
    def __init__(self, model: Pipeline, X_data, num_of_features: int) -> None:
        self.model = model
        self.shap_values = self.create_explainer(self.model, X_data)
        self.path = Path(os.getcwd()) / "artifact_storage" / "model_evaluation"
        self.num_of_features = num_of_features

        # os.path.abspath(os.path.join(os.getcwd(),'artifact_storage', 'model_evaluation'))

    def create_explainer(self, model: Pipeline, X_data):
        """Function to create a shap explainer. The input parameters are the model that is trained and from which you want to
        test the effect of the different features. X_data is a parameter that represents the training data on which the model is trained.
        """

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_data)
        return shap_values

    def create_shap_beeswarm(self, num_of_features: int = 10, show: bool = False):
        """Function to create a shap beeswarm plot. The input parameters are the model that is trained and from which you want to
        test the effect of the different features. X_data is a parameter that represents the training data on which the model is trained.
        At last the num_of_features parameter represents the number of features you want to show in the beeswarm plot.
        """
        _ = shap.plots.beeswarm(
            self.shap_values, max_display=num_of_features, show=False
        )
        self.fig_handler(show, "beeswarm.png")

    def create_heatmap(self, show: bool = False):
        _ = shap.plots.heatmap(self.shap_values.sample(1000), show=False)
        self.fig_handler(show, "heatmap.png")

    def create_waterfall(self, show: bool = False):
        row = randint | (0, len(self.shap_values) - 1)
        _ = shap.plots.waterfall(self.shap_values[row], show=False)
        self.fig_handler(show, "waterfall.png")

    def fig_handler(self, show: bool = False, name: str = "plot.png"):
        if show:
            plt.show()
        else:
            plt.savefig(self.path / name)
