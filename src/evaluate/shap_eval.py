import shap
import matplotlib.pyplot as plt
import os
from sklearn.pipeline import Pipeline

from random import randint
from my_logger.custom_logger import logger


class ShapEval:
    def __init__(self, model: Pipeline, X_data, path, num_of_features: int) -> None:
        self.model = model
        self.shap_values = self.create_explainer(self.model, X_data)
        self.path = path / "artifact_storage" / "model_evaluation"
        logger.info(f"Creating directory {self.path}")
        os.makedirs(self.path, exist_ok=True)

        self.num_of_features = num_of_features

        # os.path.abspath(os.path.join(os.getcwd(),'artifact_storage', 'model_evaluation'))

    def create_shap_insights(self):
        """
        Creates SHAP insights by calling the following methods:
        - create_shap_beeswarm: creates a beeswarm plot of SHAP values
        - create_heatmap: creates a heatmap of SHAP values
        - create_waterfall: creates a waterfall plot of SHAP values
        """

        self.create_shap_beeswarm(self.num_of_features)
        self.create_heatmap()
        self.create_waterfall()

    def create_explainer(self, model: Pipeline, X_data):
        """
        Function to create a shap explainer. The input parameters are the model that is trained and from which you want to
        test the effect of the different features.
        Args:
            num_of_features (int): This argument specifies the number of features to be displayed in the beeswarm plot. By default, it is set to 10. You can modify this value to control the number of features shown in the plot.

            show (bool): This argument determines whether the beeswarm plot should be displayed or saved as an image file. If show is set to True, the plot will be displayed. If show is set to False, the plot will be saved as a file named "beeswarm.png" in the current directory.

        """
        logger.info("Creating SHAP explainer")
        explainer = shap.TreeExplainer(model)
        logger.info("Creating SHAP values")
        shap_values = explainer(X_data)
        return shap_values

    def create_shap_beeswarm(self, num_of_features: int = 10, show: bool = False):
        """
        Creates a beeswarm plot using SHAP values.

        Parameters:
        - show (bool): If True, the beeswarm plot will be displayed.
                If False, the plot will be saved as "beeswarm.png" in the current directory.
        """
        _ = shap.plots.beeswarm(
            self.shap_values, max_display=num_of_features, show=False
        )
        self.fig_handler(show, "beeswarm.png")

    def create_heatmap(self, show: bool = False):
        """
        Creates a heatmap plot using SHAP values.

        Parameters:
        - show (bool): If True, the heatmap plot will be displayed.
                If False, the plot will be saved as "heatmap.png" in the current directory.
        """

        _ = shap.plots.heatmap(self.shap_values.sample(1000), show=False)
        self.fig_handler(show, "heatmap.png")

    def create_waterfall(self, show: bool = False):
        """
        Creates a waterfall plot using SHAP values.

        Parameters:
        - show (bool): If True, the waterfall plot will be displayed.
                If False, the plot will be saved as "waterfall.png" in the current directory.
        """

        row = randint(0, len(self.shap_values) - 1)
        _ = shap.plots.waterfall(self.shap_values[row], show=False)
        self.fig_handler(show, "waterfall.png")

    def fig_handler(self, show: bool = False, name: str = "plot.png"):
        logger.info(f"Saving plot as {name} in {self.path}")
        plt.title(name.split(".")[0])
        if show:
            plt.show()
        else:
            plt.savefig(self.path / name, bbox_inches="tight")
            plt.close()
