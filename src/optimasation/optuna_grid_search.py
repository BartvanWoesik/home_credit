"""
Optuna example that optimizes a classifier configuration for Iris dataset using sklearn.

In this example, we optimize a classifier configuration for Iris dataset. Classifiers are from
scikit-learn. We optimize both the choice of classifier (among SVC and RandomForest) and their
hyperparameters.

"""

import optuna
from model.modelorchastrator import ModelOrchestrator
from data_pipeline.dataset import Dataset


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial):
    # Create dataset
    dataset = Dataset.create_from_pipeline(
            partial(load_data, base_path), 
            instantiate(cfg.data_pipeline),
            data_splitter=data_splitter,
            target_column="target",
    )


    
    model_orchestrator = ModelOrchestrator(cfg)
    model = model_orchestrator.modelpipeline



    # classifier_name = trial.suggest_categorical("classifier", ["SVC", "RandomForest"])
    # if classifier_name == "SVC":
    #     svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
    #     classifier_obj = sklearn.svm.SVC(C=svc_c, gamma="auto")
    # else:
    #     rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
    #     classifier_obj = sklearn.ensemble.RandomForestClassifier(
    #         max_depth=rf_max_depth, n_estimators=10
    #     )

    # score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=3)
    # accuracy = score.mean()
    # return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print(study.best_trial)