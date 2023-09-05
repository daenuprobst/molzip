
import numpy as np
import random
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
)
from sklearn.utils.class_weight import compute_class_weight
from gzip_classifier import classify
from gzip_regressor import regress, cross_val_and_fit_kernel_ridge,predict_kernel_ridge_regression
from gzip_utils import *

random.seed(666)


def benchmark(configs: List[Dict[str, Any]]) -> None:
    results = []

    for config in configs:
        loader = molnet_loader

        if config["dataset"] in ["schneider"]:
            loader = schneider_loader

        run_results = []
        for _ in range(config["n"]):
            tasks, X_train, y_train, X_valid, y_valid, X_test, y_test = loader(
                config["dataset"],
                splitter=config["splitter"],
                preproc=config["preprocess"],
                reload=False,
                transformers=[],
            )

            if config["augment"] > 0:
                X_train, y_train = augment(X_train, y_train, config["augment"])

            if config["sub_sample"] > 0.0:
                X_train, y_train = sub_sample(X_train, y_train, config["sub_sample"])

            if config["task"] == "classification":
                # Get class weights
                class_weights = []
                if config["is_imbalanced"]:
                    for y_task in y_train.T:
                        class_weights.append(
                            compute_class_weight(
                                "balanced", classes=sorted(list(set(y_task))), y=y_task
                            )
                        )

                # Run classification
                valid_preds = classify(
                    X_train, y_train, X_valid, config["k"], class_weights
                )
                test_preds = classify(
                    X_train, y_train, X_test, config["k"], class_weights
                )

                # Compute metrics
                try:
                    valid_auroc = roc_auc_score(y_valid, valid_preds, multi_class="ovr")
                except:
                    valid_auroc = 0.0

                valid_f1 = f1_score(
                    y_valid,
                    valid_preds,
                    average="micro",
                )

                try:
                    test_auroc = roc_auc_score(y_test, test_preds, multi_class="ovr")
                except:
                    test_auroc = 0.0

                test_f1 = f1_score(
                    y_test,
                    test_preds,
                    average="micro",
                )

                print(f"\n{config['dataset']} ({len(tasks)} tasks)")
                print(config)
                print(
                    f"Valid AUROC: {valid_auroc}, Valid F1: {valid_f1} , Test AUROC: {test_auroc}, Test F1: {test_f1}"
                )

                run_results.append([valid_auroc, valid_f1, test_auroc, test_f1])
            else:
                if config["task"] == "regression_knn":
                    
                    valid_preds = regress(X_train, y_train, X_valid, config["k"])
                    test_preds = regress(X_train, y_train, X_test, config["k"])

                elif config["task"] == "regression_krr":
                    best_alpha, best_gamma, best_lambda_, best_score = cross_val_and_fit_kernel_ridge(X_train, y_train, config["kfold"], config["gammas"], config["lambdas"])
                    #print only best gamma and lambda
                    print(f"Best gamma: {best_gamma}, Best lambda: {best_lambda_}")
                    valid_preds = predict_kernel_ridge_regression(X_train, X_valid, best_alpha, best_gamma)
                    test_preds = predict_kernel_ridge_regression(X_train, X_test, best_alpha, best_gamma)
                else:
                    raise ValueError(f"Unknown task {config['task']}")

                # Compute metrics
                valid_rmse = mean_squared_error(y_valid, valid_preds, squared=False)
                valid_mae = mean_absolute_error(
                    y_valid,
                    valid_preds,
                )
                test_rmse = mean_squared_error(y_test, test_preds, squared=False)
                test_mae = mean_absolute_error(
                    y_test,
                    test_preds,
                )

                print(f"\n{config['dataset']} ({len(tasks)} tasks)")
                print(config)
                print(
                    f"Valid RMSE: {valid_rmse}, Valid MAE: {valid_mae} , Test RMSE: {test_rmse}, Test MAE: {test_mae}"
                )

                run_results.append([valid_rmse, valid_mae, test_rmse, test_mae])

        run_results = np.array(run_results)
        results_means = np.mean(run_results, axis=0)
        results_stds = np.std(run_results, axis=0)

        results.append(
            (
                config,
                {
                    "valid_auroc": f"{round(results_means[0], 3)} +/- {round(results_stds[0], 3)}",
                    "valid_f1": f"{round(results_means[1], 3)} +/- {round(results_stds[0], 3)}",
                    "test_auroc": f"{round(results_means[2], 3)} +/- {round(results_stds[0], 3)}",
                    "test_f1": f"{round(results_means[3], 3)} +/- {round(results_stds[0], 3)}",
                },
            )
        )

        write_table(results)


def main():
    
    benchmark(
        [
             {
                 "dataset": "freesolv",
                 "splitter": "random",
                 "task": "regression_knn",
                 "k": 25,
                 "augment": 0,
                 "preprocess": False,
                 "sub_sample": 0.0,
                 "is_imbalanced": False,
                 "n": 4,
             },
             #{
             #    "dataset": "freesolv",
             #    "splitter": "random",
             #    "task": "regression_krr",
             #    "kfold": 5,
             #    "augment": 0,
             #    "gammas": np.logspace(-1, 3, 13),
             #    "lambdas": [1e-7, 1e-6, 1e-5],
             #    "preprocess": False,
             #    "sub_sample": 0.0,
             #    "is_imbalanced": False,
             #    "n": 4,
             #},
             {
                 "dataset": "delaney",
                 "splitter": "random",
                 "task": "regression_knn",
                 "k": 25,
                 "augment": 0,
                 "preprocess": False,
                 "sub_sample": 0.0,
                 "is_imbalanced": False,
                 "n": 4,
             },
             {
                 "dataset": "lipo",
                 "splitter": "random",
                 "task": "regression_knn",
                 "k": 25,
                 "augment": 0,
                 "preprocess": False,
                 "sub_sample": 0.0,
                 "is_imbalanced": False,
                 "n": 4,
             },
             {
                 "dataset": "sider",
                 "splitter": "scaffold",
                 "task": "classification",
                 "k": 5,
                 "augment": 0,
                 "preprocess": False,
                 "sub_sample": 0.0,
                 "is_imbalanced": True,
                 "n": 1,
             },
             {
                 "dataset": "sider",
                 "splitter": "random",
                 "task": "classification",
                 "k": 5,
                 "augment": 0,
                 "preprocess": False,
                 "sub_sample": 0.0,
                 "is_imbalanced": True,
                 "n": 4,
             },
             {
                 "dataset": "bbbp",
                 "splitter": "scaffold",
                 "task": "classification",
                 "k": 5,
                 "augment": 0,
                 "preprocess": False,
                 "sub_sample": 0.0,
                 "is_imbalanced": True,
                 "n": 1,
             },
             {
                 "dataset": "bace_classification",
                 "splitter": "scaffold",
                 "task": "classification",
                 "k": 5,
                 "augment": 0,
                 "preprocess": False,
                 "sub_sample": 0.0,
                 "is_imbalanced": True,
                 "n": 1,
             },
             {
                 "dataset": "bace_classification",
                 "splitter": "random",
                 "task": "classification",
                 "k": 5,
                 "augment": 0,
                 "preprocess": False,
                 "sub_sample": 0.0,
                 "is_imbalanced": True,
                 "n": 4,
             },
             {
                 "dataset": "clintox",
                 "splitter": "scaffold",
                 "task": "classification",
                 "k": 5,
                 "augment": 0,
                 "preprocess": False,
                 "sub_sample": 0.0,
                 "is_imbalanced": True,
                 "n": 1,
             },
             {
                 "dataset": "clintox",
                 "splitter": "random",
                 "task": "classification",
                 "k": 5,
                 "augment": 0,
                 "preprocess": False,
                 "sub_sample": 0.0,
                 "is_imbalanced": True,
                 "n": 4,
             },
             {
                 "dataset": "tox21",
                 "splitter": "scaffold",
                 "task": "classification",
                 "k": 5,
                 "augment": 0,
                 "preprocess": False,
                 "sub_sample": 0.0,
                 "is_imbalanced": True,
                 "n": 1,
             },
             {
                 "dataset": "tox21",
                 "splitter": "random",
                 "task": "classification",
                 "k": 5,
                 "augment": 0,
                 "preprocess": False,
                 "sub_sample": 0.0,
                 "is_imbalanced": True,
                 "n": 4,
             },
             {
                 "dataset": "hiv",
                 "splitter": "scaffold",
                 "task": "classification",
                 "k": 5,
                 "augment": 0,
                 "preprocess": False,
                 "sub_sample": 0.0,
                 "is_imbalanced": True,
                 "n": 1,
             },
             {
                 "dataset": "muv",
                 "splitter": "random",
                 "task": "classification",
                 "k": 5,
                 "augment": 0,
                 "preprocess": False,
                 "sub_sample": 0.0,
                 "is_imbalanced": True,
                 "n": 4,
             },
            {
               "dataset": "schneider",
               "splitter": "random",
                "task": "classification",
                "k": 5,
                "augment": 0,
                "preprocess": True,
                "sub_sample": 0.0,
                "is_imbalanced": False,
                "n": 1,
            },
        ]
    )


if __name__ == "__main__":
    main()
