from typing import List, Dict, Any
import numpy as np
import random
from scipy.stats import pearsonr
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.utils.class_weight import compute_class_weight
from gzip_utils import (
    molnet_loader,
    schneider_loader,
    adme_loader,
    pdbbind_loader,
    local_molnet_loader,
    write_table,
)
from moleculenet_config_nok import get_all_tests

from molzip import (
    ZipRegressor,
    ZipClassifier,
    RDKitRegressor,
    RDKitClassifier,
    ZipClassifier_CV,
    ZipRegressor_CV,
)
from molzip.featurizers import ZipFeaturizer
from molzip.compressors import GzipCompressor


def benchmark(configs: List[Dict[str, Any]]) -> None:
    results = []

    regressor_CV, regressor = ZipRegressor_CV(), ZipRegressor()
    classifier_CV, classifier = ZipClassifier_CV(), ZipClassifier()

    for config in configs:
        print(f"Loading dataset {config['dataset']}...")
        is_k_set = "k" in config
        n_bins = config["bins"] if "bins" in config else 128
        zip_featurizer = ZipFeaturizer(n_bins)

        # loader = molnet_loader
        loader = local_molnet_loader

        if config["dataset"] in ["schneider"]:
            loader = schneider_loader

        if config["dataset"].startswith("adme"):
            loader = adme_loader

        if config["dataset"] == "pdbbind":
            loader = pdbbind_loader

        if "result_props" not in config:
            config["result_props"] = {}

        run_results = []
        for i_n in range(config["n"]):
            print(f"Running split {i_n}...")
            random.seed(i_n)
            tasks, X_train, y_train, X_valid, y_valid, X_test, y_test = loader(
                config["dataset"],
                splitter=config["splitter"],
                task_name=config["task_name"] if "task_name" in config else None,
                properties=config["properties"] if "properties" in config else None,
                reload=False,
                transformers=[],
            )

            if "transforms" in config:
                for transform in config["transforms"]:
                    X_train = transform(X_train)
                    X_valid = transform(X_valid)
                    X_test = transform(X_test)

            if config["task"] in ["classification", "classification_vec"]:
                # Get class weights
                class_weights = []
                if config["is_imbalanced"]:
                    for y_task in y_train.T:
                        class_weights.append(
                            compute_class_weight(
                                "balanced",
                                classes=np.array(sorted(list(set(y_task)))),
                                y=y_task,
                            )
                        )

                if config["task"] == "classification_vec":
                    X_train, nan_inds_train = zip_featurizer(X_train)
                    X_valid, nan_inds_valid = zip_featurizer(X_valid)
                    X_test, nan_inds_test = zip_featurizer(X_test)
                    # Some features are nan (because rdkit failed to compute partial charges for some molecules)

                    if len(nan_inds_train) > 0:
                        y_train = np.delete(y_train, nan_inds_train, axis=0)
                    if len(nan_inds_valid) > 0:
                        y_valid = np.delete(y_valid, nan_inds_valid, axis=0)
                    if len(nan_inds_test) > 0:
                        y_test = np.delete(y_test, nan_inds_test, axis=0)

                    k = config.get("k", 0)

                    if k == 0:
                        k, valid_preds = classifier_CV.fit_predict(
                            X_train,
                            y_train,
                            X_valid,
                            class_weights,
                            config.get("compressor", GzipCompressor()),
                        )

                    test_preds = classifier.fit_predict(
                        X_train,
                        y_train,
                        X_test,
                        k,
                        class_weights,
                        config.get("compressor", GzipCompressor()),
                    )
                else:
                    k = config.get("k", 0)

                    if k == 0:
                        k, valid_preds = classifier_CV.fit_predict(
                            X_train,
                            y_train,
                            X_valid,
                            class_weights,
                            config.get("compressor", GzipCompressor()),
                        )

                    test_preds = classifier.fit_predict(
                        X_train,
                        y_train,
                        X_test,
                        k,
                        class_weights,
                        config.get("compressor", GzipCompressor()),
                    )

                # Compute metrics
                if not is_k_set:
                    try:
                        valid_auroc = roc_auc_score(
                            y_valid, valid_preds, multi_class="ovr"
                        )
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

                if not is_k_set:
                    print(
                        f"Valid AUROC: {valid_auroc}, Valid F1: {valid_f1} , Test AUROC: {test_auroc}, Test F1: {test_f1}"
                    )
                    run_results.append(
                        [valid_auroc, valid_f1, 0, test_auroc, test_f1, 0]
                    )
                else:
                    print(f"Test AUROC: {test_auroc}, Test F1: {test_f1}")
                    run_results.append([0, 0, 0, test_auroc, test_f1, 0])

            else:
                if config["task"] == "regression":
                    k = config.get("k", 0)

                    if k == 0:
                        k, valid_preds = regressor_CV.fit_predict(
                            X_train,
                            y_train,
                            X_valid,
                            config.get("compressor", GzipCompressor()),
                        )

                    test_preds = regressor.fit_predict(
                        X_train,
                        y_train,
                        X_test,
                        k,
                        config.get("compressor", GzipCompressor()),
                    )

                elif config["task"] == "regression_vec":
                    X_train, nan_inds_train = zip_featurizer(X_train)
                    X_valid, nan_inds_valid = zip_featurizer(X_valid)
                    X_test, nan_inds_test = zip_featurizer(X_test)
                    # Some features are nan (because rdkit failed to compute partial charges for some molecules)
                    if len(nan_inds_train) > 0:
                        y_train = np.delete(y_train, nan_inds_train).reshape(-1, 1)
                    if len(nan_inds_valid) > 0:
                        y_valid = np.delete(y_valid, nan_inds_valid).reshape(-1, 1)
                    if len(nan_inds_test) > 0:
                        y_test = np.delete(y_test, nan_inds_test).reshape(-1, 1)

                    k = config.get("k", 0)

                    if k == 0:
                        k, valid_preds = regressor_CV.fit_predict(
                            X_train,
                            y_train,
                            X_valid,
                            config.get("compressor", GzipCompressor()),
                        )

                    test_preds = regressor.fit_predict(
                        X_train,
                        y_train,
                        X_test,
                        k,
                        config.get("compressor", GzipCompressor()),
                    )

                else:
                    raise ValueError(f"Unknown task {config['task']}")
                # Compute metrics
                if not is_k_set:
                    valid_r = pearsonr(y_valid.flatten(), valid_preds.flatten())
                    valid_rmse = mean_squared_error(y_valid, valid_preds, squared=False)
                    valid_mae = mean_absolute_error(
                        y_valid,
                        valid_preds,
                    )

                test_r = pearsonr(y_test.flatten(), test_preds.flatten())
                test_rmse = mean_squared_error(y_test, test_preds, squared=False)
                test_mae = mean_absolute_error(y_test, test_preds)

                print(f"\n{config['dataset']} ({len(tasks)} tasks)")
                print(config)
                print("best k valid", k)
                config["k"] = k

                if not is_k_set:
                    print(
                        f"Valid R: {valid_r[0]}, Valid RMSE: {valid_rmse}, Valid MAE: {valid_mae}, Test R: {test_r[0]}, Test RMSE: {test_rmse}, Test MAE: {test_mae}"
                    )
                    run_results.append(
                        [
                            valid_rmse,
                            valid_mae,
                            valid_r[0],
                            test_rmse,
                            test_mae,
                            test_r[0],
                        ]
                    )
                else:
                    print(
                        f"Test R: {test_r[0]}, Test RMSE: {test_rmse}, Test MAE: {test_mae}"
                    )
                    run_results.append(
                        [
                            0,
                            0,
                            0,
                            test_rmse,
                            test_mae,
                            test_r[0],
                        ]
                    )

        run_results = np.array(run_results)
        results_means = np.mean(run_results, axis=0)
        results_stds = np.std(run_results, axis=0)

        results.append(
            (
                config,
                {
                    "valid_auroc": f"{round(results_means[0], 3)} +/- {round(results_stds[0], 3)}",
                    "valid_f1": f"{round(results_means[1], 3)} +/- {round(results_stds[1], 3)}",
                    "valid_r": f"{round(results_means[2], 3)} +/- {round(results_stds[2], 3)}",
                    "test_auroc": f"{round(results_means[3], 3)} +/- {round(results_stds[3], 3)}",
                    "test_f1": f"{round(results_means[4], 3)} +/- {round(results_stds[4], 3)}",
                    "test_r": f"{round(results_means[5], 3)} +/- {round(results_stds[5], 3)}",
                    "best_k_valid": k,
                },
            )
        )

        write_table(results)


def main():
    benchmark(get_all_tests())


if __name__ == "__main__":
    main()
