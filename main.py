
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
from gzip_classifier import classify
from gzip_regressor import regress
from gzip_utils import *
from config import *
random.seed(666)


def benchmark(configs: List[Dict[str, Any]]) -> None:
    results = []

    for config in configs:
        loader = molnet_loader

        if config["dataset"] in ["schneider"]:
            loader = schneider_loader

        if config["dataset"].startswith("adme"):
            loader = adme_loader

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
                if config["task"] == "regression":
                    
                    valid_preds = regress(X_train, y_train, X_valid, config["k"])
                    test_preds = regress(X_train, y_train, X_test, config["k"])
                
                elif config["task"] == "regression_vec":
                    X_train, nan_inds_train = get_smiles_vec_rep(X_train, config=config)
                    X_valid, nan_inds_valid = get_smiles_vec_rep(X_valid, config=config)
                    X_test,  nan_inds_test  = get_smiles_vec_rep(X_test, config=config)
                    #Some features are nan (because rdkit failed to compute partial charges for some molecules)
                    if len(nan_inds_train) > 0:
                        y_train = np.delete(y_train, nan_inds_train).reshape(-1,1)
                    if len(nan_inds_valid) > 0:
                        y_valid = np.delete(y_valid, nan_inds_valid).reshape(-1,1)
                    if len(nan_inds_test) > 0:
                        y_test = np.delete(y_test, nan_inds_test).reshape(-1,1)

                    valid_preds = regress(X_train, y_train, X_valid, config["k"])
                    test_preds  = regress(X_train, y_train, X_test, config["k"])
                    
                else:
                    raise ValueError(f"Unknown task {config['task']}")
                # Compute metrics
                valid_r = pearsonr(y_valid.flatten(), valid_preds.flatten())
                valid_rmse = mean_squared_error(y_valid, valid_preds, squared=False)
                valid_mae = mean_absolute_error(
                    y_valid,
                    valid_preds,
                )

                test_r = pearsonr(y_test.flatten(), test_preds.flatten())
                test_rmse = mean_squared_error(y_test, test_preds, squared=False)
                test_mae = mean_absolute_error(
                    y_test,
                    test_preds
                )



                print(f"\n{config['dataset']} ({len(tasks)} tasks)")
                print(config)
                print(
                    f"Valid R: {valid_r[0]}, Valid RMSE: {valid_rmse}, Valid MAE: {valid_mae}, Test R: {test_r[0]}, Test RMSE: {test_rmse}, Test MAE: {test_mae}"
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
    benchmark(all_tests)

if __name__ == "__main__":
    main()
