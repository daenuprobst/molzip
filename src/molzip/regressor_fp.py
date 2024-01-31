from typing import Any, Iterable
from rdkit.Chem import AllChem
import multiprocessing
import numpy as np
from rdkit import Chem
from functools import partial
from rdkit import DataStructs
from sklearn.model_selection import KFold


def regress(x1: str, X_train: Iterable[str], y_train: np.ndarray, k: int) -> Iterable:
    # Convert SMILES strings to RDKit molecules and compute ECFP fingerprints
    mol_x1 = Chem.MolFromSmiles(x1)
    fp_x1 = AllChem.GetMorganFingerprint(mol_x1, 2)  # Radius 2 ECFP

    fps_train = [
        AllChem.GetMorganFingerprint(Chem.MolFromSmiles(x), 2) for x in X_train
    ]

    # Calculate Jaccard distances
    distance_from_x1 = [DataStructs.TanimotoSimilarity(fp_x1, fp) for fp in fps_train]
    distance_from_x1 = 1 - np.array(distance_from_x1)  # Convert similarity to distance

    # Proceed with the rest of the function as before
    sorted_idx = np.argsort(distance_from_x1)
    top_k_values = y_train[sorted_idx[:k]]
    top_k_dists = distance_from_x1[sorted_idx[:k]]

    task_preds = []
    for vals, dists in zip(np.array(top_k_values).T, np.array(top_k_dists).T):
        dists = 1 - dists
        task_preds.append(np.mean(vals * dists) / np.sum(dists))

    return task_preds


class RDKitRegressor(object):
    def __init__(self) -> "RDKitRegressor":
        pass

    def fit_predict(
        self, X_train: Iterable[str], y_train: Iterable, X: Iterable[str], k: int = 25
    ) -> np.ndarray:
        preds = []

        y_train = np.array(y_train)
        if len(y_train.shape) == 1:
            y_train = np.expand_dims(y_train, axis=1)

        cpu_count = multiprocessing.cpu_count()

        with multiprocessing.Pool(cpu_count) as p:
            preds = p.map(
                partial(
                    regress,
                    X_train=X_train,
                    y_train=y_train,
                    k=k,
                ),
                X,
            )

        return np.array(preds)


def custom_mse(y_true, y_pred):
    # Custom MSE function that ignores NaN values, for some molecules rdkit fails (again)
    nan_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_filtered, y_pred_filtered = y_true[nan_mask], y_pred[nan_mask]
    return ((y_true_filtered - y_pred_filtered) ** 2).mean()


class RDKitRegressor_CV(object):
    def __init__(self) -> "RDKitRegressor_CV":
        self.regressor = RDKitRegressor()

    def fit_predict(
        self, X_train: Iterable[str], y_train: Iterable, X: Iterable[str]
    ) -> np.ndarray:
        y_train = np.array(y_train)
        if len(y_train.shape) == 1:
            y_train = np.expand_dims(y_train, axis=1)

        # Determine the best k using 5-fold cross-validation
        kf = KFold(n_splits=5)
        k_values = range(1, 35)  # Range of k values to try
        k_performance = {}

        for k in k_values:
            k_scores = []
            for train_index, test_index in kf.split(X_train):
                X_train_fold, X_val_fold = (
                    np.array(X_train)[train_index],
                    np.array(X_train)[test_index],
                )
                y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]

                # Train and predict with the current fold and k value
                preds_fold = self._predict_with_k(
                    X_train_fold, y_train_fold, X_val_fold, k
                )
                fold_score = custom_mse(y_val_fold, preds_fold)
                k_scores.append(fold_score)

            k_performance[k] = np.mean(k_scores)

        # Select the k with the lowest average score (MSE)
        best_k = min(k_performance, key=k_performance.get)

        # Train the final model on the entire training set with the best k
        final_predictions = self._predict_with_k(X_train, y_train, X, best_k)
        return best_k, final_predictions

    def _predict_with_k(self, X_train, y_train, X, k):
        # Modified to use a specific value of k for predictions
        return self.regressor.fit_predict(X_train, y_train, X, k)