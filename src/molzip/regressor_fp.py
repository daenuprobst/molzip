from typing import Any, Iterable
from rdkit.Chem import AllChem
import multiprocessing
import numpy as np
from rdkit import Chem
from functools import partial
from rdkit import DataStructs


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
