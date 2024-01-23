from typing import Any, Iterable, Optional
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import multiprocessing
import numpy as np
from functools import partial
from collections import Counter

def classify(
    x1: str,
    X_train: Iterable[str],
    y_train: np.ndarray,
    k: int,
    class_weights: Optional[Iterable] = None,
) -> Iterable:
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
    top_k_class = y_train[sorted_idx[:k]]
    top_k_dists = distance_from_x1[sorted_idx[:k]]

    task_preds = []
    for i, task_top_k_class in enumerate(np.array(top_k_class).T):
        counts = dict(Counter([int(e) for e in task_top_k_class]))

        # Group dists based on classes
        grouped_dists = {
            k: [v for v, v2 in zip(top_k_dists, task_top_k_class) if k == v2]
            for k in set(task_top_k_class)
        }

        if isinstance(class_weights, Iterable) and len(class_weights) - 1 >= i:
            for k, v in enumerate(class_weights[i]):
                if k in counts:
                    counts[k] *= v * (1 - np.mean(grouped_dists[k]))

        task_preds.append(max(counts, key=counts.get))

    return task_preds


class RDKitClassifier(object):
    def __init__(self) -> "RDKitClassifier":
        pass

    def fit_predict(
        self,
        X_train: Iterable[str],
        y_train: Iterable,
        X: Iterable[str],
        k: int = 5,
        class_weights: Optional[Iterable] = None,
    ) -> np.ndarray:
        preds = []
        y_train = np.array(y_train)

        if len(y_train.shape) == 1:
            y_train = np.expand_dims(y_train, axis=1)

        cpu_count = multiprocessing.cpu_count()

        with multiprocessing.Pool(cpu_count) as p:
            preds = p.map(
                partial(
                    classify,
                    X_train=X_train,
                    y_train=y_train,
                    k=k,
                    class_weights=class_weights,
                ),
                X,
            )

        return np.array(preds)
