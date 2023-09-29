import gzip
import multiprocessing
from typing import Any, Iterable
from functools import partial
import numpy as np


def regress(
    x1: str, X_train: Iterable[str], y_train: np.ndarray, k: int, compressor: Any = gzip
) -> Iterable:
    Cx1 = len(compressor.compress(x1.encode()))
    distance_from_x1 = []

    for x2 in X_train:
        Cx2 = len(compressor.compress(x2.encode()))
        x1x2 = " ".join([x1, x2])
        Cx1x2 = len(compressor.compress(x1x2.encode()))
        ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
        distance_from_x1.append(ncd)

    distance_from_x1 = np.array(distance_from_x1)
    sorted_idx = np.argsort(distance_from_x1)
    top_k_values = y_train[sorted_idx[:k]]
    top_k_dists = distance_from_x1[sorted_idx[:k]]

    task_preds = []
    for vals, dists in zip(np.array(top_k_values).T, np.array(top_k_dists).T):
        dists = 1 - dists
        task_preds.append(np.mean(vals * dists) / np.sum(dists))

    return task_preds


class ZipRegressor(object):
    def __init__(self) -> "ZipRegressor":
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
