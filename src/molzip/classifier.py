import gzip
import multiprocessing
from typing import Iterable, Optional, Any
from collections import Counter
from functools import partial
import numpy as np


def classify(
    x1: str,
    X_train: Iterable[str],
    y_train: np.ndarray,
    k: int,
    class_weights: Optional[Iterable] = None,
    compressor: Any = gzip,
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


class ZipClassifier(object):
    def __init__(self) -> "ZipClassifier":
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
