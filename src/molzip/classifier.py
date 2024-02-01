from bz2 import compress
import gzip
import multiprocessing
from typing import Iterable, Optional, Any
from collections import Counter
from functools import partial
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from molzip.compressors import Compressor, GzipCompressor


def classify(
    x1: str,
    X_train: Iterable[str],
    y_train: np.ndarray,
    k: int,
    class_weights: Optional[Iterable] = None,
    compressor: Compressor = GzipCompressor(),
) -> Iterable:
    Cx1 = len(compressor.compress(x1))
    distance_from_x1 = []

    for x2 in X_train:
        Cx2 = len(compressor.compress(x2))
        x1x2 = " ".join([x1, x2])
        x2x1 = " ".join([x2, x1])
        Cx1x2 = len(compressor.compress(x1x2))
        Cx2x1 = len(compressor.compress(x2x1))
        ncd = (0.5 * (Cx1x2 + Cx2x1) - min(Cx1, Cx2)) / max(Cx1, Cx2)
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
        compressor: Compressor = GzipCompressor(),
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
                    compressor=compressor,
                ),
                X,
            )

        return np.array(preds)


class ZipClassifier_CV(object):
    def __init__(self) -> "ZipClassifier":
        pass

    def fit_predict(
        self,
        X_train: Iterable[str],
        y_train: Iterable,
        X: Iterable[str],
        class_weights: Optional[Iterable] = None,
        compressor: Compressor = GzipCompressor(),
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
                    X_train_fold,
                    y_train_fold,
                    X_val_fold,
                    k,
                    class_weights,
                    compressor=compressor,
                )
                fold_score = accuracy_score(y_val_fold, preds_fold)
                k_scores.append(fold_score)

            k_performance[k] = np.mean(k_scores)

        # Select the k with the highest average score
        best_k = max(k_performance, key=k_performance.get)

        # Make predictions with the best k value
        return best_k, self._predict_with_k(X_train, y_train, X, best_k, class_weights)

    def _predict_with_k(self, X_train, y_train, X, k, class_weights, compressor):
        cpu_count = multiprocessing.cpu_count()
        with multiprocessing.Pool(cpu_count) as p:
            preds = p.map(
                partial(
                    classify,
                    X_train=X_train,
                    y_train=y_train,
                    k=k,
                    class_weights=class_weights,
                    compressor=compressor,
                ),
                X,
            )
        return np.array(preds)
