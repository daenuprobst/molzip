import gzip
import multiprocessing
from typing import Iterable
from collections import Counter
from functools import partial
import numpy as np
from tqdm import tqdm


def regress_(x1, X_train, y_train, k):
    Cx1 = len(gzip.compress(x1.encode()))
    distance_from_x1 = []

    for x2 in X_train:
        Cx2 = len(gzip.compress(x2.encode()))
        x1x2 = " ".join([x1, x2])
        Cx1x2 = len(gzip.compress(x1x2.encode()))
        ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
        distance_from_x1.append(ncd)

    distance_from_x1 = np.array(distance_from_x1)
    sorted_idx = np.argsort(distance_from_x1)
    top_k_values = y_train[sorted_idx[:k]]
    # top_k_dists = distance_from_x1[sorted_idx[:k]]

    task_preds = []
    for i, task_top_k_values in enumerate(np.array(top_k_values).T):
        task_preds.append(np.mean(task_top_k_values))

    return task_preds


def regress(X_train, y_train, X_test, k):
    preds = []

    cpu_count = multiprocessing.cpu_count()

    with multiprocessing.Pool(cpu_count) as p:
        preds = p.map(
            partial(
                regress_,
                X_train=X_train,
                y_train=y_train,
                k=k,
            ),
            X_test,
        )

    return np.array(preds)
