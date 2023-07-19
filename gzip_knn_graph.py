import gzip
import multiprocessing
from typing import Iterable
from collections import Counter
from functools import partial
import numpy as np
from tqdm import tqdm

compressor = gzip


def get_knn_graph_(x1, X_train, k):
    x1_idx = x1[0]
    x1 = x1[1]
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
    top_k_dists = distance_from_x1[sorted_idx[:k]]

    return tuple(x1_idx, zip(sorted_idx, top_k_dists))


def get_knn_graph_(X_train, X_test, k):
    preds = []

    cpu_count = multiprocessing.cpu_count()

    with multiprocessing.Pool(cpu_count) as p:
        preds = p.map(
            partial(
                get_knn_graph_,
                X_train=X_train,
                k=k,
            ),
            enumerate(X_test),
        )

    return np.array(preds)
