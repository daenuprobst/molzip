import gzip
import multiprocessing
from typing import Iterable
from collections import Counter
from functools import partial
import numpy as np


def get_knn(x, X, k, compressor=gzip):
    x1, idx = x
    Cx1 = len(compressor.compress(x1.encode()))
    distance_from_x1 = []

    for x2 in X:
        Cx2 = len(compressor.compress(x2.encode()))
        x1x2 = " ".join([x1, x2])
        Cx1x2 = len(compressor.compress(x1x2.encode()))
        ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
        distance_from_x1.append(ncd)

    distance_from_x1 = np.array(distance_from_x1)
    sorted_idx = np.argsort(distance_from_x1)

    top_k_idx = sorted_idx[:k]
    top_k_dists = distance_from_x1[top_k_idx]

    return [
        tuple([int(i), int(j), float(d)])
        for i, j, d in zip([idx] * len(top_k_idx), top_k_idx, top_k_dists)
    ]


class ZipKNNGraph(object):
    def __init__(self) -> "ZipKNNGraph":
        pass

    def fit_predict(self, X, k):
        edge_list = []

        cpu_count = multiprocessing.cpu_count()

        with multiprocessing.Pool(cpu_count) as p:
            edge_list = p.map(
                partial(
                    get_knn,
                    X=X,
                    k=k,
                ),
                list(zip(X, range(len(X)))),
            )
        # Returned the flattened list
        return [item for sublist in edge_list for item in sublist]
