import gzip
import multiprocessing
from typing import Iterable
from collections import Counter
from functools import partial
import numpy as np
from tqdm import tqdm
from scipy.linalg import solve
from sklearn.metrics.pairwise import pairwise_distances


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
    top_k_dists = distance_from_x1[sorted_idx[:k]]

    task_preds = []

    top_k_dists_array = np.array(top_k_dists).T
    top_k_values_array = np.array(top_k_values).T#.reshape(1,-1)

    # print("top_k_dist",top_k_dists_array.shape)
    # print("top_k_values",top_k_values_array.shape)

    dists = 1 - top_k_dists_array  # apply the transformation to all distances at once
    weighted_values = top_k_values_array * dists  # element-wise multiplication
    task_preds = np.sum(weighted_values) / np.sum(dists)  # a single prediction

    # print("task_pred", task_preds)
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


def compute_pairwise_ncd(pair):
    x1, x2 = pair
    Cx1 = len(gzip.compress(x1.encode()))
    Cx2 = len(gzip.compress(x2.encode()))
    Cx1x2 = len(gzip.compress(" ".join([x1, x2]).encode()))
    ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
    return ncd


def compute_ncd(X1, X2):
    pairs = [(x1, x2) for x1 in X1 for x2 in X2]
    with multiprocessing.Pool() as pool:
        ncd_values = pool.map(compute_pairwise_ncd, pairs)
    NCD = np.array(ncd_values).reshape(len(X1), len(X2))
    return NCD


def train_kernel_ridge_regression(X_train, y_train, gamma, lambda_):
    # Compute the pairwise distance matrix
    NCD = compute_ncd(X_train, X_train)

    # Compute the Laplacian kernel matrix
    K = np.exp(-gamma * NCD)

    # Solve for alpha
    alpha = solve(K + lambda_ * np.eye(K.shape[0]), y_train)

    return alpha


def predict_kernel_ridge_regression(X_train, X_test, alpha, gamma):
    # Compute the pairwise distance matrix between X_test and X_train
    NCD_test = compute_ncd(X_test, X_train)

    # Compute the Laplacian kernel matrix
    K_test = np.exp(-gamma * NCD_test)

    # Compute the predictions
    y_pred = K_test.dot(alpha)

    return y_pred
