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
    """Generate a byte compression matrix from x1 based upon
    passed X_train input concats

    PARAMETERS
    ----------
    x1 : str
        Input smiles molecule string

    X_train : tuple | list (iterable)
        list or tuple of smiles moleclue strings

    y_train : list (iterable)
        y labels for each element in X_train

    k : int
        Number of neighbors to collect in K-NN


    RETURNS
    -------
    task_preds : list
        
    """
    # Compress encoded (utf-8) x1 passed in to fun
    # take length
    Cx1 = len(gzip.compress(x1.encode()))
    distance_from_x1 = []

    # Loop through x2 to generate 
    # normalized compression distances from Cx1x2 based
    # upon Cx1 and Cx2. We normalize by longest compression
    # sequence
    for x2 in X_train:
        Cx2 = len(gzip.compress(x2.encode()))
        x1x2 = " ".join([x1, x2])
        Cx1x2 = len(gzip.compress(x1x2.encode()))
        ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
        distance_from_x1.append(ncd)

    # Sort ncd indexes in ascending order based upon distance
    # These are used to grab top K-NN for both values and dists
    distance_from_x1 = np.array(distance_from_x1)
    sorted_idx = np.argsort(distance_from_x1)
    top_k_values = y_train[sorted_idx[:k]]
    top_k_dists = distance_from_x1[sorted_idx[:k]]

    task_preds = []
    # Loop through both top_k_value.T and top_k_dists.T
    # Scale the predictions by the total distance found within a sample
    for vals, dists in zip(np.array(top_k_values).T, np.array(top_k_dists).T):
        dists = 1 - dists
        task_preds.append(np.mean(vals * dists) / np.sum(dists))

    return task_preds


def regress(X_train, y_train, X_test, k):
    """Perform Regression based task on dataset

    PARAMETERS
    ----------
    X_train : list | tuple (iterable)
        Smiles train dataset

    y_train : list | tuple (iterable)
        Particular y prediction values associated with
        regression task specified

    X_test : list | tuple (iterable)
        Smiles testing dataset

    k : int
        Amount of neighbors to consider in K-NN

    RETURNS
    -------
    preds : np.array
        Array of predicted y values for the X_test dataset
    """
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
    """Compute the normalized compression pairwise distance 
    between two smiles

    PARAMTERS
    ---------
    pair : tuple
        Smiles to compare

    RETURNS
    -------
    ncd : float
        Normalized Compression Distance
    """
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
