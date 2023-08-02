import gzip
import multiprocessing
from typing import Iterable
from collections import Counter
from functools import partial
import numpy as np
from tqdm import tqdm
from scipy.linalg import solve
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

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
    for vals, dists in zip(np.array(top_k_values).T, np.array(top_k_dists).T):
        dists = 1 - dists
        task_preds.append(np.mean(vals * dists) / np.sum(dists))

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


def cross_val_and_fit_kernel_ridge(X, y, k, gammas, lambdas):
    """
    Perform k-fold cross validation to find the best gamma and lambda for kernel ridge regression, 
    and then fit the model to the entire dataset using these hyperparameters.
    
    Parameters:
    X : array-like of shape (n_samples, n_features)
        The input samples. 
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The target values.
    k : int
        The number of folds in the cross-validation.
    gammas : array-like
        The gamma values to try. Gamma is the inverse of the standard deviation of 
        the RBF kernel (used as similarity measure between samples).
    lambdas : array-like
        The lambda values to try. Lambda is the regularization parameter.
    
    Returns:
    best_alpha : array-like of shape (n_samples,) or (n_samples, n_outputs)
        The fitted weights for the kernel ridge regression model.
    best_gamma : float
        The best gamma value found in the cross-validation.
    best_lambda_ : float
        The best lambda value found in the cross-validation.
    best_score : float
        The mean squared error of the model with the best gamma and lambda.
    """
    kf = KFold(n_splits=k)
    
    best_gamma, best_lambda_ = None, None
    best_score = float('inf')
    
    for gamma in gammas:
        for lambda_ in lambdas:
            mse_scores = []
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                alpha  = train_kernel_ridge_regression(X_train, y_train, gamma, lambda_)
                y_pred = predict_kernel_ridge_regression(X_train, X_val, alpha, gamma)
                
                mse = mean_squared_error(y_val, y_pred)
                mse_scores.append(mse)
            
            avg_mse = np.mean(mse_scores)
            
            if avg_mse < best_score:
                best_score = avg_mse
                best_gamma = gamma
                best_lambda_ = lambda_

    # Perform one final fit on all the data using the best hyperparameters
    best_alpha = train_kernel_ridge_regression(X, y, best_gamma, best_lambda_)

    return best_alpha, best_gamma, best_lambda_, best_score