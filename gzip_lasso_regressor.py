import gzip
import multiprocessing
from typing import Iterable, Tuple
from collections import Counter
from functools import partial
import numpy as np
from tqdm import tqdm
from scipy.linalg import solve
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold

def genonehot_smiles(smiles: list, onehot_dict: dict=dict()) -> dict:
    """Convert smiles into a categorical onehot dictionary.
    The model needs to be able to track what it is looking
    at not only NCD

    PARAMETERS
    ----------
    smiles : List[str]
        List containing smiles.

    onehot_dict : dict (default is dict())
        This is either an empty dict or one that needs to be added
        to. The keys are the smiles and the values are its column
        in the onehot matrix.

    RETURNS
    -------
    onehot_dict : dict
        The keys are the smiles and the values are its columns
        in the onehot matrix.
    """
    # determine if empty dict or not
    if onehot_dict == {}:
        # Pull out unique smiles
        unique_smiles = set(smiles)
        # make a numpy array that goes from 0 to len
        Nunique_smiles = np.arange(len(unique_smiles))
        onehot_dict = dict(zip(
            unique_smiles,
            Nunique_smiles
            ))
    else:
        # pull out old smiles
        key_smiles = list(onehot_dict.keys())
        # combine the list
        key_smiles.extend(smiles)
        # Pull out unique smiles
        unique_smiles = set(key_smiles)
        # make a numpy array that goes from 0 to len
        Nunique_smiles = np.arange(len(unique_smiles))
        onehot_dict = dict(
                zip(
                    unique_smiles,
                    Nunique_smiles
                    )
                )
    return onehot_dict


def regress_lasso_traindata_(x1: str,
                             X_train: np.array,
                             k: int,
                             onehot_dict:dict) -> Tuple[np.array, np.array]:
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

    k: int
        Number of Neighbors to consider in k-NN

    RETURNS
    -------
    distance_from_x1 : np.array
        NCD values associated with each Cx1x2 pair

    k_pairs : np.array
        An Array of size (K, 2). The top k Cx1x2 distances
        in kNN, but the indices 
    """
    # Compress encoded (utf-8) x1 passed in to fun
    # take length
    Cx1 = len(gzip.compress(x1.encode()))
    distance_from_x1 = []
    smile_list = []

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
        smile_list.append(x2)

    # Return all of the distances from x1
    distance_from_x1 = np.array(distance_from_x1)

    # if kNN is set then apply neighbors only
    if k != 0:
        # Sort distances and pull out top k neighbors
        sorted_idx = np.argsort(distance_from_x1)
        distance_from_x1 = distance_from_x1[sorted_idx[:k]]
        top_k_distance = [smile_list[i] for i in sorted_idx[:k]]
        # pull out top 5 indexes per row
        # k_pairs = sorted_idx[:5]
        vectorize_keys = np.vectorize(lambda key1, key2: (onehot_dict[key1],
                                                          onehot_dict[key2]))
        k_pairs = vectorize_keys([x1]*k, top_k_distance)
        k_pairs = np.vstack(
                (k_pairs[0], k_pairs[1]),
                )
    else:
        k_pairs = np.array(0)

    return distance_from_x1, k_pairs.T


def regress_lasso(X_train, y_train, X_test,
                  k, onehot_dict) -> np.array:
    """Perform Lasso based Regression on dataset to
    train and predict

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
        Determine number of neighbors to consider in kNN
    RETURNS
    -------
    preds : np.array
        Array of predicted y values for the X_test dataset
    """
    # dataset_train = []
    # preds = []
    # k_pairs = []

    cpu_count = multiprocessing.cpu_count()

    with multiprocessing.Pool(cpu_count) as p:
        train_tuple = p.map(
                partial(
                    regress_lasso_traindata_,
                    X_train=X_train,
                    k=k,
                    onehot_dict=onehot_dict,
                ),
                X_train,
                )
        dataset_train, k_pairs_train = zip(*train_tuple)

    with multiprocessing.Pool(cpu_count) as p:
        test_tuple = p.map(
                partial(
                    regress_lasso_traindata_,
                    X_train=X_train,
                    k=k,
                    onehot_dict=onehot_dict
                    ),
                X_test
                )
        dataset_test, k_pairs_test = zip(*test_tuple)


    # Convert both datasets to np.arrays for fitting
    dataset_train = np.array(dataset_train)
    dataset_test = np.array(dataset_test)

    # grab length
    train_len = dataset_train.shape[0]
    test_len = dataset_test.shape[0]

    # Full dataset
    full_dataset = np.concatenate((
        dataset_train,
        dataset_test,
        ))

    # Number of pairs in data
    pairs = np.concatenate((
        np.concatenate(k_pairs_train),
        np.concatenate(k_pairs_test)
        ))

    # encode string pairs into categorical data
    onehot_mat = np.zeros((pairs.shape[0], len(onehot_dict)))
    onehot_mat[np.arange(pairs.shape[0]), pairs[:, 0]] = 1.
    onehot_mat[np.arange(pairs.shape[0]), pairs[:, 1]] = 1.

    # combine distances and onehote pairs
    X_full = np.concatenate(
            (onehot_mat, full_dataset.flatten().reshape(-1,1)),
            axis=1
            )

    if k != 0:
        train_X = X_full[:train_len*k,:]
        test_X = X_full[train_len*k:,:]
        # reformat y to fit train_X
        y_reshape = np.repeat(y_train, k, axis=0)
    else:
        train_X = X_full[:train_len*train_len,:]
        test_X = X_full[train_len*train_len:,:]
        # reformat y to fit train_X
        y_reshape = np.repeat(y_train, train_len, axis=0)

    gridsearch_alpha(train_X, y_reshape)

    # Init lasso model
    # alpha generated from GridSearchCV
    model = Lasso(alpha=0.01)

    # fit and predict values
    model.fit(train_X, y_reshape)
    preds = model.predict(test_X)

    return np.array(preds)

def gridsearch_alpha(X_train, y_train) -> int:
    print('\n')
    print("*********STARTING GRIDSEARCH***********")
    model = Lasso()

    # define model evaluation method
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define grid
    grid = dict()
    grid['alpha'] = np.linspace(0, 1, 21)
    # define search
    search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    # perform the search
    results = search.fit(X_train, y_train)
    # summarize
    print('MAE: %.3f' % results.best_score_)
    print('Config: %s' % results.best_params_)
    
    return 0
