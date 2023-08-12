import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt
from gzip_regressor import regress, cross_val_and_fit_kernel_ridge,predict_kernel_ridge_regression
import matplotlib.pyplot as plt
import numpy as np
from xyz2mol import *
import deepchem.molnet as mn
import deepchem as dc
import numpy as np
import os
from smiles_tokenizer import tokenize
import selfies as sf
from sklearn.metrics import mean_squared_error,mean_absolute_error


def bin_vectors(X, num_bins):
    """
    Convert a 2D numpy array of vectors into a list of string representations based on binning.

    Parameters:
    - X (numpy.ndarray): A 2D array of shape (n_samples, n_features) to be binned.
    - num_bins (int): The number of bins to be used for binning.

    Returns:
    - list: A list of string representations of the binned vectors.

    Description:
    The function first determines the global minimum and maximum values across all vectors.
    It then creates separate bins for positive and negative values. Each value in the vectors
    is then assigned to a bin and represented by a unique Unicode character. Negative values
    are prefixed with a special '✖' character. The binned representations of the vectors are 
    then returned as a list of strings.
    """

    # Create bins for positive and negative numbers separately
    X_flattened = X.flatten()
    pos_vector = X_flattened[X_flattened >= 0]
    neg_vector = -X_flattened[X_flattened < 0]  # Flip sign for binning

    pos_bins = np.linspace(0, max(pos_vector) if len(pos_vector) > 0 else 1, num_bins+1)
    neg_bins = np.linspace(0, max(neg_vector) if len(neg_vector) > 0 else 1, num_bins+1)

    # Create a mapping from bin number to Unicode character
    bin_to_char = {i+1: chr(9786 + i) for i in range(num_bins)}

    # Apply binning to each vector
    string_reps = []
    for vector in X:
        # Digitize the vectors
        pos_digitized = np.digitize(vector[vector >= 0], pos_bins)
        neg_digitized = np.digitize(-vector[vector < 0], neg_bins)

        # Convert digitized vectors to string representation
        pos_string_rep = [bin_to_char.get(num, '?') for num in pos_digitized]
        neg_string_rep = [f'✖{bin_to_char.get(num, "?")}' for num in neg_digitized]

        # Combine the representations in the original order
        string_rep = []
        pos_index = 0
        neg_index = 0
        for num in vector:
            if num >= 0:
                string_rep.append(pos_string_rep[pos_index])
                pos_index += 1
            else:
                string_rep.append(neg_string_rep[neg_index])
                neg_index += 1

        string_reps.append(''.join(string_rep))

    return string_reps


def preprocess(smiles: str, type_preproc: str, preproc: bool = False) -> str:
    if not preproc:
        return smiles

    else:
        if type_preproc == "tok_smiles":
            return " ".join(tokenize(sf.encoder(smiles, strict=False)))
        if type_preproc == "selfies":
            return " ".join(tokenize(sf.encoder(smiles, strict=False)))

def molnet_loader(type_preproc,
    name: str, preproc: bool = False, **kwargs
) -> Tuple[str, np.array, np.array, np.array]:
    mn_loader = getattr(mn, f"load_{name}")
    dc_set = mn_loader(**kwargs)

    tasks, dataset, _ = dc_set
    train, valid, test = dataset
    

    SMILES_train = train.ids
    SMILES_valid = valid.ids
    SMILES_test = test.ids


    X_train = np.array([preprocess(x,type_preproc, preproc) for x in SMILES_train])
    X_valid = np.array([preprocess(x,type_preproc, preproc) for x in SMILES_valid])
    X_test  = np.array([preprocess(x,type_preproc, preproc) for x in SMILES_test])

    y_train = np.array(train.y, dtype=float)
    y_valid = np.array(valid.y, dtype=float)
    y_test = np.array(test.y, dtype=float)

    return tasks,SMILES_train,SMILES_valid,SMILES_test, X_train, y_train, X_valid, y_valid, X_test, y_test


if __name__ == "__main__":
    configs = [
{
        "dataset": "freesolv",
        "splitter": "random",
        "task": "regression_knn",
        "k": 25,
        "augment": 0,
        "preprocess": False,
        "type_preproc": "default",
        "sub_sample": 0.0,
        "is_imbalanced": False,
        "n": 4,
        "bins": 70,
        "smilesANDvec": False
    },        
    {
        "dataset": "freesolv",
        "splitter": "random",
        "task": "regression_knn",
        "k": 25,
        "augment": 0,
        "preprocess": False,
        "type_preproc": "default",
        "sub_sample": 0.0,
        "is_imbalanced": False,
        "n": 4,
        "bins": 70,
        "smilesANDvec": True
    },
    #{
        #"dataset": "freesolv",
        #"splitter": "random",
        #"task": "regression_krr",
        #"kfold": 5,
        #"augment": 0,
        #"gammas": np.logspace(-1, 3, 13),
        #"lambdas": [1e-7, 1e-6, 1e-5],
        #"preprocess": False,
        #"type_preproc": "default",
        #"sub_sample": 0.0,
        #"is_imbalanced": False,
        #"n": 4,
        #"bins": 70,
        #"smilesANDvec": False
        #},
        #{
        #"dataset": "freesolv",
        #"splitter": "random",
        #"task": "regression_krr",
        #"kfold": 5,
        #"augment": 0,
        #"gammas": np.logspace(-1, 3, 13),
        #"lambdas": [1e-7, 1e-6, 1e-5],
        #"preprocess": False,
        #"type_preproc": "default",
        #"sub_sample": 0.0,
        #"is_imbalanced": False,
        #"n": 4,
        #"bins": 70,
        #"smilesANDvec": True
        #}
    ]
    #N = [100, 200, 300, 400, 512]
    
    results = []

    for config in configs:
        run_results = []
        for _ in range(config["n"]):
            

            tasks,SMILES_train,SMILES_valid,SMILES_test, X_train, y_train, X_valid, y_valid, X_test, y_test = molnet_loader(config["type_preproc"],
                config["dataset"],
                splitter=config["splitter"],
                preproc=config["preprocess"],
                reload=False,
                transformers=[])

            if config["smilesANDvec"]:
                #after featurization we have to add the vectors to the SMILES
                featurizer_rdkit = dc.feat.RDKitDescriptors(is_normalized=True)
                vectors_train = bin_vectors(featurizer_rdkit.featurize(SMILES_train), config["bins"])
                featurizer_rdkit = 0 
                featurizer_rdkit = dc.feat.RDKitDescriptors(is_normalized=True)
                vectors_valid = bin_vectors(featurizer_rdkit.featurize(SMILES_valid), config["bins"])
                featurizer_rdkit = 0
                featurizer_rdkit = dc.feat.RDKitDescriptors(is_normalized=True)
                vectors_test  = bin_vectors(featurizer_rdkit.featurize(SMILES_test), config["bins"])

                X_train = np.array([np.array(s+x) for s,x in zip(SMILES_train,vectors_train)])
                X_valid = np.array([np.array(s+x) for s,x in zip(SMILES_valid,vectors_valid)])
                X_test = np.array([np.array(s+x) for s,x in zip(SMILES_test,vectors_test)])

            if config["task"] == "regression_knn":
                valid_preds = regress(X_train, y_train, X_valid, config["k"])
                test_preds = regress(X_train, y_train, X_test, config["k"])

            elif config["task"] == "regression_krr":
                    best_alpha, best_gamma, best_lambda_, best_score = cross_val_and_fit_kernel_ridge(X_train, y_train, config["kfold"], config["gammas"], config["lambdas"])
                    print(f"Best gamma: {best_gamma}, Best lambda: {best_lambda_}")
                    valid_preds = predict_kernel_ridge_regression(X_train, X_valid, best_alpha, best_gamma)
                    test_preds  = predict_kernel_ridge_regression(X_train, X_test, best_alpha, best_gamma)
            else:
                raise ValueError(f"Unknown task {config['task']}")

            # Compute metrics
            valid_rmse = mean_squared_error(y_valid, valid_preds, squared=False)
            valid_mae = mean_absolute_error(y_valid,valid_preds)
            test_rmse = mean_squared_error(y_test, test_preds, squared=False)
            test_mae = mean_absolute_error(y_test,test_preds)
            run_results.append([valid_rmse, valid_mae, test_rmse, test_mae])

        run_results = np.array(run_results)
        results_means = np.mean(run_results, axis=0)
        results_stds = np.std(run_results, axis=0)

        print(results_means)
        print(results_stds)


        results.append(
            (
                config,
                {
                    "valid_auroc": f"{round(results_means[0], 3)} +/- {round(results_stds[0], 3)}",
                    "valid_f1": f"{round(results_means[1], 3)} +/- {round(results_stds[0], 3)}",
                    "test_auroc": f"{round(results_means[2], 3)} +/- {round(results_stds[0], 3)}",
                    "test_f1": f"{round(results_means[3], 3)} +/- {round(results_stds[0], 3)}",
                },
            )
        )