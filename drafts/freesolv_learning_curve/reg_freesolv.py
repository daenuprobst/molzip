import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from gzip_regressor import regress, cross_val_and_fit_kernel_ridge,predict_kernel_ridge_regression
import matplotlib.pyplot as plt
import numpy as np
import deepchem.molnet as mn
import deepchem as dc
import numpy as np
import selfies as sf
from sklearn.metrics import mean_squared_error,mean_absolute_error
import random
from gzip_utils import *
import pdb
random.seed(42)


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
    y_test  = np.array(test.y, dtype=float)

    return tasks,SMILES_train,SMILES_valid,SMILES_test, X_train, y_train, X_valid, y_valid, X_test, y_test


if __name__ == "__main__":

    configs = [
            {
        "dataset": "delaney",
        "splitter": "random",
        "task": "regression_knn",
        "k": 25,
        "augment": 0,
        "type_preproc": "default",
        "preprocess": False,
        "sub_sample": 0.0,
        "is_imbalanced": False,
        "n": 4,
        "bins": 70,
        "smilesANDvec": True,
        "label" : "SMILES & Vector, Delaney"
    },
            {
        "dataset": "delaney",
        "splitter": "random",
        "task": "regression_knn",
        "k": 25,
        "augment": 0,
        "type_preproc": "default",
        "preprocess": False,
        "sub_sample": 0.0,
        "is_imbalanced": False,
        "n": 4,
        "bins": 70,
        "smilesANDvec": False,
        "label" : "SMILES, Delaney"
    },
    {
        "dataset": "sampl",
        "label" : "SMILES",
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
        "dataset": "sampl",
        "label" : "SMILES & Vector",
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
    }
    ]

    N = [2**i for i in range(4, 10)]
    
    results = []
    all_learning_curves = []

    for config in configs:
        run_results = []
        run_learning_curve = []
        for _ in range(config["n"]):
            
            tasks,SMILES_train,SMILES_valid,SMILES_test, X_train, y_train, X_valid, y_valid, X_test, y_test = molnet_loader(config["type_preproc"],
                config["dataset"],
                splitter=config["splitter"],
                preproc=config["preprocess"],
                reload=False,
                transformers=[])

            if config["smilesANDvec"]:
                X_train = get_smiles_vec_rep(SMILES_train, config=config)
                X_valid = get_smiles_vec_rep(SMILES_valid, config=config)
                X_test  = get_smiles_vec_rep(SMILES_test, config=config)

            curr_lrn_curve = []
            if config["task"] == "regression_knn":
                for n in N:
                    valid_preds = regress(X_train, y_train, X_valid, config["k"])
                    #timing for the prediction

                    
                    test_preds = regress(X_train[:n], y_train[:n], X_test, config["k"])
                    
                    
                    test_mae = mean_absolute_error(y_test,test_preds)
                    curr_lrn_curve.append(test_mae)

            elif config["task"] == "regression_krr":
                    best_alpha, best_gamma, best_lambda_, best_score = cross_val_and_fit_kernel_ridge(X_train, y_train, config["kfold"], config["gammas"], config["lambdas"])
                    print(f"Best gamma: {best_gamma}, Best lambda: {best_lambda_}")
                    valid_preds = predict_kernel_ridge_regression(X_train, X_valid, best_alpha, best_gamma)
                    test_preds  = predict_kernel_ridge_regression(X_train, X_test, best_alpha, best_gamma)
            else:
                raise ValueError(f"Unknown task {config['task']}")

            
            run_learning_curve.append(curr_lrn_curve)
            
            # Compute metrics
            valid_rmse = mean_squared_error(y_valid, valid_preds, squared=False)
            valid_mae = mean_absolute_error(y_valid,valid_preds)
            test_rmse = mean_squared_error(y_test, test_preds, squared=False)
            test_mae = mean_absolute_error(y_test,test_preds)
            run_results.append([valid_rmse, valid_mae, test_rmse, test_mae])        

        all_learning_curves.append([np.mean(np.array(run_learning_curve), axis=0), np.std(np.array(run_learning_curve), axis=0)])
        run_results = np.array(run_results)
        results_means = np.mean(run_results, axis=0)
        results_stds = np.std(run_results, axis=0)

        print(config["label"] , results_means,results_stds)


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

    all_learning_curves = np.array(all_learning_curves)


    
    fontsize = 20
    # Plot learning curves
    fig, ax = plt.subplots(figsize=(6, 8))


    to_be_plotted = ["SMILES", "SMILES & Vector", "SELFIES", "SELFIES & Vector"]
    for i, config in enumerate(configs):
        curr_label = config["label"]
        if curr_label not in to_be_plotted:
            continue
        else:
            ax.plot(N, all_learning_curves[i][0], "-o", label=curr_label, linewidth=3)

            ax.fill_between(N, all_learning_curves[i][0] - all_learning_curves[i][1], 
                            all_learning_curves[i][0] + all_learning_curves[i][1], alpha=0.3)

        
    # Set labels with increased font size
    ax.set_xlabel(r"$N$", fontsize=fontsize+3)
    ax.set_title("MAE [kcal/mol]", fontsize=fontsize+3, loc='left', pad=20)
    #ax.set_title("Learning Curves of Different Models", fontsize=fontsize+2)

    # Set log scales
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Adjust ticks for better clarity in log scale
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

    # Increase tick label font size
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    #add ticks at N 
    ax.set_xticks(N)
    ax.set_xticklabels(N)
    ax.tick_params(axis='both', which='minor', size=0)
    #set y ticks
    yticks = [3.0, 2.0, 1.5, 1.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    # Add grid
    ax.grid(True, which="both", ls="--", linewidth=0.5, axis='y')
    ax.grid(True, which="major", ls="--", linewidth=0.5, axis='x')

    # Adjust legend properties
    ax.legend(fontsize=fontsize-4, loc='lower left')

    # Adjust layout
    fig.tight_layout()

    # Save figure
    fig.savefig("learning_curves.png", dpi=300)
    #save also as pdf
    fig.savefig("learning_curves.pdf", dpi=300)