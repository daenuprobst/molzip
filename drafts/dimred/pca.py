import pdb
from xyz2mol import *
import pandas as pd
from gzip_regressor import ncd_pca
from smiles_tokenizer import tokenize
import numpy as np
import matplotlib.pyplot as plt
import random
import deepchem.molnet as mn
from typing import List, Dict, Any, Tuple
from rdkit.Chem.AllChem import MolFromSmiles, MolToSmiles, MolToInchi
random.seed(42)

DATASET = "tox21" #tox21
config =  {
                 "dataset": f"{DATASET}",
                 "splitter": "random",
                 "task": "regression", #regression
                 "k": 25,
                 "augment": 0,
                 "preprocess": False,
                 "sub_sample": 0.0,
                 "is_imbalanced": False,
                 "n": 4,
             }

def preprocess(smiles: str, preproc: bool = False) -> str:
    if not preproc:
        return smiles

    else:
        smiles = MolToSmiles(
            MolFromSmiles(smiles),
            kekuleSmiles=True,
            allBondsExplicit=True,
            allHsExplicit=True,
        )
        return " ".join(tokenize(smiles))

def molnet_loader(
    name: str, preproc: bool = False, **kwargs
) -> Tuple[str, np.array, np.array, np.array]:
    mn_loader = getattr(mn, f"load_{name}")
    dc_set = mn_loader(**kwargs)

    tasks, dataset, _ = dc_set
    train, valid, test = dataset
    X_train = np.array([preprocess(x, preproc) for x in train.ids])
    y_train = np.array(train.y, dtype=float)

    X_valid = np.array([preprocess(x, preproc) for x in valid.ids])
    y_valid = np.array(valid.y, dtype=float)

    X_test = np.array([preprocess(x, preproc) for x in test.ids])
    y_test = np.array(test.y, dtype=float)

    return tasks, X_train, y_train, X_valid, y_valid, X_test, y_test

loader = molnet_loader


run_results = []
for _ in range(config["n"]):
    tasks, X_train, y_train, X_valid, y_valid, X_test, y_test = loader(
        config["dataset"],
        splitter=config["splitter"],
        preproc=config["preprocess"],
        reload=False,
        transformers=[],
    )

N = 2500
X_train, X_valid, X_test = X_train[:N], X_valid[:N], X_test[:N]
if len(tasks) > 1:
    if DATASET == "clintox":
        task_ind = 1
    else:
        task_ind = 0
else:
    task_ind = 0

y_train, y_valid, y_test = y_train[:, task_ind][:N], y_valid[:, task_ind][:N], y_test[:, task_ind][:N]
X_transformed = ncd_pca(np.concatenate((X_train,X_valid,X_test)), n_components=2)
if len(tasks) > 1:
    if DATASET == "clintox":
        task = tasks[1]
    else:
        task = tasks[0]
else:
    task = "G [kcal/mol]"

fig, ax = plt.subplots(figsize=(10, 8), dpi=100) 
sc = ax.scatter(X_transformed[:, 0], X_transformed[:, 1] , c=np.concatenate((y_train,y_valid,y_test)), s=50, alpha=0.7, edgecolors='w')

cbar = plt.colorbar(sc, label=task)
cbar.ax.tick_params(labelsize=14)  # Increase colorbar label size
cbar.set_label(task, size=16)
# Label the axes and increase label size
ax.set_xlabel("PC1", fontsize=16)
ax.set_ylabel("PC2", fontsize=16)

# Increase the size of the axis ticks
ax.tick_params(axis='both', which='major', labelsize=14)

# Add grid for better readability of the plot
ax.grid(True, linestyle='--', alpha=0.6)

plt.savefig(f"PCA_{DATASET}.png")