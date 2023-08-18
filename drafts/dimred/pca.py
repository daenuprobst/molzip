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
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA

random.seed(42)

def smiles_to_fps(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    else:
        return None


DATASET = "tox21" #"clintox" #tox21
config =  {
                 "dataset": f"{DATASET}",
                 "splitter": "random",
                 "task": "classification",
                 "k": 5,
                 "augment": 0,
                 "preprocess": False,
                 "sub_sample": 0.0,
                 "is_imbalanced": False,
                 "n": 1,
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

if len(tasks) > 1:
    if DATASET == "clintox":
        task_ind = 1
    else:
        task_ind = 0
else:
    task_ind = 0


N = 2500
y_train, y_valid, y_test = y_train[:, task_ind][:N], y_valid[:, task_ind][:N], y_test[:, task_ind][:N]
#also convert to morgen fingerprints and compare strucuture of latent space

X_set = np.concatenate((X_train,X_valid,X_test))[:N]
y_set = np.concatenate((y_train,y_valid,y_test))[:N]
X_transformed = ncd_pca(X_set, n_components=2)

fps = [smiles_to_fps(s) for s in X_set]
fps_array = np.array([list(fp) for fp in fps if fp is not None])

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(fps_array)




if len(tasks) > 1:
    if DATASET == "clintox":
        task = tasks[1]
    else:
        task = tasks[0]
else:
    task = "G [kcal/mol]"

fig, ax = plt.subplots(figsize=(10, 8), dpi=100) 

fontsize = 26

sc = ax.scatter(X_transformed[:, 0], X_transformed[:, 1] , c=y_set, s=50, alpha=0.7, edgecolors='w')
#add a legend 
legend= ax.legend(*sc.legend_elements(), loc="lower left", fontsize=fontsize-2)

if task == "G [kcal/mol]":
    cbar = plt.colorbar(sc, label=task)
    cbar.ax.tick_params(labelsize=16)  # Increase colorbar label size
    cbar.set_label(task, size=21)
# Label the axes and increase label size
ax.set_xlabel("PC1", fontsize=fontsize)
ax.set_ylabel("PC2", fontsize=fontsize)

# Increase the size of the axis ticks
ax.tick_params(axis='both', which='major', labelsize=fontsize-2)

# Add grid for better readability of the plot
ax.grid(True, linestyle='--', alpha=0.6)
#make tight layout
plt.tight_layout()
plt.savefig(f"PCA_{DATASET}.pdf")
plt.close()

fig2, ax2 = plt.subplots(figsize=(8, 8), dpi=300)
sc2 = ax2.scatter(reduced_data[:, 0], reduced_data[:, 1] , c=y_set, s=50, alpha=0.7, edgecolors='w')
#legend 

#legend2= ax2.legend(*sc2.legend_elements(), loc="lower left", title=task, fontsize=21)
legend2= ax2.legend(*sc2.legend_elements(), loc="upper left", fontsize=fontsize-2)



if task  == "G [kcal/mol]":
    cbar2 = plt.colorbar(sc2, label=task)
    cbar2.ax.tick_params(labelsize=fontsize)  # Increase colorbar label size
    cbar2.set_label(task, size=fontsize)
# Label the axes and increase label size
ax2.set_xlabel("PC1", fontsize=fontsize)
ax2.set_ylabel("PC2", fontsize=fontsize)

# Increase the size of the axis ticks
ax2.tick_params(axis='both', which='major', labelsize=fontsize)

# Add grid for better readability of the plot
ax2.grid(True, linestyle='--', alpha=0.6)
#make tight layout
plt.tight_layout()
plt.savefig(f"PCA_{DATASET}_fps.pdf")
plt.close()