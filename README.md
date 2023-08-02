[![DOI](https://zenodo.org/badge/666335439.svg)](https://zenodo.org/badge/latestdoi/666335439)

# Parameter-Free Molecular Classification and Regression with Gzip
### Daniel Probst<sup>1</sup>, You?
<sup>1</sup>Institute of Electrical and Micro Engineering, LTS2, EPFL

## Abstract
TBD

## Introduction
The classification of a molecule on a wide variety of physicochemical and pharmakological properties, such as solubility, efficacy against specific diseases, or toxicity, has become a task of high interest in chemistry and biology. With the rise of deep learning during tha past decade, molecular classification has increasingly be carried out by ever-larger models, with mixed results. The newly published parameter-free text classification approach that makes use of Gzip compression has shown excellent performance compared to deep learning architectures, such as transformers, on benchmark data sets.[^1] As the SMILES string encoding of molecular graphs has been shown to be a well-performing molecular representation for applying NLP methods, such as transformers, to chemical tasks including molecular classification, a comparison with the Gzip-based classification method is also relevant in the context of molecular classification.

## Methods
The Gzip-based classifier introduced in this article has been adapted from the implementation presented by Jiang et al. and differs in three points: (1) as, as the authors have noted, the Gzip-based classification method has a relatively high time complexity, multiprocessing has been added; (2) multi-task classification has been added; and (3) a class weighing scheme has been implemented to account for unbalanced data. Furthermore, the capability to preprocess data, in this case the SMILES strings, has been added to the calling program.

## Results
The current results are presented in the table below. Data sets with random splits were ran a total of four times.

|     Data Set      | Split  | AUROC (Valid) |  F1 (Valid)   | AUROC (Test)  |   F1 (Test)   |
|-------------------|--------|---------------|---------------|---------------|---------------|
|bbbp               |scaffold|0.891 +/- 0.0  |0.902 +/- 0.0  |0.679 +/- 0.0  |0.686 +/- 0.0  |
|bace_classification|random  |0.793 +/- 0.038|0.793 +/- 0.038|0.789 +/- 0.038|0.789 +/- 0.038|
|clintox            |random  |0.805 +/- 0.038|0.965 +/- 0.038|0.77 +/- 0.038 |0.958 +/- 0.038|
|tox21              |random  |0.6 +/- 0.007  |0.308 +/- 0.007|0.599 +/- 0.007|0.303 +/- 0.007|
|sider              |random  |0.56 +/- 0.007 |0.788 +/- 0.007|0.563 +/- 0.007|0.778 +/- 0.007|

Implementing a weighted version of the kNN algorithm does not necessary lead to better classification performance on unbalanced data sets.
|     Data Set      | Split  |AUROC/RMSE (Valid)|F1/MAE (Valid) |AUROC/RMSE (Test)| F1/MAE (Test) |
|-------------------|--------|------------------|---------------|-----------------|---------------|
|sider              |scaffold|0.551 +/- 0.0     |0.707 +/- 0.0  |0.577 +/- 0.0    |0.666 +/- 0.0  |
|sider              |random  |0.454 +/- 0.262   |0.657 +/- 0.262|0.581 +/- 0.262  |0.647 +/- 0.262|
|bbbp               |scaffold|0.931 +/- 0.0     |0.931 +/- 0.0  |0.639 +/- 0.0    |0.627 +/- 0.0  |
|bace_classification|scaffold|0.694 +/- 0.0     |0.702 +/- 0.0  |0.701 +/- 0.0    |0.697 +/- 0.0  |
|bace_classification|random  |0.817 +/- 0.005   |0.815 +/- 0.005|0.774 +/- 0.005  |0.771 +/- 0.005|
|clintox            |scaffold|0.805 +/- 0.0     |0.854 +/- 0.0  |0.891 +/- 0.0    |0.891 +/- 0.0  |
|clintox            |random  |0.925 +/- 0.032   |0.924 +/- 0.032|0.913 +/- 0.032  |0.91 +/- 0.032 |
|tox21              |scaffold|0.635 +/- 0.0     |0.247 +/- 0.0  |0.618 +/- 0.0    |0.227 +/- 0.0  |
|tox21              |random  |0.705 +/- 0.006   |0.295 +/- 0.006|0.694 +/- 0.006  |0.29 +/- 0.006 |
|hiv                |scaffold|0.714 +/- 0.0     |0.901 +/- 0.0  |0.689 +/- 0.0    |0.887 +/- 0.0  |

Using SECFP (ECFP-style circular substructures as SMILES) doesn't increase the classification performance of the weighted kNN.

|     Data Set      | Split  | AUROC (Valid) |  F1 (Valid)   | AUROC (Test)  |   F1 (Test)   |
|-------------------|--------|---------------|---------------|---------------|---------------|
|bbbp               |scaffold|0.83 +/- 0.0   |0.819 +/- 0.0  |0.632 +/- 0.0  |0.627 +/- 0.0  |
|bace_classification|random  |0.833 +/- 0.015|0.829 +/- 0.015|0.826 +/- 0.015|0.821 +/- 0.015|
|clintox            |random  |0.74 +/- 0.076 |0.831 +/- 0.076|0.747 +/- 0.076|0.84 +/- 0.076 |
|tox21              |random  |0.712 +/- 0.011|0.305 +/- 0.011|0.718 +/- 0.011|0.31 +/- 0.011 |
|sider              |random  |0.604 +/- 0.022|0.62 +/- 0.022 |0.614 +/- 0.022|0.624 +/- 0.022|

Implementing a GZip-based regressor (weighted kNN, k=10) shows performance comparable to baseline performance of common ML implementations from MoleculeNet (https://moleculenet.org/full-results).
Interestingly there are improvements when the SMILES are tokenised.

|Data Set|Split |AUROC/RMSE (Valid)|F1/MAE (Valid) |AUROC/RMSE (Test)| F1/MAE (Test) |
|--------|------|------------------|---------------|-----------------|---------------|
|freesolv|random|0.641 +/- 0.144   |0.375 +/- 0.144|0.527 +/- 0.144  |0.321 +/- 0.144|
|delaney |random|1.443 +/- 0.088   |1.097 +/- 0.088|1.283 +/- 0.088  |0.966 +/- 0.088|
|lipo    |random|0.938 +/- 0.042   |0.765 +/- 0.042|0.911 +/- 0.042  |0.727 +/- 0.042|

The classifier is also able to classify raw reaction SMILES from the Schneider50k data set (no class weighting).

|Data Set |Split |AUROC/RMSE (Valid)|F1/MAE (Valid)|AUROC/RMSE (Test)|F1/MAE (Test)|
|---------|------|------------------|--------------|-----------------|-------------|
|schneider|random|0.0 +/- 0.0       |0.801 +/- 0.0 |0.0 +/- 0.0      |0.801 +/- 0.0|


## Discussion
TBD

## References
[^1] https://arxiv.org/abs/2212.09410

# What is this?
This is an experiment for a small open source manuscript/article that aims to validate and evaluate the performance of compression-based molecular classification using Gzip. If you want to join/help out, leave a message or a pull request that includes your name and, if available, your affiliation.
