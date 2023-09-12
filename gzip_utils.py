import numpy as np
from typing import List, Dict, Any, Tuple
import re
from sklearn.model_selection import train_test_split
from rdkit.Chem.AllChem import MolFromSmiles, MolToSmiles
from pytablewriter import MarkdownTableWriter
from mhfp.encoder import MHFPEncoder
import selfies as sf
import deepsmiles as ds
import deepchem.molnet as mn
import pandas as pd
from pathlib import Path

def sub_sample(
    X: np.array, Y: np.array, p: float = 0.5, seed=666
) -> Tuple[np.array, np.array]:
    X_sample, _, y_sample, _ = train_test_split(
        X,
        Y,
        train_size=int(p * len(X)),
        stratify=Y,
        random_state=seed,
    )

    return X_sample, y_sample


def augment(X: np.array, Y: np.array, n: int = 5) -> Tuple[np.array, np.array]:
    X_aug = []
    y_aug = []

    for x, y in zip(X, Y):
        mol = MolFromSmiles(x)
        for _ in range(n):
            x_rand = MolToSmiles(
                mol,
                canonical=False,
                doRandom=True,
                kekuleSmiles=True,
                allBondsExplicit=True,
                allHsExplicit=True,
            )

            X_aug.append(x_rand)
            y_aug.append(y)

    return np.array(X_aug), np.array(y_aug)



def combined_bin_vectors(X, num_bins):

    """

    Convert a 2D numpy array of vectors into a list of string representations based on variable length binning and delta encoding.

    """

    # Calculate the differences for each vector

    X_diff = np.diff(X, axis=1)
    # Create bins for positive and negative numbers separately
    X_flattened = X_diff.flatten()
    pos_vector = X_flattened[X_flattened >= 0]
    neg_vector = -X_flattened[X_flattened < 0]  # Flip sign for binning

    # Create variable bins
    pos_bins = np.percentile(pos_vector, np.linspace(0, 100, num_bins + 1)) if len(pos_vector) > 0 else np.array([0, 1])
    neg_bins = np.percentile(neg_vector, np.linspace(0, 100, num_bins + 1)) if len(neg_vector) > 0 else np.array([0, 1])

    # Create a mapping from bin number to Unicode character
    bin_to_char = {i+1: chr(9786 + i) for i in range(num_bins)}

    # Apply binning to each vector difference

    string_reps = []

    for vector in X_diff:

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



def delta_variable_bin_vectors(X, num_bins):

    """

    Convert a 2D numpy array of vectors into a list of string representations based on variable length binning and delta encoding.

    """

    # Calculate the differences for each vector

    X_diff = np.diff(X, axis=1)

    

    # Create bins for positive and negative numbers separately

    X_flattened = X_diff.flatten()
    pos_vector = X_flattened[X_flattened >= 0]
    neg_vector = -X_flattened[X_flattened < 0]  # Flip sign for binning



    # Create variable bins
    pos_bins = np.percentile(pos_vector, np.linspace(0, 100, num_bins + 1)) if len(pos_vector) > 0 else np.array([0, 1])
    neg_bins = np.percentile(neg_vector, np.linspace(0, 100, num_bins + 1)) if len(neg_vector) > 0 else np.array([0, 1])



    # Create a mapping from bin number to Unicode character
    bin_to_char = {i+1: chr(9786 + i) for i in range(num_bins)}
    # Apply binning to each vector difference
    string_reps = []

    for vector in X_diff:

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



def variable_bin_vectors(X, num_bins):

    """

    Convert a 2D numpy array of vectors into a list of string representations based on variable length binning.

    """

    # Create bins for positive and negative numbers separately

    X_flattened = X.flatten()
    pos_vector = X_flattened[X_flattened >= 0]
    neg_vector = -X_flattened[X_flattened < 0]  # Flip sign for binning
    # Create variable bins
    pos_bins = np.percentile(pos_vector, np.linspace(0, 100, num_bins + 1)) if len(pos_vector) > 0 else np.array([0, 1])
    neg_bins = np.percentile(neg_vector, np.linspace(0, 100, num_bins + 1)) if len(neg_vector) > 0 else np.array([0, 1])

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


enc = MHFPEncoder()


def to_secfp(
    smiles: str,
    radius: int = 3,
    rings: bool = True,
    kekulize: bool = True,
    min_radius: int = 1,
) -> str:
    return " ".join(
        [
            str(s)
            for s in MHFPEncoder.shingling_from_mol(
                MolFromSmiles(smiles), radius, rings, kekulize, min_radius
            )
        ]
    )


def write_table(results: List) -> None:
    values = []

    for config, result in results:
        values.append(
            [
                config["dataset"],
                config["splitter"],
                result["valid_auroc"],
                result["valid_f1"],
                result["test_auroc"],
                result["test_f1"],
            ]
        )

    writer = MarkdownTableWriter(
        table_name="Results Gzip-based Molecular Classification",
        headers=[
            "Data Set",
            "Split",
            "AUROC/RMSE (Valid)",
            "F1/MAE (Valid)",
            "AUROC/RMSE (Test)",
            "F1/MAE (Test)",
        ],
        value_matrix=values,
    )

    with open("RESULTS.md", "w+") as f:
        writer.stream = f
        writer.write_table()



# Adapted from https://github.com/rxn4chemistry/rxnfp
REGEX = re.compile(
    r"(\%\([0-9]{3}\)|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\||\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"
)


def tokenize(smiles: str) -> List[str]:
    return [token for token in REGEX.findall(smiles)]


def preprocess(smiles: str, preproc: bool = False) -> str:
    if not preproc:
        return smiles
        # return to_secfp(smiles, min_radius=0)
        # return sf.encoder(smiles, strict=False)
        # return ds.Converter(rings=True, branches=True).encode(smiles)

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
    X_valid = np.array([preprocess(x, preproc) for x in valid.ids])
    X_test = np.array([preprocess(x, preproc) for x in test.ids])

    if name == "freesolv" or name == "delaney" or name == "lipo":
        # for regression tasks
        y_train = np.array(train.y, dtype=float)
        y_valid = np.array(valid.y, dtype=float)
        y_test = np.array(test.y, dtype=float)
        print("Task is regression!")
    else:
        # for classification tasks
        y_train = np.array(train.y, dtype=int)
        y_valid = np.array(valid.y, dtype=int)
        y_test = np.array(test.y, dtype=int)
        print("Task is classification!")

    return tasks, X_train, y_train, X_valid, y_valid, X_test, y_test


# Just use the same signature as for molnet_loader... it feels so wrong so it probably is pythonic
def schneider_loader(
    name: str, preproc: bool = False, **kwargs
) -> Tuple[str, np.array, np.array, np.array]:
    base_path = Path(__file__).resolve().parent
    df = pd.read_csv(Path(base_path, "data/schneider50k.tsv.gz"), sep="\t")
    X_train = np.array([row["rxn"] for _, row in df[df.split == "train"].iterrows()])
    y_train = np.array(
        [
            [int(row["rxn_class"].split(".")[0])]
            for _, row in df[df.split == "train"].iterrows()
        ],
        dtype=int,
    )

    X_test = np.array([row["rxn"] for _, row in df[df.split == "test"].iterrows()])
    y_test = np.array(
        [
            [int(row["rxn_class"].split(".")[0])]
            for _, row in df[df.split == "test"].iterrows()
        ],
        dtype=int,
    )

    # Just use test set as valid as no valid set is profided as is
    return ["Reaction Class"], X_train, y_train, X_test, y_test, X_test, y_test