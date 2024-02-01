import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import re
from sklearn.model_selection import train_test_split
from rdkit.Chem.AllChem import MolFromSmiles, MolToSmiles
from pytablewriter import MarkdownTableWriter
import selfies as sf
import deepsmiles as ds
import deepchem.molnet as mn
import pandas as pd
import deepchem as dc
from descriptastorus.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
from pathlib import Path
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

molnet_tasks = {
    "bace": ["Class"],
    "bbbp": ["p_np"],
    "clintox": ["FDA_APPROVED", "CT_TOX"],
    "esol": ["ESOL predicted log solubility in mols per litre"],
    "freesolv": ["expt"],
    "hiv": ["HIV_active"],
    "lipo": ["exp"],
    "muv": [
        "MUV-692",
        "MUV-689",
        "MUV-846",
        "MUV-859",
        "MUV-644",
        "MUV-548",
        "MUV-852",
        "MUV-600",
        "MUV-810",
        "MUV-712",
        "MUV-737",
        "MUV-858",
        "MUV-713",
        "MUV-733",
        "MUV-652",
        "MUV-466",
        "MUV-832",
    ],
    "qm7": ["u0_atom"],
    "qm8": [
        "E1-CC2",
        "E2-CC2",
        "f1-CC2",
        "f2-CC2",
        "E1-PBE0",
        "E2-PBE0",
        "f1-PBE0",
        "f2-PBE0",
        "E1-CAM",
        "E2-CAM",
        "f1-CAM",
        "f2-CAM",
    ],
    "qm9": ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv"],
    "sider": [
        "Hepatobiliary disorders",
        "Metabolism and nutrition disorders",
        "Product issues",
        "Eye disorders",
        "Investigations",
        "Musculoskeletal and connective tissue disorders",
        "Gastrointestinal disorders",
        "Social circumstances",
        "Immune system disorders",
        "Reproductive system and breast disorders",
        "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
        "General disorders and administration site conditions",
        "Endocrine disorders",
        "Surgical and medical procedures",
        "Vascular disorders",
        "Blood and lymphatic system disorders",
        "Skin and subcutaneous tissue disorders",
        "Congenital, familial and genetic disorders",
        "Infections and infestations",
        "Respiratory, thoracic and mediastinal disorders",
        "Psychiatric disorders",
        "Renal and urinary disorders",
        "Pregnancy, puerperium and perinatal conditions",
        "Ear and labyrinth disorders",
        "Cardiac disorders",
        "Nervous system disorders",
        "Injury, poisoning and procedural complications",
    ],
    "tox21": [
        "NR-AR",
        "NR-AR-LBD",
        "NR-AhR",
        "NR-Aromatase",
        "NR-ER",
        "NR-ER-LBD",
        "NR-PPAR-gamma",
        "SR-ARE",
        "SR-ATAD5",
        "SR-HSE",
        "SR-MMP",
        "SR-p53",
    ],
}


@dataclass
class CustomDataset:
    ids: np.ndarray
    y: np.ndarray

    @staticmethod
    def from_df(df, ids_column: str, y_columns: List[str]):
        return CustomDataset(
            df[ids_column].to_numpy(), df[df.columns.intersection(y_columns)].to_numpy()
        )


def _generate_scaffold(smiles, include_chirality=False):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold


def generate_scaffolds(dataset, log_every_n=1000):
    scaffolds = {}
    data_len = len(dataset)
    print(data_len)

    print("About to generate scaffolds")
    for ind, row in dataset.iterrows():
        if ind % log_every_n == 0:
            print("Generating scaffold %d/%d" % (ind, data_len))

        scaffold = _generate_scaffold(row["smiles"])

        # Adapted from original to account for SMILES not readable by MolFromSmiles
        if scaffold is not None:
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [ind]
            else:
                scaffolds[scaffold].append(ind)

    # Sort from largest to smallest scaffold sets
    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set
        for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
        )
    ]
    return scaffold_sets


def scaffold_split(dataset, valid_size, test_size, seed=None, log_every_n=1000):
    train_size = 1.0 - valid_size - test_size
    scaffold_sets = generate_scaffolds(dataset)

    train_cutoff = train_size * len(dataset)
    valid_cutoff = (train_size + valid_size) * len(dataset)
    train_inds: List[int] = []
    valid_inds: List[int] = []
    test_inds: List[int] = []

    print("About to sort in scaffold sets")
    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if len(train_inds) + len(valid_inds) + len(scaffold_set) > valid_cutoff:
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


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


def get_smiles_vec_rep(smiles, config):
    generator = RDKit2DNormalized()

    feature_vectors = []
    for s in smiles:
        feature_vectors.append(generator.process(s)[1:])
    feature_vectors = np.array(feature_vectors)

    # feature_vectors = featurizer_rdkit.featurize(smiles)

    indices_with_nan = [
        i for i, subarray in enumerate(feature_vectors) if np.isnan(subarray).any()
    ]

    if len(indices_with_nan) > 0:
        feature_vectors = np.array(
            [subarray for subarray in feature_vectors if not np.isnan(subarray).any()]
        )
        vectors = bin_vectors(feature_vectors, config["bins"])
    else:
        vectors = bin_vectors(feature_vectors, config["bins"])

    return (
        np.array([np.array(s + x) for s, x in zip(smiles, vectors)]),
        indices_with_nan,
    )


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
    pos_bins = (
        np.percentile(pos_vector, np.linspace(0, 100, num_bins + 1))
        if len(pos_vector) > 0
        else np.array([0, 1])
    )
    neg_bins = (
        np.percentile(neg_vector, np.linspace(0, 100, num_bins + 1))
        if len(neg_vector) > 0
        else np.array([0, 1])
    )

    # Create a mapping from bin number to Unicode character
    bin_to_char = {i + 1: chr(9786 + i) for i in range(num_bins)}

    # Apply binning to each vector difference

    string_reps = []

    for vector in X_diff:
        # Digitize the vectors

        pos_digitized = np.digitize(vector[vector >= 0], pos_bins)
        neg_digitized = np.digitize(-vector[vector < 0], neg_bins)

        # Convert digitized vectors to string representation
        pos_string_rep = [bin_to_char.get(num, "?") for num in pos_digitized]
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

        string_reps.append("".join(string_rep))

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
    pos_bins = (
        np.percentile(pos_vector, np.linspace(0, 100, num_bins + 1))
        if len(pos_vector) > 0
        else np.array([0, 1])
    )
    neg_bins = (
        np.percentile(neg_vector, np.linspace(0, 100, num_bins + 1))
        if len(neg_vector) > 0
        else np.array([0, 1])
    )

    # Create a mapping from bin number to Unicode character
    bin_to_char = {i + 1: chr(9786 + i) for i in range(num_bins)}
    # Apply binning to each vector difference
    string_reps = []

    for vector in X_diff:
        # Digitize the vectors

        pos_digitized = np.digitize(vector[vector >= 0], pos_bins)
        neg_digitized = np.digitize(-vector[vector < 0], neg_bins)

        # Convert digitized vectors to string representation

        pos_string_rep = [bin_to_char.get(num, "?") for num in pos_digitized]
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

        string_reps.append("".join(string_rep))

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
    pos_bins = (
        np.percentile(pos_vector, np.linspace(0, 100, num_bins + 1))
        if len(pos_vector) > 0
        else np.array([0, 1])
    )
    neg_bins = (
        np.percentile(neg_vector, np.linspace(0, 100, num_bins + 1))
        if len(neg_vector) > 0
        else np.array([0, 1])
    )

    # Create a mapping from bin number to Unicode character
    bin_to_char = {i + 1: chr(9786 + i) for i in range(num_bins)}

    # Apply binning to each vector
    string_reps = []

    for vector in X:
        # Digitize the vectors

        pos_digitized = np.digitize(vector[vector >= 0], pos_bins)
        neg_digitized = np.digitize(-vector[vector < 0], neg_bins)

        # Convert digitized vectors to string representation
        pos_string_rep = [bin_to_char.get(num, "?") for num in pos_digitized]
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

        string_reps.append("".join(string_rep))
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

    pos_bins = np.linspace(
        0, max(pos_vector) if len(pos_vector) > 0 else 1, num_bins + 1
    )
    neg_bins = np.linspace(
        0, max(neg_vector) if len(neg_vector) > 0 else 1, num_bins + 1
    )

    # Create a mapping from bin number to Unicode character
    bin_to_char = {i + 1: chr(9786 + i) for i in range(num_bins)}

    # Apply binning to each vector
    string_reps = []
    for vector in X:
        # Digitize the vectors
        pos_digitized = np.digitize(vector[vector >= 0], pos_bins)
        neg_digitized = np.digitize(-vector[vector < 0], neg_bins)

        # Convert digitized vectors to string representation
        pos_string_rep = [bin_to_char.get(num, "?") for num in pos_digitized]
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

        string_reps.append("".join(string_rep))

    return string_reps


def write_table(results: List) -> None:
    values = []

    for config, result in results:
        values.append(
            [
                config["dataset"],
                config["task"],
                config["splitter"],
                result["valid_auroc"],
                result["valid_f1"],
                result["valid_r"],
                result["test_auroc"],
                result["test_f1"],
                result["test_r"],
            ]
            + [v for k, v in config["result_props"].items()]
        )

    writer = MarkdownTableWriter(
        table_name="Results Gzip-based Molecular Classification",
        headers=[
            "Data Set",
            "Task",
            "Split",
            "AUROC/RMSE (Valid)",
            "F1/MAE (Valid)",
            "-/R (Valid)",
            "AUROC/RMSE (Test)",
            "F1/MAE (Test)",
            "-/R (Test)",
        ]
        + [k for k, v in config["result_props"].items()],
        value_matrix=values,
    )

    with open("results.csv", "w+") as f:
        result_props_head = ""

        if "result_props" in results[0][0]:
            result_props_head = ",".join(list(results[0][0]["result_props"].keys()))
            result_props_head = f",{result_props_head}"

        f.write(
            f"dataset,task,splitter,valid_auroc,valid_f1,valid_r,test_auroc,test_f1,test_r{result_props_head}\n"
        )
        for config, result in results:
            result_props_vals = ""

            if "result_props" in config:
                result_props_vals = ",".join(
                    map(str, list(config["result_props"].values()))
                )

            result_props_vals = f",{result_props_vals}"

            f.write(
                ",".join(
                    [
                        config["dataset"],
                        config["task"],
                        config["splitter"],
                        result["valid_auroc"].split(" ")[0],
                        result["valid_f1"].split(" ")[0],
                        result["valid_r"].split(" ")[0],
                        result["test_auroc"].split(" ")[0],
                        result["test_f1"].split(" ")[0],
                        result["test_r"].split(" ")[0],
                    ]
                )
                + f"{result_props_vals}\n"
            )
        ...

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

    smiles = MolToSmiles(
        MolFromSmiles(smiles),
        kekuleSmiles=True,
        allBondsExplicit=True,
        allHsExplicit=True,
    )

    return " ".join(tokenize(smiles))


def molnet_loader(
    name: str, preproc: bool = False, task_name: str = None, **kwargs
) -> Tuple[str, np.array, np.array, np.array]:
    mn_loader = getattr(mn, f"load_{name}")
    dc_set = mn_loader(**kwargs)

    tasks, dataset, _ = dc_set
    train, valid, test = dataset

    y_train = train.y
    y_valid = valid.y
    y_test = test.y

    task_idx = tasks.index(task_name) if task_name in tasks else -1
    task_idx = 0

    if task_idx >= 0:
        y_train = np.expand_dims(y_train[:, task_idx], axis=1)
        y_valid = np.expand_dims(y_valid[:, task_idx], axis=1)
        y_test = np.expand_dims(y_test[:, task_idx], axis=1)

    X_train = np.array([preprocess(x, preproc) for x in train.ids])
    X_valid = np.array([preprocess(x, preproc) for x in valid.ids])
    X_test = np.array([preprocess(x, preproc) for x in test.ids])

    if name in ["freesolv", "delaney", "lipo", "bace_regression", "sample"]:
        # for regression tasks
        y_train = np.array(y_train, dtype=float)
        y_valid = np.array(y_valid, dtype=float)
        y_test = np.array(y_test, dtype=float)
    else:
        # for classification tasks
        y_train = np.array(y_train, dtype=int)
        y_valid = np.array(y_valid, dtype=int)
        y_test = np.array(y_test, dtype=int)

    return tasks, X_train, y_train, X_valid, y_valid, X_test, y_test


def local_molnet_loader(
    name: str, featurizer=None, split_ratio=0.7, seed=42, task_name=None, **kwargs
):
    root_path = Path(__file__).resolve().parent
    file_path = Path(root_path, f"data/moleculenet/{name}.csv.xz")

    df = pd.read_csv(file_path)

    # Drop NAs. Needed in Tox21
    if name in ["tox21"]:
        df = df.replace("", np.nan)
        if task_name:
            df = df.dropna(subset=[task_name])
        else:
            df = df.dropna()

    train_ids, valid_ids, test_ids = scaffold_split(df, 0.1, 0.1, seed)

    train = df.loc[train_ids]
    valid = df.loc[valid_ids]
    test = df.loc[test_ids]

    tasks = molnet_tasks[name]

    cd_train = CustomDataset.from_df(train, "smiles", tasks)
    cd_valid = CustomDataset.from_df(valid, "smiles", tasks)
    cd_test = CustomDataset.from_df(test, "smiles", tasks)

    return (
        tasks,
        cd_train.ids,
        cd_train.y,
        cd_valid.ids,
        cd_valid.y,
        cd_test.ids,
        cd_test.y,
    )


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


def adme_loader(name: str, preproc: bool = False, **kwargs):
    _, task_name = name.split("-")
    print(task_name)

    base_path = Path(__file__).resolve().parent
    adme_train_file = Path(base_path, f"data/adme/ADME_{task_name}_train.csv")
    adme_test_file = Path(base_path, f"data/adme/ADME_{task_name}_test.csv")

    train = pd.read_csv(adme_train_file)
    test = pd.read_csv(adme_test_file)

    # Validation samples are not needed
    valid = train.sample(frac=0.1)

    X_train = np.array([preprocess(x, preproc) for x in train["smiles"]])
    X_valid = np.array([preprocess(x, preproc) for x in valid["smiles"]])
    X_test = np.array([preprocess(x, preproc) for x in test["smiles"]])

    y_train = np.expand_dims(np.array(train["activity"], dtype=float), axis=1)
    y_valid = np.expand_dims(np.array(valid["activity"], dtype=float), axis=1)
    y_test = np.expand_dims(np.array(test["activity"], dtype=float), axis=1)

    tasks = ["activity"]

    return tasks, X_train, y_train, X_valid, y_valid, X_test, y_test


def pdbbind_loader(
    name: str, preproc: bool = False, properties: Optional[List[str]] = None, **kwargs
):
    if properties is None or len(properties) == 0:
        properties = ["ligand_smiles"]

    root_path = Path(__file__).resolve().parent
    meta_path = Path(root_path, "data/pdbbind/meta.csv")

    data = {"train": [[], []], "valid": [[], []], "test": [[], []]}
    df = pd.read_csv(meta_path)

    for _, row in df.iterrows():
        data[row["split"]][0].append(
            " ".join([v for k, v in row.items() if k in properties])
        )
        data[row["split"]][1].append([row["label"]])

    return (
        ["-logKd/Ki"],
        np.array(data["train"][0]),
        np.array(data["train"][1]),
        np.array(data["valid"][0]),
        np.array(data["valid"][1]),
        np.array(data["test"][0]),
        np.array(data["test"][1]),
    )
