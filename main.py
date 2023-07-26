from typing import List, Dict, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import deepchem.molnet as mn
import selfies as sf
import deepsmiles as ds

from mhfp.encoder import MHFPEncoder

from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from pytablewriter import MarkdownTableWriter

from rdkit.Chem.AllChem import MolFromSmiles, MolToSmiles, MolToInchi
from rdkit import rdBase

blocker = rdBase.BlockLogs()

from gzip_classifier import classify
from gzip_regressor import regress
from smiles_tokenizer import tokenize

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
    y_train = np.array(train.y, dtype=int)

    X_valid = np.array([preprocess(x, preproc) for x in valid.ids])
    y_valid = np.array(valid.y, dtype=int)

    X_test = np.array([preprocess(x, preproc) for x in test.ids])
    y_test = np.array(test.y, dtype=int)

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


def benchmark(configs: List[Dict[str, Any]]) -> None:
    results = []

    for config in configs:
        loader = molnet_loader

        if config["dataset"] in ["schneider"]:
            loader = schneider_loader

        run_results = []
        for _ in range(config["n"]):
            tasks, X_train, y_train, X_valid, y_valid, X_test, y_test = loader(
                config["dataset"],
                splitter=config["splitter"],
                preproc=config["preprocess"],
                reload=False,
                transformers=[],
            )

            if config["augment"] > 0:
                X_train, y_train = augment(X_train, y_train, config["augment"])

            if config["sub_sample"] > 0.0:
                X_train, y_train = sub_sample(X_train, y_train, config["sub_sample"])

            if config["task"] == "classification":
                # Get class weights
                class_weights = []
                if config["is_imbalanced"]:
                    for y_task in y_train.T:
                        class_weights.append(
                            compute_class_weight(
                                "balanced", classes=sorted(list(set(y_task))), y=y_task
                            )
                        )

                # Run classification
                valid_preds = classify(
                    X_train, y_train, X_valid, config["k"], class_weights
                )
                test_preds = classify(
                    X_train, y_train, X_test, config["k"], class_weights
                )

                # Compute metrics
                try:
                    valid_auroc = roc_auc_score(y_valid, valid_preds, multi_class="ovr")
                except:
                    valid_auroc = 0.0

                valid_f1 = f1_score(
                    y_valid,
                    valid_preds,
                    average="micro",
                )

                try:
                    test_auroc = roc_auc_score(y_test, test_preds, multi_class="ovr")
                except:
                    test_auroc = 0.0

                test_f1 = f1_score(
                    y_test,
                    test_preds,
                    average="micro",
                )

                print(f"\n{config['dataset']} ({len(tasks)} tasks)")
                print(config)
                print(
                    f"Valid AUROC: {valid_auroc}, Valid F1: {valid_f1} , Test AUROC: {test_auroc}, Test F1: {test_f1}"
                )

                run_results.append([valid_auroc, valid_f1, test_auroc, test_f1])
            else:
                valid_preds = regress(X_train, y_train, X_valid, config["k"])
                test_preds = regress(X_train, y_train, X_test, config["k"])

                # Compute metrics
                valid_rmse = mean_squared_error(y_valid, valid_preds, squared=False)
                valid_mae = mean_absolute_error(
                    y_valid,
                    valid_preds,
                )
                test_rmse = mean_squared_error(y_test, test_preds, squared=False)
                test_mae = mean_absolute_error(
                    y_test,
                    test_preds,
                )

                print(f"\n{config['dataset']} ({len(tasks)} tasks)")
                print(config)
                print(
                    f"Valid RMSE: {valid_rmse}, Valid MAE: {valid_mae} , Test RMSE: {test_rmse}, Test MAE: {test_mae}"
                )

                run_results.append([valid_rmse, valid_mae, test_rmse, test_mae])

        run_results = np.array(run_results)
        results_means = np.mean(run_results, axis=0)
        results_stds = np.std(run_results, axis=0)

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

        write_table(results)


def main():
    benchmark(
        [
            {
                "dataset": "freesolv",
                "splitter": "random",
                "task": "regression",
                "k": 10,
                "augment": 0,
                "preprocess": False,
                "sub_sample": 0.0,
                "is_imbalanced": True,
                "n": 4,
            },
            {
                "dataset": "delaney",
                "splitter": "random",
                "task": "regression",
                "k": 10,
                "augment": 0,
                "preprocess": False,
                "sub_sample": 0.0,
                "is_imbalanced": True,
                "n": 4,
            },
            {
                "dataset": "lipo",
                "splitter": "random",
                "task": "regression",
                "k": 10,
                "augment": 0,
                "preprocess": False,
                "sub_sample": 0.0,
                "is_imbalanced": True,
                "n": 4,
            },
            # {
            #     "dataset": "hiv",
            #     "splitter": "random",
            #     "task": "classification",
            #     "k": 5,
            #     "augment": 0,
            #     "preprocess": False,
            #     "sub_sample": 0.0,
            #     "is_imbalanced": True,
            #     "n": 4,
            # },
            # {
            #     "dataset": "sider",
            #     "splitter": "random",
            #     "task": "classification",
            #     "k": 5,
            #     "augment": 0,
            #     "preprocess": False,
            #     "sub_sample": 0.0,
            #     "is_imbalanced": True,
            #     "n": 4,
            # },
            # {
            #     "dataset": "bbbp",
            #     "splitter": "scaffold",
            #     "task": "classification",
            #     "k": 5,
            #     "augment": 0,
            #     "preprocess": False,
            #     "sub_sample": 0.0,
            #     "is_imbalanced": True,
            #     "n": 1,
            # },
            # {
            #     "dataset": "bace_classification",
            #     "splitter": "random",
            #     "task": "classification",
            #     "k": 5,
            #     "augment": 0,
            #     "preprocess": False,
            #     "sub_sample": 0.0,
            #     "is_imbalanced": True,
            #     "n": 4,
            # },
            # {
            #     "dataset": "clintox",
            #     "splitter": "random",
            #     "task": "classification",
            #     "k": 5,
            #     "augment": 0,
            #     "preprocess": False,
            #     "sub_sample": 0.0,
            #     "is_imbalanced": True,
            #     "n": 4,
            # },
            # {
            #     "dataset": "tox21",
            #     "splitter": "random",
            #     "task": "classification",
            #     "k": 5,
            #     "augment": 0,
            #     "preprocess": False,
            #     "sub_sample": 0.0,
            #     "is_imbalanced": True,
            #     "n": 4,
            # },
            # {
            #     "dataset": "muv",
            #     "splitter": "random",
            #     "task": "classification",
            #     "k": 5,
            #     "augment": 0,
            #     "preprocess": False,
            #     "sub_sample": 0.0,
            #     "is_imbalanced": True,
            #     "n": 4,
            # },
            # {
            #     "dataset": "schneider",
            #     "splitter": "random",
            #     "task": "classification",
            #     "k": 5,
            #     "augment": 0,
            #     "preprocess": False,
            #     "sub_sample": 0.0,
            #     "is_imbalanced": False,
            #     "n": 1,
            # },
        ]
    )


if __name__ == "__main__":
    main()
