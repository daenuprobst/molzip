from typing import List, Dict, Any, Tuple
import numpy as np
import deepchem.molnet as mn

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from pytablewriter import MarkdownTableWriter

from rdkit.Chem.AllChem import MolFromSmiles, MolToSmiles, MolToInchi
from rdkit import rdBase

blocker = rdBase.BlockLogs()

from gzip_classifier import classify
from smiles_tokenizer import tokenize


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
            "AUROC (Valid)",
            "F1 (Valid)",
            "AUROC (Test)",
            "F1 (Test)",
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

    smiles = MolToSmiles(
        MolFromSmiles(smiles),
        kekuleSmiles=True,
        allBondsExplicit=True,
        allHsExplicit=True,
    )
    return " ".join(tokenize(smiles))


def molnet_loader(
    name: str, preproc: bool = False, **kwargs
) -> List[Tuple[str, np.array, np.array, np.array]]:
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


def benchmark(configs: List[Dict[str, Any]]) -> None:
    results = []

    for config in configs:
        # Load data sets

        run_results = []
        for _ in range(config["n"]):
            tasks, X_train, y_train, X_valid, y_valid, X_test, y_test = molnet_loader(
                config["dataset"],
                splitter=config["splitter"],
                preproc=config["preprocess"],
                reload=False,
            )

            if config["augment"] > 0:
                X_train, y_train = augment(X_train, y_train, config["augment"])

            if config["sub_sample"] > 0.0:
                X_train, y_train = sub_sample(X_train, y_train, config["sub_sample"])

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
            test_preds = classify(X_train, y_train, X_test, config["k"], class_weights)

            # Compute metrics
            valid_auroc = roc_auc_score(y_valid, valid_preds)
            valid_f1 = f1_score(
                y_valid,
                valid_preds,
                average="micro",
            )
            test_auroc = roc_auc_score(y_test, test_preds)
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
                "dataset": "bbbp",
                "splitter": "scaffold",
                "k": 5,
                "augment": 0,
                "preprocess": False,
                "sub_sample": 0.5,
                "is_imbalanced": True,
                "n": 1,
            },
            {
                "dataset": "bace_classification",
                "splitter": "random",
                "k": 5,
                "augment": 0,
                "preprocess": False,
                "sub_sample": 0.0,
                "is_imbalanced": True,
                "n": 4,
            },
            {
                "dataset": "clintox",
                "splitter": "random",
                "k": 5,
                "augment": 0,
                "preprocess": False,
                "sub_sample": 0.0,
                "is_imbalanced": True,
                "n": 4,
            },
            {
                "dataset": "tox21",
                "splitter": "random",
                "k": 5,
                "augment": 0,
                "preprocess": False,
                "sub_sample": 0.0,
                "is_imbalanced": True,
                "n": 4,
            },
            {
                "dataset": "sider",
                "splitter": "random",
                "k": 5,
                "augment": 0,
                "preprocess": False,
                "sub_sample": 0.0,
                "is_imbalanced": True,
                "n": 4,
            },
        ]
    )


if __name__ == "__main__":
    main()
