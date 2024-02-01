from molzip.transforms import DeepsmilesTransform, SelfiesTransform, AugmentTransform
from molzip.compressors import (
    SmiZipCompressor,
    LZMACompressor,
    BrotliCompressor,
    SnappyCompressor,
    LZ4Compressor,
    GzipCompressor,
)


def get_all_tests():
    # return [

    #     # {
    #     #     "dataset": "pdbbind",
    #     #     "splitter": "scaffold",
    #     #     "task": "regression",
    #     #     "vector": True,
    #     #     "bins": 256,
    #     #     "k": 5,
    #     #     "augment": 0,
    #     #     "preprocess": False,
    #     #     "sub_sample": 0.0,
    #     #     "is_imbalanced": False,
    #     #     "properties": ["ligand_smiles", "pocket_seq"],
    #     #     "transforms": [AugmentTransform(1)],
    #     #     "n": 1,
    #     # },
    # ]

    all_tests = []

    # all_tests.append(
    #     {
    #         "dataset": "bace_classification",
    #         "splitter": "scaffold",
    #         "task": "classification",
    #         "k": 25,
    #         "augment": 0,
    #         "preprocess": False,
    #         "sub_sample": 0.0,
    #         "is_imbalanced": True,
    #         "n": 1,
    #     }
    # )

    # all_tests.append(
    #     {
    #         "dataset": "bace_classification",
    #         "splitter": "scaffold",
    #         "task": "classification_vec",
    #         "k": 25,
    #         "bins": 256,
    #         "vector": True,
    #         "is_imbalanced": True,
    #         "n": 1,
    #     }
    # )

    # all_tests.append(
    #     {
    #         "dataset": "bbbp",
    #         "splitter": "scaffold",
    #         "task": "classification",
    #         "k": 25,
    #         "augment": 0,
    #         "preprocess": False,
    #         "sub_sample": 0.0,
    #         "is_imbalanced": True,
    #         "n": 1,
    #     }
    # )

    # all_tests.append(
    #     {
    #         "dataset": "bbbp",
    #         "splitter": "scaffold",
    #         "task": "classification_vec",
    #         "k": 25,
    #         "bins": 256,
    #         "vector": True,
    #         "is_imbalanced": True,
    #         "n": 1,
    #     }
    # )

    # all_tests.append(
    #     {
    #         "dataset": "clintox",
    #         "splitter": "scaffold",
    #         "task": "classification",
    #         "task_name": "CT_TOX",
    #         "k": 25,
    #         "is_imbalanced": True,
    #         "n": 1,
    #     }
    # )

    # all_tests.append(
    #     {
    #         "dataset": "clintox",
    #         "splitter": "scaffold",
    #         "task": "classification_vec",
    #         "task_name": "CT_TOX",
    #         "k": 25,
    #         "bins": 256,
    #         "vector": True,
    #         "is_imbalanced": True,
    #         "n": 1,
    #     }
    # )

    # all_tests.append(
    #     {
    #         "dataset": "tox21",
    #         "splitter": "scaffold",
    #         "task": "classification",
    #         "task_name": "SR-p53",
    #         "k": 25,
    #         "is_imbalanced": False,
    #         "n": 1,
    #     }
    # )

    # all_tests.append(
    #     {
    #         "dataset": "tox21",
    #         "splitter": "scaffold",
    #         "task": "classification_vec",
    #         "task_name": "SR-p53",
    #         "k": 25,
    #         "bins": 256,
    #         "vector": True,
    #         "is_imbalanced": False,
    #         "n": 1,
    #     }
    # )

    # all_tests.append(
    #     {
    #         "dataset": "delaney",
    #         "splitter": "scaffold",
    #         "task": "regression",
    #         "k": 5,
    #         "is_imbalanced": False,
    #         "n": 1,
    #     }
    # )
    # all_tests.append(
    #     {
    #         "dataset": "delaney",
    #         "splitter": "scaffold",
    #         "task": "regression_vec",
    #         "k": 5,
    #         "is_imbalanced": False,
    #         "n": 1,
    #         "bins": 256,
    #         "vector": True,
    #     }
    # )

    # all_tests.append(
    #     {
    #         "dataset": "bace_regression",
    #         "splitter": "scaffold",
    #         "task": "regression",
    #         "k": 5,
    #         "is_imbalanced": False,
    #         "n": 1,
    #     }
    # )

    # all_tests.append(
    #     {
    #         "dataset": "bace_regression",
    #         "splitter": "scaffold",
    #         "task": "regression_vec",
    #         "k": 5,
    #         "is_imbalanced": False,
    #         "n": 1,
    #         "bins": 256,
    #         "vector": True,
    #     }
    # )

    all_tests.append(
        {
            "dataset": "lipo",
            "splitter": "scaffold",
            "task": "regression",
            "k": 5,
            "is_imbalanced": False,
            "n": 1,
            "compressor": GzipCompressor(),
        }
    )

    # all_tests.append(
    #     {
    #         "dataset": "lipo",
    #         "splitter": "scaffold",
    #         "task": "regression_vec",
    #         "k": 5,
    #         "is_imbalanced": False,
    #         "n": 1,
    #         "bins": 256,
    #         "vector": True,
    #     }
    # )

    # all_tests.append(
    #     {
    #         "dataset": "clearance",
    #         "splitter": "scaffold",
    #         "task": "regression",
    #         "k": 5,
    #         "is_imbalanced": False,
    #         "n": 1,
    #     }
    # )

    # all_tests.append(
    #     {
    #         "dataset": "clearance",
    #         "splitter": "scaffold",
    #         "task": "regression_vec",
    #         "k": 5,
    #         "is_imbalanced": False,
    #         "n": 1,
    #         "bins": 256,
    #         "vector": True,
    #     }
    # )

    # all_tests.append(
    #     {
    #         "dataset": "hiv",
    #         "splitter": "scaffold",
    #         "task": "regression",
    #         "k": 5,
    #         "is_imbalanced": False,
    #         "n": 1,
    #     }
    # )

    # all_tests.append(
    #     {
    #         "dataset": "hiv",
    #         "splitter": "scaffold",
    #         "task": "regression_vec",
    #         "k": 5,
    #         "is_imbalanced": False,
    #         "n": 1,
    #         "bins": 256,
    #         "vector": True,
    #     }
    # )

    return all_tests
