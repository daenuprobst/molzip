all_tests = [
    # {
    #      "dataset": "sampl",
    #      "splitter": "random",
    #      "task": "regression",
    #      "k": 25,
    #      "augment": 0,
    #      "preprocess": False,
    #      "sub_sample": 0.0,
    #      "is_imbalanced": False,
    #      "n": 4,
    #      "vector": False
    # },
    # {
    #      "dataset": "sampl",
    #      "splitter": "random",
    #      "task": "regression_vec",
    #      "k": 25,
    #      "augment": 0,
    #      "preprocess": False,
    #      "type_preproc": "default",
    #      "sub_sample": 0.0,
    #      "is_imbalanced": False,
    #      "n": 4,
    #      "bins": 70,
    #      "vector": True
    # },
    #     {
    #         "dataset": "delaney",
    #         "splitter": "scaffold",
    #         "task": "regression",
    #         "k": 5,
    #         "augment": 0,
    #         "preprocess": False,
    #         "sub_sample": 0.0,
    #         "is_imbalanced": False,
    #         "n": 1,
    #     },
    #     {
    #         "dataset": "delaney",
    #         "splitter": "scaffold",
    #         "task": "regression_vec",
    #         "k": 5,
    #         "augment": 0,
    #         "preprocess": False,
    #         "sub_sample": 0.0,
    #         "is_imbalanced": False,
    #         "n": 1,
    #         "bins": 256,
    #         "vector": True,
    #     },
    {
        "dataset": "bace_regression",
        "splitter": "scaffold",
        "task": "regression",
        "k": 5,
        "augment": 0,
        "preprocess": False,
        "sub_sample": 0.0,
        "is_imbalanced": True,
        "n": 1,
    },
    #     {
    #         "dataset": "bace_regression",
    #         "splitter": "scaffold",
    #         "task": "regression_vec",
    #         "k": 5,
    #         "augment": 0,
    #         "preprocess": False,
    #         "sub_sample": 0.0,
    #         "is_imbalanced": True,
    #         "n": 1,
    #         "bins": 256,
    #         "vector": True,
    #     },
    #     {
    #         "dataset": "lipo",
    #         "splitter": "scaffold",
    #         "task": "regression",
    #         "k": 5,
    #         "augment": 0,
    #         "preprocess": False,
    #         "sub_sample": 0.0,
    #         "is_imbalanced": False,
    #         "n": 1,
    #     },
    #     {
    #         "dataset": "lipo",
    #         "splitter": "scaffold",
    #         "task": "regression_vec",
    #         "k": 5,
    #         "augment": 0,
    #         "preprocess": False,
    #         "sub_sample": 0.0,
    #         "is_imbalanced": False,
    #         "n": 1,
    #         "bins": 128,
    #         "vector": True,
    #     },
    #     {
    #         "dataset": "clearance",
    #         "splitter": "scaffold",
    #         "task": "regression",
    #         "k": 5,
    #         "augment": 0,
    #         "preprocess": False,
    #         "sub_sample": 0.0,
    #         "is_imbalanced": False,
    #         "n": 1,
    #     },
    #     {
    #         "dataset": "clearance",
    #         "splitter": "scaffold",
    #         "task": "regression_vec",
    #         "k": 5,
    #         "augment": 0,
    #         "preprocess": False,
    #         "sub_sample": 0.0,
    #         "is_imbalanced": False,
    #         "n": 1,
    #         "bins": 128,
    #         "vector": True,
    #     },
    # {
    #      "dataset": "lipo",
    #      "splitter": "random",
    #      "task": "regression",
    #      "k": 25,
    #      "augment": 0,
    #      "preprocess": False,
    #      "sub_sample": 0.0,
    #      "is_imbalanced": False,
    #      "n": 4,
    # },
    # {
    #      "dataset": "lipo",
    #      "splitter": "random",
    #      "task": "regression_vec",
    #      "k": 25,
    #      "augment": 0,
    #      "preprocess": False,
    #      "sub_sample": 0.0,
    #      "is_imbalanced": False,
    #      "n": 4,
    #      "bins": 70,
    #      "vector": True
    # },
    # {
    #      "dataset": "sider",
    #      "splitter": "scaffold",
    #      "task": "classification",
    #      "k": 5,
    #      "augment": 0,
    #      "preprocess": False,
    #      "sub_sample": 0.0,
    #      "is_imbalanced": True,
    #      "n": 1,
    # },
    # {
    #      "dataset": "sider",
    #      "splitter": "random",
    #      "task": "classification",
    #      "k": 5,
    #      "augment": 0,
    #      "preprocess": False,
    #      "sub_sample": 0.0,
    #      "is_imbalanced": True,
    #      "n": 4,
    # },
    #     {
    #         "dataset": "bbbp",
    #         "splitter": "scaffold",
    #         "task": "classification_vec",
    #         "k": 25,
    #         "augment": 0,
    #         "preprocess": False,
    #         "type_preproc": "default",
    #         "sub_sample": 0.0,
    #         "bins": 256,
    #         "vector": True,
    #         "augment": 0,
    #         "preprocess": False,
    #         "sub_sample": 0.0,
    #         "is_imbalanced": True,
    #         "n": 4,
    #     },
    #     {
    #         "dataset": "bbbp",
    #         "splitter": "scaffold",
    #         "task": "classification",
    #         "k": 25,
    #         "augment": 0,
    #         "preprocess": False,
    #         "sub_sample": 0.0,
    #         "is_imbalanced": True,
    #         "n": 4,
    #     },
    #    {
    #        "dataset": "bace_classification",
    #        "splitter": "scaffold",
    #        "task": "classification",
    #        "k": 25,
    #        "augment": 0,
    #        "preprocess": False,
    #        "sub_sample": 0.0,
    #        "is_imbalanced": True,
    #        "n": 1,
    #    },
    #    {
    #        "dataset": "bace_classification",
    #        "splitter": "scaffold",
    #        "task": "classification_vec",
    #        "k": 25,
    #        "augment": 0,
    #        "preprocess": False,
    #        "type_preproc": "default",
    #        "sub_sample": 0.0,
    #        "bins": 256,
    #        "vector": True,
    #        "augment": 0,
    #        "preprocess": False,
    #        "sub_sample": 0.0,
    #        "is_imbalanced": True,
    #        "n": 1,
    #    },
    #     {
    #         "dataset": "qm7",
    #         "splitter": "scaffold",
    #         "task": "regression",
    #         "k": 5,
    #         "augment": 0,
    #         "preprocess": False,
    #         "sub_sample": 0.0,
    #         "is_imbalanced": False,
    #         "n": 1,
    #     },
    #     {
    #         "dataset": "qm7",
    #         "splitter": "scaffold",
    #         "task": "regression_vec",
    #         "k": 25,
    #         "augment": 0,
    #         "preprocess": False,
    #         "type_preproc": "default",
    #         "sub_sample": 0.0,
    #         "bins": 256,
    #         "vector": True,
    #         "augment": 0,
    #         "preprocess": False,
    #         "sub_sample": 0.0,
    #         "is_imbalanced": False,
    #         "n": 1,
    #     },
    #     {
    #         "dataset": "bace_classification",
    #         "splitter": "scaffold",
    #         "task": "classification",
    #         "k": 5,
    #         "augment": 0,
    #         "preprocess": False,
    #         "sub_sample": 0.0,
    #         "is_imbalanced": True,
    #         "n": 4,
    #     },
    # {
    #      "dataset": "bace_classification",
    #      "splitter": "random",
    #      "task": "classification",
    #      "k": 5,
    #      "augment": 0,
    #      "preprocess": False,
    #      "sub_sample": 0.0,
    #      "is_imbalanced": True,
    #      "n": 4,
    # },
    #     {
    #         "dataset": "clintox",
    #         "splitter": "scaffold",
    #         "task": "classification_vec",
    #         "task_name": "CT_TOX",
    #         "k": 25,
    #         "augment": 0,
    #         "preprocess": False,
    #         "type_preproc": "default",
    #         "sub_sample": 0.0,
    #         "bins": 256,
    #         "vector": True,
    #         "augment": 0,
    #         "preprocess": False,
    #         "sub_sample": 0.0,
    #         "is_imbalanced": True,
    #         "n": 1,
    #     },
    #     {
    #         "dataset": "clintox",
    #         "splitter": "scaffold",
    #         "task": "classification",
    #         "task_name": "CT_TOX",
    #         "k": 25,
    #         "augment": 0,
    #         "preprocess": False,
    #         "sub_sample": 0.0,
    #         "is_imbalanced": True,
    #         "n": 1,
    #     },
    #     {
    #         "dataset": "tox21",
    #         "splitter": "scaffold",
    #         "task": "classification",
    #         "task_name": "SR-p53",
    #         "k": 25,
    #         "augment": 0,
    #         "preprocess": False,
    #         "sub_sample": 0.0,
    #         "is_imbalanced": False,
    #         "n": 1,
    #     },
    #     {
    #         "dataset": "tox21",
    #         "splitter": "scaffold",
    #         "task": "classification_vec",
    #         "task_name": "SR-p53",
    #         "k": 25,
    #         "augment": 0,
    #         "preprocess": False,
    #         "type_preproc": "default",
    #         "sub_sample": 0.0,
    #         "bins": 256,
    #         "vector": True,
    #         "augment": 0,
    #         "preprocess": False,
    #         "sub_sample": 0.0,
    #         "is_imbalanced": False,
    #         "n": 1,
    #     },
    #     {
    #         "dataset": "hiv",
    #         "splitter": "scaffold",
    #         "task": "classification",
    #         "k": 25,
    #         "augment": 0,
    #         "preprocess": False,
    #         "sub_sample": 0.0,
    #         "is_imbalanced": True,
    #         "n": 1,
    #     },
    #     {
    #         "dataset": "hiv",
    #         "splitter": "scaffold",
    #         "task": "classification_vec",
    #         "k": 25,
    #         "augment": 0,
    #         "preprocess": False,
    #         "type_preproc": "default",
    #         "sub_sample": 0.0,
    #         "bins": 256,
    #         "vector": True,
    #         "augment": 0,
    #         "preprocess": False,
    #         "sub_sample": 0.0,
    #         "is_imbalanced": True,
    #         "n": 1,
    #     },
    # {
    #      "dataset": "muv",
    #      "splitter": "random",
    #      "task": "classification",
    #      "k": 5,
    #      "augment": 0,
    #      "preprocess": False,
    #      "sub_sample": 0.0,
    #      "is_imbalanced": True,
    #      "n": 4,
    # },
    # {
    # "dataset": "schneider",
    # "splitter": "random",
    #      "task": "classification",
    #      "k": 5,
    #      "augment": 0,
    #      "preprocess": True,
    #      "sub_sample": 0.0,
    #      "is_imbalanced": False,
    #      "n": 1,
    # },
]