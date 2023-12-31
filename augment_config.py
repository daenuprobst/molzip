from molzip.transforms import DeepsmilesTransform, SelfiesTransform, AugmentTransform


def get_all_tests():
    all_tests = []

    for i in range(20):
        all_tests.append(
            {
                "dataset": "bace_classification",
                "splitter": "scaffold",
                "task": "classification",
                "k": 25,
                "augment": 0,
                "preprocess": False,
                "sub_sample": 0.0,
                "is_imbalanced": True,
                "n": 1,
                "transforms": [AugmentTransform(i)],
                "result_props": {"augment_n": i},
            }
        )
        all_tests.append(
            {
                "dataset": "bace_classification",
                "splitter": "scaffold",
                "task": "classification_vec",
                "k": 25,
                "augment": 0,
                "preprocess": False,
                "type_preproc": "default",
                "sub_sample": 0.0,
                "bins": 256,
                "vector": True,
                "augment": 0,
                "preprocess": False,
                "sub_sample": 0.0,
                "is_imbalanced": True,
                "n": 1,
                "transforms": [AugmentTransform(i)],
                "result_props": {"augment_n": i},
            }
        )
        all_tests.append(
            {
                "dataset": "bbbp",
                "splitter": "scaffold",
                "task": "classification",
                "k": 25,
                "augment": 0,
                "preprocess": False,
                "sub_sample": 0.0,
                "is_imbalanced": True,
                "n": 1,
                "transforms": [AugmentTransform(i)],
                "result_props": {"augment_n": i},
            }
        )
        all_tests.append(
            {
                "dataset": "bbbp",
                "splitter": "scaffold",
                "task": "classification_vec",
                "k": 25,
                "augment": 0,
                "preprocess": False,
                "type_preproc": "default",
                "sub_sample": 0.0,
                "bins": 256,
                "vector": True,
                "augment": 0,
                "preprocess": False,
                "sub_sample": 0.0,
                "is_imbalanced": True,
                "n": 1,
                "transforms": [AugmentTransform(i)],
                "result_props": {"augment_n": i},
            }
        )
        all_tests.append(
            {
                "dataset": "delaney",
                "splitter": "scaffold",
                "task": "regression",
                "k": 5,
                "augment": 0,
                "preprocess": False,
                "sub_sample": 0.0,
                "is_imbalanced": False,
                "n": 1,
                "transforms": [AugmentTransform(i)],
                "result_props": {"augment_n": i},
            }
        )
        all_tests.append(
            {
                "dataset": "delaney",
                "splitter": "scaffold",
                "task": "regression_vec",
                "k": 5,
                "augment": 0,
                "preprocess": False,
                "sub_sample": 0.0,
                "is_imbalanced": False,
                "n": 1,
                "bins": 256,
                "vector": True,
                "transforms": [AugmentTransform(i)],
                "result_props": {"augment_n": i},
            }
        )
        all_tests.append(
            {
                "dataset": "bace_regression",
                "splitter": "scaffold",
                "task": "regression",
                "k": 5,
                "augment": 0,
                "preprocess": False,
                "sub_sample": 0.0,
                "is_imbalanced": False,
                "n": 1,
                "transforms": [AugmentTransform(i)],
                "result_props": {"augment_n": i},
            }
        )
        all_tests.append(
            {
                "dataset": "bace_regression",
                "splitter": "scaffold",
                "task": "regression_vec",
                "k": 5,
                "augment": 0,
                "preprocess": False,
                "sub_sample": 0.0,
                "is_imbalanced": False,
                "n": 1,
                "bins": 256,
                "vector": True,
                "transforms": [AugmentTransform(i)],
                "result_props": {"augment_n": i},
            }
        )

    return all_tests
