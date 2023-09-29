from molzip.transforms import DeepsmilesTransform, SelfiesTransform, AugmentTransform


def get_all_tests():
    all_tests = []

    all_tests.append(
        {
            "dataset": "adme-HLM",
            "splitter": "scaffold",
            "task": "regression_vec",
            "bins": 256,
            "vector": True,
            "k": 5,
            "is_imbalanced": False,
            "n": 1,
            # "transforms": [AugmentTransform(1)],
            # "result_props": {"transform": "None"},
        }
    )

    return all_tests
