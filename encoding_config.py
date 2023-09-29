from molzip.transforms import DeepsmilesTransform, SelfiesTransform, AugmentTransform


def get_all_tests():
    all_tests = []

    for transforms in [
        [],
        [DeepsmilesTransform(rings=True, branches=True)],
        [SelfiesTransform()],
    ]:
        all_tests.append(
            {
                "dataset": "delaney",
                "splitter": "scaffold",
                "task": "regression",
                "k": 5,
                "is_imbalanced": False,
                "n": 1,
                "transforms": transforms,
                "result_props": {
                    "transform": transforms[0].name if len(transforms) > 0 else "None"
                },
            }
        )

        all_tests.append(
            {
                "dataset": "bace_regression",
                "splitter": "scaffold",
                "task": "regression",
                "k": 5,
                "is_imbalanced": False,
                "n": 1,
                "transforms": transforms,
                "result_props": {
                    "transform": transforms[0].name if len(transforms) > 0 else "None"
                },
            }
        )

        all_tests.append(
            {
                "dataset": "lipo",
                "splitter": "scaffold",
                "task": "regression",
                "k": 5,
                "is_imbalanced": False,
                "n": 1,
                "transforms": transforms,
                "result_props": {
                    "transform": transforms[0].name if len(transforms) > 0 else "None"
                },
            }
        )

        all_tests.append(
            {
                "dataset": "clearance",
                "splitter": "scaffold",
                "task": "regression",
                "k": 5,
                "is_imbalanced": False,
                "n": 1,
                "transforms": transforms,
                "result_props": {
                    "transform": transforms[0].name if len(transforms) > 0 else "None"
                },
            }
        )

        all_tests.append(
            {
                "dataset": "bbbp",
                "splitter": "scaffold",
                "task": "classification",
                "k": 25,
                "is_imbalanced": True,
                "n": 1,
                "transforms": transforms,
                "result_props": {
                    "transform": transforms[0].name if len(transforms) > 0 else "None"
                },
            }
        )

        all_tests.append(
            {
                "dataset": "clintox",
                "splitter": "scaffold",
                "task": "classification",
                "task_name": "CT_TOX",
                "k": 25,
                "is_imbalanced": True,
                "n": 1,
                "transforms": transforms,
                "result_props": {
                    "transform": transforms[0].name if len(transforms) > 0 else "None"
                },
            }
        )

        all_tests.append(
            {
                "dataset": "tox21",
                "splitter": "scaffold",
                "task": "classification",
                "task_name": "SR-p53",
                "k": 25,
                "is_imbalanced": False,
                "n": 1,
                "transforms": transforms,
                "result_props": {
                    "transform": transforms[0].name if len(transforms) > 0 else "None"
                },
            }
        )

        all_tests.append(
            {
                "dataset": "hiv",
                "splitter": "scaffold",
                "task": "classification",
                "k": 25,
                "is_imbalanced": True,
                "n": 1,
                "transforms": transforms,
                "result_props": {
                    "transform": transforms[0].name if len(transforms) > 0 else "None"
                },
            }
        )

        all_tests.append(
            {
                "dataset": "bace_classification",
                "splitter": "scaffold",
                "task": "classification",
                "k": 25,
                "is_imbalanced": True,
                "n": 1,
                "transforms": transforms,
                "result_props": {
                    "transform": transforms[0].name if len(transforms) > 0 else "None"
                },
            }
        )

    return all_tests
