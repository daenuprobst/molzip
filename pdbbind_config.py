from molzip.transforms import DeepsmilesTransform, SelfiesTransform, AugmentTransform


def get_all_tests():
    all_tests = []

    all_tests.append(
        {
            "dataset": "pdbbind",
            "splitter": "scaffold",
            "task": "regression",
            "vector": True,
            "bins": 256,
            "k": 3,
            "is_imbalanced": False,
            "properties": ["ligand_smiles", "pocket_seq"],
            "transforms": [AugmentTransform(1)],
            "n": 5,
        }
    )

    return all_tests
