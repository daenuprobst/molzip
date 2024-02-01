from molzip.transforms import DeepsmilesTransform, SelfiesTransform, AugmentTransform
from molzip.compressors import BrotliCompressor, LZ4Compressor, SnappyCompressor


def get_all_tests():
    all_tests = []

    all_tests.append(
        {
            "dataset": "pdbbind",
            "splitter": "scaffold",
            "task": "regression",
            "vector": False,
            "bins": 256,
            "k": 5,
            "is_imbalanced": False,
            "properties": ["ligand_smiles", "pocket_seq"],
            "transforms": [AugmentTransform(3)],
            "n": 5,
        }
    )

    return all_tests
