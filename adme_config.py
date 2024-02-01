from molzip.transforms import DeepsmilesTransform, SelfiesTransform, AugmentTransform
from molzip.compressors import LZ4Compressor, GzipCompressor


def get_all_tests():
    all_tests = []

    all_tests.append(
        {
            "dataset": "adme-HLM",
            "splitter": "random",
            "task": "regression",
            "k": 5,
            "is_imbalanced": False,
            "n": 1,
            "transforms": [AugmentTransform(3)],
            "compressor": GzipCompressor(),
        }
    )

    all_tests.append(
        {
            "dataset": "adme-hPPB",
            "splitter": "random",
            "task": "regression",
            "k": 5,
            "is_imbalanced": False,
            "n": 1,
            "transforms": [AugmentTransform(3)],
            "compressor": GzipCompressor(),
        }
    )

    all_tests.append(
        {
            "dataset": "adme-MDR1_ER",
            "splitter": "random",
            "task": "regression",
            "k": 5,
            "is_imbalanced": False,
            "n": 1,
            "transforms": [AugmentTransform(3)],
            "compressor": GzipCompressor(),
        }
    )

    all_tests.append(
        {
            "dataset": "adme-RLM",
            "splitter": "random",
            "task": "regression",
            "k": 5,
            "is_imbalanced": False,
            "n": 1,
            "transforms": [AugmentTransform(3)],
            "compressor": GzipCompressor(),
        }
    )

    all_tests.append(
        {
            "dataset": "adme-rPPB",
            "splitter": "random",
            "task": "regression",
            "k": 5,
            "is_imbalanced": False,
            "n": 1,
            "transforms": [AugmentTransform(3)],
            "compressor": GzipCompressor(),
        }
    )

    all_tests.append(
        {
            "dataset": "adme-Sol",
            "splitter": "random",
            "task": "regression",
            "k": 5,
            "is_imbalanced": False,
            "n": 1,
            "transforms": [AugmentTransform(3)],
            "compressor": GzipCompressor(),
        }
    )

    return all_tests
