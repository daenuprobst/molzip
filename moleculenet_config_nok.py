import gzip
from molzip.transforms import AugmentTransform
from molzip.compressors import (
    GzipCompressor,
    LZ4Compressor,
    BrotliCompressor,
    SnappyCompressor,
)

compressors = [
    (GzipCompressor(), "Gzip"),
    # (LZ4Compressor(), "LZ4"),
    # (BrotliCompressor(), "Brotli"),
    # (SnappyCompressor(), "Snappy"),
]


def get_all_tests():
    all_tests = []

    for compressor, name in compressors:
        all_tests.append(
            {
                "dataset": "bbbp",
                "splitter": "scaffold",
                "task": "classification",
                "is_imbalanced": True,
                "n": 1,
                # "transforms": [AugmentTransform(3)],
                "compressor": compressor,
                "result_props": {"compressor": name},
            }
        )

        all_tests.append(
            {
                "dataset": "tox21",
                "splitter": "scaffold",
                "task": "classification",
                "is_imbalanced": True,
                "n": 1,
                # "transforms": [AugmentTransform(3)],
                "compressor": compressor,
                "result_props": {"compressor": name},
            }
        )

        all_tests.append(
            {
                "dataset": "sider",
                "splitter": "scaffold",
                "task": "classification",
                "is_imbalanced": True,
                "n": 1,
                # "transforms": [AugmentTransform(3)],
                "compressor": compressor,
                "result_props": {"compressor": name},
            }
        )

        all_tests.append(
            {
                "dataset": "clintox",
                "splitter": "scaffold",
                "task": "classification",
                "is_imbalanced": True,
                "n": 1,
                # "transforms": [AugmentTransform(3)],
                "compressor": compressor,
                "result_props": {"compressor": name},
            }
        )

        all_tests.append(
            {
                "dataset": "lipo",
                "splitter": "scaffold",
                "task": "regression",
                "is_imbalanced": False,
                "n": 1,
                # "transforms": [AugmentTransform(3)],
                "compressor": compressor,
                "result_props": {"compressor": name},
            }
        )

        all_tests.append(
            {
                "dataset": "esol",
                "splitter": "scaffold",
                "task": "regression",
                "is_imbalanced": False,
                "n": 1,
                # "transforms": [AugmentTransform(3)],
                "compressor": compressor,
                "result_props": {"compressor": name},
            }
        )

        all_tests.append(
            {
                "dataset": "freesolv",
                "splitter": "scaffold",
                "task": "regression",
                "is_imbalanced": False,
                "n": 1,
                # "transforms": [AugmentTransform(3)],
                "compressor": compressor,
                "result_props": {"compressor": name},
            }
        )

        all_tests.append(
            {
                "dataset": "hiv",
                "splitter": "scaffold",
                "task": "classification",
                "is_imbalanced": True,
                "n": 5,
                "transforms": [AugmentTransform(3)],
                "compressor": compressor,
                "result_props": {"compressor": name},
            }
        )

        all_tests.append(
            {
                "dataset": "qm8",
                "splitter": "scaffold",
                "task": "regression",
                "is_imbalanced": False,
                "n": 5,
                "transforms": [AugmentTransform(3)],
                "compressor": compressor,
                "result_props": {"compressor": name},
            }
        )

    return all_tests
