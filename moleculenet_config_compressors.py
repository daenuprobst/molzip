import gzip
from molzip.transforms import (
    AugmentTransform,
    DeepsmilesTransform,
    SelfiesTransform,
    DummyTransform,
)
from molzip.compressors import (
    GzipCompressor,
    LZ4Compressor,
    SnappyCompressor,
)

compressors = [
    (GzipCompressor(), "Gzip"),
    (LZ4Compressor(), "LZ4"),
    (SnappyCompressor(), "Snappy"),
]

transforms = [
    ([DummyTransform()], "None"),
    ([DeepsmilesTransform()], "DeepSmiles"),
    ([SelfiesTransform()], "Selfies"),
    ([AugmentTransform(1)], "Aug 1"),
    ([AugmentTransform(3)], "Aug 3"),
    ([AugmentTransform(5)], "Aug 5"),
    ([AugmentTransform(10)], "Aug 10"),
]


def get_all_tests():
    all_tests = []

    for transform, transform_name in transforms:
        for compressor, name in compressors:
            all_tests.append(
                {
                    "dataset": "bbbp",
                    "splitter": "scaffold",
                    "task": "classification",
                    "k": 5,
                    "is_imbalanced": True,
                    "n": 1,
                    "transforms": transform,
                    "compressor": compressor,
                    "result_props": {"compressor": name, "transform": transform_name},
                }
            )

            all_tests.append(
                {
                    "dataset": "tox21",
                    "splitter": "scaffold",
                    "task": "classification",
                    "k": 5,
                    "is_imbalanced": True,
                    "n": 1,
                    "transforms": transform,
                    "compressor": compressor,
                    "result_props": {"compressor": name, "transform": transform_name},
                }
            )

            all_tests.append(
                {
                    "dataset": "sider",
                    "splitter": "scaffold",
                    "task": "classification",
                    "k": 5,
                    "is_imbalanced": True,
                    "n": 1,
                    "transforms": transform,
                    "compressor": compressor,
                    "result_props": {"compressor": name, "transform": transform_name},
                }
            )

            all_tests.append(
                {
                    "dataset": "clintox",
                    "splitter": "scaffold",
                    "task": "classification",
                    "k": 5,
                    "is_imbalanced": True,
                    "n": 1,
                    "transforms": transform,
                    "compressor": compressor,
                    "result_props": {"compressor": name, "transform": transform_name},
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
                    "transforms": transform,
                    "compressor": compressor,
                    "result_props": {"compressor": name, "transform": transform_name},
                }
            )

            all_tests.append(
                {
                    "dataset": "esol",
                    "splitter": "scaffold",
                    "task": "regression",
                    "k": 5,
                    "is_imbalanced": False,
                    "n": 1,
                    "transforms": transform,
                    "compressor": compressor,
                    "result_props": {"compressor": name, "transform": transform_name},
                }
            )

            all_tests.append(
                {
                    "dataset": "freesolv",
                    "splitter": "scaffold",
                    "task": "regression",
                    "k": 5,
                    "is_imbalanced": False,
                    "n": 1,
                    "transforms": transform,
                    "compressor": compressor,
                    "result_props": {"compressor": name, "transform": transform_name},
                }
            )

            all_tests.append(
                {
                    "dataset": "hiv",
                    "splitter": "scaffold",
                    "task": "classification",
                    "k": 5,
                    "is_imbalanced": True,
                    "n": 1,
                    "transforms": transform,
                    "compressor": compressor,
                    "result_props": {"compressor": name, "transform": transform_name},
                }
            )

            all_tests.append(
                {
                    "dataset": "qm8",
                    "splitter": "scaffold",
                    "task": "regression",
                    "k": 5,
                    "is_imbalanced": False,
                    "n": 1,
                    "transforms": transform,
                    "compressor": compressor,
                    "result_props": {"compressor": name, "transform": transform_name},
                }
            )

    return all_tests
