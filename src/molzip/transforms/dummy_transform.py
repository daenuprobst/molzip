from typing import Iterable, List
from molzip.transforms.base_transform import BaseTransform
import selfies as sf


class DummyTransform(BaseTransform):
    def __init__(self, **kwargs) -> "DummyTransform":
        super().__init__("DummyTransform")

    def transform(self, smiles: Iterable[str]) -> List[str]:
        return smiles
