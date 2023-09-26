from typing import Iterable, List
from molzip.transforms.base_transform import BaseTransform
import deepsmiles


class DeepsmilesTransform(BaseTransform):
    def __init__(self, **kwargs) -> "DeepsmilesTransform":
        self.converter = deepsmiles.Converter(**kwargs)

        super().__init__()

    def transform(self, smiles: Iterable[str]) -> List[str]:
        return [self.converter.encode(s) for s in smiles]
