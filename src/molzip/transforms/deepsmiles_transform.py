from typing import Iterable, List
from molzip.transforms.base_transform import BaseTransform
import deepsmiles


class DeepsmilesTransform(BaseTransform):
    def __init__(
        self, rings: bool = True, branches: bool = True, **kwargs
    ) -> "DeepsmilesTransform":
        self.converter = deepsmiles.Converter(rings, branches)

        super().__init__("DeepsmilesTransform")

    def transform(self, smiles: Iterable[str]) -> List[str]:
        result = []

        for s in smiles:
            s, rest = self.split(s)
            try:
                result.append(self.converter.encode(s) + rest)
            except:
                result.append(s + rest)

        return result
