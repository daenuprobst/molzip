from typing import Iterable, List
from molzip.transforms.base_transform import BaseTransform
import selfies as sf


class SelfiesTransform(BaseTransform):
    def __init__(self, **kwargs) -> "SelfiesTransform":
        super().__init__("SelfiesTransform")

    def transform(self, smiles: Iterable[str]) -> List[str]:
        result = []

        for s in smiles:
            s, rest = self.split(s)
            try:
                result.append(sf.encoder(s) + rest)
            except:
                result.append(s + rest)

        return result
