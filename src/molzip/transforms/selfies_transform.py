from typing import Iterable, List
from molzip.transforms.base_transform import BaseTransform
import selfies as sf


class SelfiesTransform(BaseTransform):
    def __init__(self, **kwargs) -> "SelfiesTransform":
        super().__init__()

    def transform(self, smiles: Iterable[str]) -> List[str]:
        return [sf.encoder(s) for s in smiles]
