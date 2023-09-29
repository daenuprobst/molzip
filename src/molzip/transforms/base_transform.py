from typing import Iterable, List, Tuple
from abc import ABC, abstractmethod


class BaseTransform(ABC):
    @abstractmethod
    def __init__(self, name: str) -> "BaseTransform":
        self.name = name

    def __call__(self, smiles: Iterable[str]) -> List[str]:
        return self.transform(smiles)

    def split(self, s: str) -> Tuple[str, str]:
        sp = s.split(" ", 1)
        if len(sp) == 1:
            sp.append("")

        if len(sp[1]) > 0:
            sp[1] = " " + sp[1]

        return tuple(sp)

    @abstractmethod
    def transform(self, smiles: Iterable[str]) -> List[str]:
        raise NotImplementedError()
