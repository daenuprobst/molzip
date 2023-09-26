from typing import Iterable, List
from abc import ABC, abstractmethod


class BaseTransform(ABC):
    @abstractmethod
    def __init__(self) -> "BaseTransform":
        ...

    def __call__(self, smiles: Iterable[str]) -> List[str]:
        return self.transform(smiles)

    @abstractmethod
    def transform(self, smiles: Iterable[str]) -> List[str]:
        raise NotImplementedError()
