from abc import ABC, abstractmethod


class Compressor(ABC):
    def __init__(self) -> None:
        ...

    @abstractmethod
    def compress(self, text: str) -> bytes:
        ...
