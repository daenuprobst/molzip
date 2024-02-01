import snappy
from molzip.compressors.compressor import Compressor


class SnappyCompressor(Compressor):
    def __init__(self) -> None:
        super().__init__()

    def compress(self, text: str):
        return snappy.compress(text.encode())
