import lzma
from molzip.compressors.compressor import Compressor


class LZMACompressor(Compressor):
    def __init__(self) -> None:
        super().__init__()

    def compress(self, text: str):
        return lzma.compress(text.encode())
