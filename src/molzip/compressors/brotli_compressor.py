import brotli
from molzip.compressors.compressor import Compressor


class BrotliCompressor(Compressor):
    def __init__(self) -> None:
        super().__init__()

    def compress(self, text: str):
        return brotli.compress(text.encode())
