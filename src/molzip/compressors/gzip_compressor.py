import gzip
from molzip.compressors.compressor import Compressor


class GzipCompressor(Compressor):
    def __init__(self) -> None:
        super().__init__()

    def compress(self, text: str):
        return gzip.compress(text.encode())
