import lz4.block as lz4compressor
from molzip.compressors.compressor import Compressor


class LZ4Compressor(Compressor):
    def __init__(self) -> None:
        super().__init__()

    def compress(self, text: str):
        return lz4compressor.compress(text.encode())
