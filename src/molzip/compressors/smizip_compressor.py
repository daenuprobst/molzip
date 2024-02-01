import json
from typing import List, Optional
from smizip import SmiZip
from molzip.compressors.compressor import Compressor


class SmiZipCompressor(Compressor):
    def __init__(self, json_file: str) -> None:
        super().__init__()
        ngrams = []

        with open(json_file) as inp:
            ngrams = json.load(inp)["ngrams"]

        self.compressor = SmiZip(ngrams)

    def compress(self, text: str):
        return self.compressor.zip(text)
