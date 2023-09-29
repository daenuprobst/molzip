from typing import Iterable, List
from molzip.transforms.base_transform import BaseTransform
from rdkit.Chem import MolFromSmiles, MolToSmiles


class AugmentTransform(BaseTransform):
    def __init__(self, n: int = 2, **kwargs) -> "AugmentTransform":
        super().__init__("AugmentTransform")
        self.n = n

    def transform(self, smiles: Iterable[str]) -> List[str]:
        out_smiles = []

        for s in smiles:
            s, rest = self.split(s)

            s_augmented = [s]
            for _ in range(self.n):
                mol = MolFromSmiles(s)
                if mol is not None:
                    s_rand = MolToSmiles(
                        mol,
                        canonical=False,
                        doRandom=True,
                        kekuleSmiles=True,
                        allBondsExplicit=True,
                        allHsExplicit=True,
                    )

                    s_augmented.append(s_rand)
            out_smiles.append(" ".join(s_augmented) + rest)

        return out_smiles
