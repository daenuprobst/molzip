from typing import List, Dict, Any, Tuple
import deepchem.molnet as mn
import selfies as sf
import deepsmiles as ds
from mhfp.encoder import MHFPEncoder
from smiles_tokenizer import tokenize
from rdkit.Chem.AllChem import MolFromSmiles, MolToSmiles, MolToInchi

def preprocess(smiles: str,
               preproc: bool = False,
               preprocess_task: str = "smiles"
) -> str:
    """Preprocess SMILES string
    Parameters
    ----------
    smiles: str
        SMILES string to preprocess
    preproc: bool
        Whether to preprocess
    preprocess_task: str
        Preprocessing task to perform
        valid preprocessing tasks:
            - smiles
            - selfies
            - deepsmiles TODO: Fix deepsmiles
            - secfp
            TODO: Add more preprocessing tasks

    Returns
    -------
    str
        Preprocessed SMILES string
    """
    if not preproc:
        return smiles
    if preprocess_task == "smiles":
        smiles = MolToSmiles(
            MolFromSmiles(smiles),
            kekuleSmiles=True,
            allBondsExplicit=True,
            allHsExplicit=True,
        )
        return " ".join(tokenize(smiles))

    elif preprocess_task == "selfies":
        return sf.encoder(smiles)

    elif preprocess_task == "deepsmiles":
        # TODO: Fix deepsmiles
        # return ds.encode(smiles)
        raise NotImplementedError(f"Preprocessing task {preprocess_task} not implemented.")

    elif preprocess_task == "secfp":
        return to_secfp(smiles)

    else:
        raise NotImplementedError(f"Preprocessing task {preprocess_task} not implemented.")


def to_secfp(
        smiles: str,
        radius: int = 3,
        rings: bool = True,
        kekulize: bool = True,
        min_radius: int = 1,
) -> str:
    return " ".join(
        [
            str(s)
            for s in MHFPEncoder.shingling_from_mol(
            MolFromSmiles(smiles), radius, rings, kekulize, min_radius
        )
        ]
    )
