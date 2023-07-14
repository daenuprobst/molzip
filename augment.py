# Adapted from https://github.com/undeadpixel/reinvent-randomized
import random
import rdkit.Chem as rkc


def randomize_smiles(mol, random_type="restricted"):
    """
    Returns a random SMILES given a SMILES of a molecule.
    :param mol: A Mol object
    :param random_type: The type (unrestricted, restricted) of randomization performed.
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    """
    if not mol:
        return None

    if random_type == "unrestricted":
        return rkc.MolToSmiles(
            mol, canonical=False, doRandom=True, isomericSmiles=False
        )
    if random_type == "restricted":
        new_atom_order = list(range(mol.GetNumAtoms()))
        random.shuffle(new_atom_order)
        random_mol = rkc.RenumberAtoms(mol, newOrder=new_atom_order)
        return rkc.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)

    raise ValueError("Type '{}' is not valid".format(random_type))
