import copy
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def remove_identical_atoms(rdkit_mol, atom_list):
    idx_list = []
    rank_kept = []
    atom_rank = list(Chem.CanonicalRankAtoms(rdkit_mol, breakTies=False))
    for idx, atom in enumerate(atom_list):
        if atom_rank[atom] not in rank_kept:
            rank_kept.append(atom_rank[atom])
            idx_list.append(idx)
    
    atom_list = np.array(atom_list)[idx_list].tolist()
    
    return atom_list


def find_eas_sites(pmol):

    # Save a copy of the input to avoid changes of the mol due to Chem.Kekulize
    rdkit_mol = copy.deepcopy(pmol)

    # Return list with atoms numbers
    atom_list = []

    # Reaction formats
    # __rxn1__ = AllChem.ReactionFromSmarts('[C;H1:1]=[C,N;H1:2]>>[CH2:1][*H+:2]')
    # __rxn2__ = AllChem.ReactionFromSmarts('[C;H1:1]=[C,N;H0:2]>>[CH2:1][*+;H0:2]')

    __rxn1__ = AllChem.ReactionFromSmarts('[C;R;H1:1]=[C,N;R;H1:2]>>[CH2:1][*H+:2]')
    __rxn2__ = AllChem.ReactionFromSmarts('[C;R;H1:1]=[C,N;R;H0:2]>>[CH2:1][*+;H0:2]')

    # Bromine
    # __rxn1__ = AllChem.ReactionFromSmarts('[C;R;H1:1]=[C,N;R;H1:2]>>[CH:1](Br)[*H+:2]')
    # __rxn2__ = AllChem.ReactionFromSmarts('[C;R;H1:1]=[C,N;R;H0:2]>>[CH:1](Br)[*+;H0:2]')

    Chem.Kekulize(rdkit_mol,clearAromaticFlags=True)

    # target = Chem.MolFromSmarts('[C;H1:1]=[C,N;H1:2]')
    target = Chem.MolFromSmarts('[C;R;H1:1]=[C,N;R;H1:2]')
    atoms = rdkit_mol.GetSubstructMatches(target)

    # Convert tuple of tuple to one-dimensional list
    atoms = [element for tupl in atoms for element in tupl]

    i = 0
    ps = __rxn1__.RunReactants((rdkit_mol,))
    for x in ps:
        i += 1
        atom_list.append(atoms[i-1])
    
    # target = Chem.MolFromSmarts('[C;H1:1]=[C,N;H0:2]')
    target = Chem.MolFromSmarts('[C;R;H1:1]=[C,N;R;H0:2]')
    atoms = rdkit_mol.GetSubstructMatches(target)

    # Convert tuple of tuple to one-dimensional list
    atoms = [element for tupl in atoms for element in tupl]

    isav = i
    ps = __rxn2__.RunReactants((rdkit_mol,))
    for x in ps:
        i += 1
        atom_list.append(atoms[2*(i-isav)-2])
    
    # Keep only one of the elements for identical atoms
    atom_list = remove_identical_atoms(pmol, atom_list)

    return atom_list



if __name__ == "__main__":
    
    import time
    start = time.perf_counter()

    input_mol = Chem.MolFromSmiles('n1cccn1c1ncccn1')

    print(find_eas_sites(input_mol))
    
    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')
