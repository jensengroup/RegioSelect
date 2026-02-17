### IMPORT MODULES ###
import os
import sys
import hashlib
import argparse
import numpy as np
import pandas as pd

from rdkit import Chem
from alfabet import model as bde_model
### END ###


def parse_args():
    """
    Argument parser so this can be run from the command line
    """
    parser = argparse.ArgumentParser(description='Run ALFABET predictions from the command line')
    parser.add_argument('-s', '--smi', default='c1cnc2ccoc2c1',
                        help='SMILES input for ALFABET predictions')
    parser.add_argument('-d', '--dir', default='./',
                        help='Base directory')
    return parser.parse_args()


def get_atom_index(smiles, bond_index):
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_molHs = Chem.AddHs(rdkit_mol)
    bond = rdkit_molHs.GetBondWithIdx(bond_index)
    if bond.GetBeginAtom().GetAtomicNum() == 1: # "1" is the atomic number and corresponds to hydrogen
        atom_site = bond.GetEndAtomIdx()
    elif bond.GetEndAtom().GetAtomicNum() == 1: # "1" is the atomic number and corresponds to hydrogen
        atom_site = bond.GetBeginAtomIdx()
    else:
        print(f'WARNING! Atom index cannot be found!')
    return atom_site


def run_alfabet(smiles: str, name: str, base_dir: str):

    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True) # canonicalize input smiles

    df_bde = bde_model.predict([smiles], drop_duplicates=False).sort_values(by='bde_pred')
    df_bde = df_bde[df_bde.bond_type == 'C-H'] # only C-H
    df_bde['Atom ID'] = df_bde.apply(lambda row: get_atom_index(smiles, row['bond_index']), axis=1) if df_bde.shape[0] else []
    df_bde = df_bde.drop_duplicates(subset=['Atom ID'])
    # df_bde = df_bde[['Atom ID', 'bde_pred']].rename(columns={'bde_pred': 'BDE Value [kcal/mol]'})
    df_bde = df_bde.rename(columns={'bde_pred': 'BDE Value [kcal/mol]'})
    df_bde = df_bde.rename(columns={'bdfe_pred': 'BDFE Value [kcal/mol]'})
    df_bde['Reactant'] = df_bde['Atom ID'].apply(lambda site: f'<a href="#" class="show-structure-link" data-atom-id="{site}" sdf_path="{name}/{name}.sdf">Show</a>')
    
    # Save the tabular data to file
    df_bde.to_pickle(os.path.join(base_dir, f'data/desc_calcs/{name}/df_bde_{name}.pkl'))
    return


if __name__ == "__main__":

    args = parse_args() # Obtain CLI
    
    smiles = args.smi
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True) # canonicalize input smiles
    name = hashlib.md5(smiles.encode()).hexdigest() # SMILES MUST BE CANONICALIZED
    base_dir = args.dir
    df_bde = run_alfabet(smiles, name, base_dir)
    print(df_bde[['Atom ID', 'BDE Value [kcal/mol]', 'BDFE Value [kcal/mol]', 'Reactant']])