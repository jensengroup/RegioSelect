## IMPORT MODULES
from morfeus import BuriedVolume, read_xyz
from rdkit import Chem
## END

def get_sterics(smiles, path):
  '''
  Return the buried volume for each non-H atom
  
  :param smiles: Smiles for input mol
  :param path: .xyz file path containing 3d geometry
  '''
  mol = Chem.MolFromSmiles(smiles)
  patt = Chem.MolFromSmarts('[*]') #all non-H centers

  matches = mol.GetSubstructMatches(patt)
  matches = [i[0] for i in matches]

  elements, coordinates = read_xyz(path)

  bv_values = []
  bv_ids = []

  for i, element in enumerate(elements):
    if i in matches:
      bv = BuriedVolume(elements, coordinates, i+1) # BuriedVolume is 1-indexed
      bv_values.append(bv.fraction_buried_volume*100)
      bv_ids.append(i)
  
  return bv_values, bv_ids