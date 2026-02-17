### IMPORT MODULES ###
import os
import sys
import argparse
import numpy as np
import pandas as pd

from rdkit import Chem
import lightgbm as lgb

from alfabet import model as bde_model

base_dir = os.path.dirname(os.path.realpath(__file__)).replace('/scripts/src', '')

#os.chdir(os.path.join(base_dir, 'scripts/src')) # change path to make the calculation run
sys.path.append(os.path.join(base_dir, 'scripts/src'))
from esnuelML.DescriptorCreator.GraphChargeShell import GraphChargeShell
from regioML.locate_EAS_sites import find_eas_sites
from HAlator.modify_smiles import remove_Hs_halator
from esnuelML.locate_atom_sites import find_nucleophilic_sites, find_electrophilic_sites
from sterics.calc_sterics import get_sterics
### END ###

### LOAD ALL MODELS ###
eas_model = lgb.Booster(model_file=os.path.join(base_dir, 'scripts/src/regioML/LGBM_measured_allData_final_model.txt'))
pka_model = lgb.Booster(model_file=os.path.join(base_dir, 'scripts/src/pKalculator/reg_model_all_data_dart_nshells3.txt'))
ha_model = lgb.Booster(model_file=os.path.join(base_dir, 'scripts/src/HAlator/final_reg_model_all_data_dart_default_nshells3.txt'))
nuc_model = lgb.Booster(model_file=os.path.join(base_dir, 'scripts/src/esnuelML/models/nuc/SMI2GCS_3_cm5_model.txt'))
elec_model = lgb.Booster(model_file=os.path.join(base_dir, 'scripts/src/esnuelML/models/elec/SMI2GCS_3_cm5_model.txt'))
### END ###


def parse_args():
    """
    Argument parser so this can be run from the command line
    """
    parser = argparse.ArgumentParser(description='Run regioselect predictions from the command line')
    parser.add_argument('-s', '--smiles', default='CCOC(=O)c1cc(Cl)n2nc(-c3ccc(Br)cc3F)cc2n1',
                        help='SMILES input for regioselect predictions')
    parser.add_argument('-n', '--name', default='f302f6b8b26306c5e315a67a2346781a', help='The name of the molecule. Only names without "_" are allowed.')
    return parser.parse_args()



def find_identical_atoms(rdkit_mol, atom_list):
    len_list = len(atom_list)
    
    atom_rank = list(Chem.CanonicalRankAtoms(rdkit_mol, breakTies=False))
    for idx, atom in enumerate(rdkit_mol.GetAtoms()):
        if atom.GetIdx() in atom_list[:len_list]:
            sym_atoms = [int(atom_idx) for atom_idx, ranking in enumerate(atom_rank) if ranking == atom_rank[idx] and atom_idx not in atom_list] 
            atom_list.extend(sym_atoms)
    return atom_list


def generate_output_tables(name, rdkit_mol, sites_list, values_list, type_list, vbur_dict, val_name='MAA Value [kJ/mol]'):
    
    sites_list_new = []
    values_list_new = []
    sdfpath_structures_new = []
    type_list_new = []
    vbur_list = []

    for idx, val in enumerate(values_list):
        site = sites_list[idx] # the atomic index of the located site
        identical_sites = find_identical_atoms(rdkit_mol, [site])
        for site in identical_sites:
            sites_list_new.append(f'{site}')
            values_list_new.append(val)
            sdfpath_structures_new.append(f'<a href="#" class="show-structure-link" data-atom-id="{site}" sdf_path="{name}/{name}.sdf">Show</a>')
            type_list_new.append(type_list[idx].replace('_', ' ').capitalize())
            vbur_list.append(vbur_dict['%Vbur'][site]) #Appends the %Vbur of the located site

    dict_table = {'Atom ID': sites_list_new, f'{val_name}': values_list_new, 'Reactant': sdfpath_structures_new, 'Type': type_list_new, f'%Vbur': vbur_list}

    if val_name in ['MCA Value [kJ/mol]', 'MAA Value [kJ/mol]', 'EAS Score [%]']:
        df_table = pd.DataFrame(dict_table).sort_values(by=[f'{val_name}'], ascending=False)
    else:
        df_table = pd.DataFrame(dict_table).sort_values(by=[f'{val_name}'], ascending=True)
    
    return df_table


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


def run_predictions(smiles: str, name: str):

    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True) # canonicalize input smiles

    # Calculate CM5 atomic charges
    desc_generator = GraphChargeShell()
    cm5_list = desc_generator.calc_CM5_charges(smiles, name=name, optimize=False, save_output=True)

    # Save structures in SDF format
    writer = Chem.rdmolfiles.SDWriter(desc_generator.xyz_file_path.replace('.xyz', '.sdf'))
    writer.write(desc_generator.rdkit_mol)
    writer.close()

    # Locate sites of interest
    eas_sites = find_eas_sites(Chem.MolFromSmiles(smiles))
    try:
        _, pka_sites, _, _, _, _, _, _ = remove_Hs_halator(name=name, smiles=smiles, rdkit_mol=None, atomsite=None, gen_all=True, remove_H=True, rxn="rm_proton")
        _, ha_sites, _, _, _, _, _, _  = remove_Hs_halator(name=name, smiles=smiles, rdkit_mol=None, atomsite=None, gen_all=True, remove_H=True, rxn="rm_hydride")
    except:
        pka_sites = []
        ha_sites = []
    nuc_sites, nuc_names, nuc_smirks = find_nucleophilic_sites(Chem.MolFromSmiles(smiles))
    elec_sites, elec_names, elec_smirks = find_electrophilic_sites(Chem.MolFromSmiles(smiles))

    # Concatenate all sites and generate all descriptors
    all_sites = list(set(eas_sites) | set(pka_sites) | set(ha_sites) | set(nuc_sites) | set(elec_sites))
    all_descriptor_vectors, _ = desc_generator.create_descriptor_vector(all_sites, n_shells=3, max_neighbors=4, use_cip_sort=True)

    # Extract the descriptors for the different models
    eas_descriptor_vectors = [all_descriptor_vectors[all_sites.index(i)] for i in eas_sites]
    pka_descriptor_vectors = [all_descriptor_vectors[all_sites.index(i)] for i in pka_sites]
    ha_descriptor_vectors = [all_descriptor_vectors[all_sites.index(i)] for i in ha_sites]
    nuc_descriptor_vectors = [all_descriptor_vectors[all_sites.index(i)] for i in nuc_sites]
    elec_descriptor_vectors = [all_descriptor_vectors[all_sites.index(i)] for i in elec_sites]

    # Run ML predictions
    eas_values = eas_model.predict(eas_descriptor_vectors) if len(eas_descriptor_vectors) else []
    pka_values = pka_model.predict(pka_descriptor_vectors) if len(pka_descriptor_vectors) else []
    ha_values = ha_model.predict(ha_descriptor_vectors) if len(ha_descriptor_vectors) else []
    nuc_values = nuc_model.predict(nuc_descriptor_vectors) if len(nuc_descriptor_vectors) else []
    elec_values = elec_model.predict(elec_descriptor_vectors) if len(elec_descriptor_vectors) else []
    
    # Run sterics (buried volume) calculation
    steric_values, steric_sites = get_sterics(smiles, desc_generator.xyz_file_path)

    # Run ALFABET predictions
    df_bde = bde_model.predict([smiles], drop_duplicates=False).sort_values(by='bde_pred')
    df_bde = df_bde[df_bde.bond_type == 'C-H'] # only C-H
    df_bde['Atom ID'] = df_bde.apply(lambda row: get_atom_index(smiles, row['bond_index']), axis=1) if df_bde.shape[0] else []
    df_bde = df_bde.drop_duplicates(subset=['Atom ID'])
    df_bde = df_bde.rename(columns={'bde_pred': 'BDE Value [kcal/mol]'})
    df_bde = df_bde.rename(columns={'bdfe_pred': 'BDFE Value [kcal/mol]'})
    df_bde['Reactant'] = df_bde['Atom ID'].apply(lambda site: f'<a href="#" class="show-structure-link" data-atom-id="{site}" sdf_path="{name}/{name}.sdf">Show</a>')

    ### Generate Result Tables ###
    df_steric = pd.DataFrame({'Atom ID': steric_sites, '%Vbur': steric_values}) # Calculate sterics first - used in each df
    df_bde = pd.merge(df_bde, df_steric, how='left', on='Atom ID') # Add sterics directly to df_bde
    df_eas = generate_output_tables(name, desc_generator.rdkit_mol, list(eas_sites), eas_values*100, ["-" for _ in range(len(eas_values))], vbur_dict=df_steric.to_dict(), val_name='EAS Score [%]').drop(columns=['Type'])
    df_pka = generate_output_tables(name, desc_generator.rdkit_mol, list(pka_sites), pka_values, ["-" for _ in range(len(pka_values))], vbur_dict=df_steric.to_dict(), val_name='pKa Value').drop(columns=['Type'])
    df_ha = generate_output_tables(name, desc_generator.rdkit_mol, list(ha_sites), ha_values, ["-" for _ in range(len(ha_values))], vbur_dict=df_steric.to_dict(), val_name='HA Value [kcal/mol]').drop(columns=['Type'])
    df_nuc = generate_output_tables(name, desc_generator.rdkit_mol, nuc_sites, nuc_values, nuc_names, vbur_dict=df_steric.to_dict(), val_name='MCA Value [kJ/mol]')
    df_elec = generate_output_tables(name, desc_generator.rdkit_mol, elec_sites, elec_values, elec_names, vbur_dict=df_steric.to_dict(), val_name='MAA Value [kJ/mol]')
    
    df_eas.to_pickle(os.path.join(base_dir, f'data/desc_calcs/{name}/df_eas_{name}.pkl'))
    df_bde.to_pickle(os.path.join(base_dir, f'data/desc_calcs/{name}/df_bde_{name}.pkl'))
    df_pka.to_pickle(os.path.join(base_dir, f'data/desc_calcs/{name}/df_pka_{name}.pkl'))
    df_ha.to_pickle(os.path.join(base_dir, f'data/desc_calcs/{name}/df_ha_{name}.pkl'))
    df_nuc.to_pickle(os.path.join(base_dir, f'data/desc_calcs/{name}/df_nuc_{name}.pkl'))
    df_elec.to_pickle(os.path.join(base_dir, f'data/desc_calcs/{name}/df_elec_{name}.pkl'))
    df_steric.to_pickle(os.path.join(base_dir, f'data/desc_calcs/{name}/df_steric_{name}.pkl'))
    return



if __name__ == "__main__":

    import sys
    import submitit
    
    sys.path.append(base_dir)
    from __init__ import app, db, regioselect_results
    
    app.app_context().push() # connect to sql database
    
    args = parse_args() # Obtain CLI

    ### Slurm settings ###
    executor = submitit.AutoExecutor(folder=os.path.join(base_dir, f'submitit_regioselect/{args.name}'))
    
    executor.update_parameters(
        name="regioselect",
        cpus_per_task=1,
        slurm_mem=3900,
        timeout_min=6000,
        slurm_partition="p1",
        slurm_array_parallelism=1,
    )
    
    ## For use on HPC below:
    #executor.update_parameters(
    #    name="regioselect",
    #    cpus_per_task=1,
    #    slurm_mem=3900,
    #    timeout_min=1000,
    #    slurm_partition="kemi1",
	#    slurm_account="chemistry",
    #    slurm_array_parallelism=1,
    #)
    ### END ###
        
    try:
        # Submit job to slurm
        jobs = []
        with executor.batch():
            job = executor.submit(run_predictions, args.smiles, args.name)
            jobs.append(job)
        
        # Wait for job to finish
        jobs[0].result()

        # Update the status to 'complete' in the database
        regioselect_res = regioselect_results.query.filter_by(hash_code=args.name).first()
        regioselect_res.ml_status = 'complete'
        db.session.commit()

        # Remove submitit folder
        folder_path = os.path.join(base_dir, f'submitit_regioselect/{args.name}')
        if os.path.exists(folder_path):
            for file_remove in os.listdir(folder_path):
                if os.path.isfile(f'{folder_path}/{file_remove}'):
                    os.remove(f'{folder_path}/{file_remove}')
            # checking whether the folder is empty or not
            if len(os.listdir(folder_path)) == 0:
                os.rmdir(folder_path)
            else:
                print("Submitit folder is not empty")
    except:
        # Update the status to 'error' in the database
        regioselect_res = regioselect_results.query.filter_by(hash_code=args.name).first()
        regioselect_res.ml_status = 'error'
        db.session.commit()