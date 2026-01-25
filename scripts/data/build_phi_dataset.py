import os
import h5py
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# --- Configuration ---
INPUT_CSV = "data/phi/phidelta_dataset.csv"
OUTPUT_HDF5 = "data/phi/phidelta_data.h5"
MAX_ATOMS = 100
NUM_CONFS = 10  # This will now actually be used

# --- Solvent Mapping (Name -> SMILES) ---
SOLVENT_TO_SMILES = {
    "MeOH": "CO", "methanol": "CO", "MeOD": "CO", "CD3OD": "CO",
    "EtOH": "CCO", "ethanol": "CCO",
    "1-PrOH": "CCCO", "n-propanol": "CCCO",
    "2-PrOH": "CC(O)C", "i-PrOH": "CC(O)C", "isopropanol": "CC(O)C",
    "1-BuOH": "CCCCO", "n-butanol": "CCCCO",
    "iso-BuOH": "CC(C)CO", "tert-BuOH": "CC(C)(O)C",
    "c-C6H11OH": "OC1CCCCC1", "C6H5CH2OH": "OCc1ccccc1",
    "2-Methoxyethanol": "COCCO", "m-Cresol": "Cc1cccc(O)c1",
    "i-C5H11OH": "CCC(C)CO",
    "hexane": "CCCCCC", "Hexane": "CCCCCC", "n-C6H14": "CCCCCC", "Hexanes": "CCCCCC",
    "cyclohexane": "C1CCCCC1", "c-C6H12": "C1CCCCC1",
    "heptane": "CCCCCCC", "n-C7H16": "CCCCCCC",
    "pentane": "CCCCC", "n-C5H12": "CCCCC",
    "i-octane": "CC(C)CC(C)(C)C",
    "CCl4": "ClC(Cl)(Cl)Cl", "CH2Cl2": "ClCCl", "DCM": "ClCCl",
    "CHCl3": "ClC(Cl)Cl", "CDCl3": "ClC(Cl)Cl",
    "C6H5Cl": "Clc1ccccc1", "C6H5F": "Fc1ccccc1",
    "C6F6": "Fc1c(F)c(F)c(F)c(F)c1F", "ClCF2CCl2F": "FC(F)(Cl)C(F)(Cl)Cl",
    "n-C3H7I": "CCC1", "C6H5Br": "Brc1ccccc1", "C6D5Br": "Brc1ccccc1",
    "toluene": "Cc1ccccc1", "C6H5CH3": "Cc1ccccc1",
    "benzene": "c1ccccc1", "C6H6": "c1ccccc1", "C6D6": "c1ccccc1",
    "THF": "C1CCOC1", "c-C4H8O": "C1CCOC1",
    "diox": "C1COCCO1", "dioxane": "C1COCCO1",
    "(C2H5)2O": "CCOCC", "EtOAc": "CC(=O)OCC",
    "acetone": "CC(=O)C", "CH3COCH3": "CC(=O)C",
    "2-Propanedione": "CC(=O)C=O", "pinacolone": "CC(=O)C(C)(C)C",
    "Propylene carbonate": "CC1COC(=O)O1",
    "CH3CN": "CC#N", "MeCN": "CC#N", "C6H5CN": "N#Cc1ccccc1",
    "DMSO": "CS(=O)C", "DMF": "CN(C)C=O", "HCONH2": "NC=O",
    "C5H5N": "c1ccncc1", "CS2": "S=C=S",
    "water": "O", "H2O": "O", "D2O": "O",
    "H2O (mic)": "O", "D2O (mic)": "O", "H2O (ves)": "O", "D2O (ves)": "O",
    "air": "N", "air (mic)": "N",
}

def generate_conformers(mol, n_confs):
    """Generates multiple conformers and returns the molecule object and list of conf IDs."""
    mol = Chem.AddHs(mol)
    
    # Use ETKDGv3 for better geometry generation
    ps = AllChem.ETKDGv3()
    ps.randomSeed = 42
    
    # Generate multiple conformers
    # This returns a list of conformer IDs (integers)
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=ps)
    
    # If generation failed, try standard method with random coords
    if len(conf_ids) == 0:
        conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, useRandomCoords=True)
        
    if len(conf_ids) > 0:
        try:
            # Optimize all conformers at once
            AllChem.UFFOptimizeMoleculeConfs(mol)
        except Exception as _:
            pass
            
    return mol, list(conf_ids)

def get_single_conformer(mol):
    """Helper specifically for Solvent (we usually only need 1 solvent rep)."""
    mol = Chem.AddHs(mol)
    res = AllChem.EmbedMolecule(mol, randomSeed=42)
    if res == 0:
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception as _:
            pass
    return mol, res

def build_phi_hdf5():
    print(f"Reading {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.strip()
    
    if 'Solvent' not in df.columns and 'solvent' in df.columns:
        df = df.rename(columns={'solvent': 'Solvent'})

    df = df.dropna(subset=['SMILES', 'PhiDelta', 'Solvent'])
    print(f"Found {len(df)} unique solute-solvent pairs.")
    print(f"Generating up to {NUM_CONFS} conformers per solute...")

    valid_data = []
    solvent_cache = {} # Cache solvent RDKit objects to speed up loop

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # --- 1. Process Solvent (Do this first to skip if invalid) ---
            solv_name = str(row['Solvent']).strip()
            if solv_name.lower() in ["solvent", "none", "nan", ""]: 
                continue

            # Check Cache
            if solv_name in solvent_cache:
                solv_mol = solvent_cache[solv_name]
            else:
                solv_smi = SOLVENT_TO_SMILES.get(solv_name)
                # Fallbacks
                if solv_smi is None:
                    if Chem.MolFromSmiles(solv_name) is not None:
                        solv_smi = solv_name
                    elif "/" in solv_name:
                         solv_smi = SOLVENT_TO_SMILES.get(solv_name.split("/")[0].strip())
                
                if solv_smi is None: 
                    continue

                solv_mol_raw = Chem.MolFromSmiles(solv_smi)
                if solv_mol_raw is None: 
                    continue
                
                # Generate 1 conformer for solvent
                solv_mol, s_res = get_single_conformer(solv_mol_raw)
                if s_res != 0: 
                    continue
                
                solvent_cache[solv_name] = solv_mol

            # --- 2. Process Solute (Multiple Conformers) ---
            solute_smi = row['SMILES']
            solute_mol = Chem.MolFromSmiles(solute_smi)
            if solute_mol is None: 
                continue
            
            # Generate List of Conformers
            solute_mol, conf_ids = generate_conformers(solute_mol, NUM_CONFS)
            
            if not conf_ids:
                continue

            # --- 3. Expand Data: Create one entry per conformer ---
            for cid in conf_ids:
                valid_data.append({
                    "solute_mol": solute_mol,
                    "solute_conf_id": cid,      # Store ID to retrieve specific pos later
                    "solvent_mol": solv_mol,    # Same solvent obj for all confs
                    "phi_delta": float(row['PhiDelta']),
                    "lambda_max": float(row['lambda_max']) if 'lambda_max' in row and pd.notna(row['lambda_max']) else np.nan,
                    "mol_id": idx,              # IMPORTANT: Keep same ID for all confs of same molecule
                    "smiles": solute_smi,
                    "solvent_name": solv_name
                })
            
        except Exception as _:
            # print(f"Error on row {idx}: {e}")
            pass

    N = len(valid_data)
    if N == 0: 
        print("Error: No valid molecules generated.")
        return

    print(f"\nSuccessfully processed {N} total entries (from {len(df)} original rows).")

    # --- Initialize Arrays ---
    z_all = np.zeros((N, MAX_ATOMS), dtype=np.int32)
    pos_all = np.zeros((N, MAX_ATOMS, 3), dtype=np.float32)
    z_solv_all = np.zeros((N, MAX_ATOMS), dtype=np.int32)
    pos_solv_all = np.zeros((N, MAX_ATOMS, 3), dtype=np.float32)
    
    phi_all = np.zeros((N,), dtype=np.float32)
    lmax_all = np.zeros((N,), dtype=np.float32)
    mol_ids = np.zeros((N,), dtype=np.int32)
    smiles_list = []

    print("Packing data into arrays...")
    for i, item in enumerate(tqdm(valid_data)):
        # Solute
        mol = item['solute_mol']
        conf_id = item['solute_conf_id']
        n = min(mol.GetNumAtoms(), MAX_ATOMS)
        
        z_vals = [a.GetAtomicNum() for a in mol.GetAtoms()]
        z_all[i, :n] = z_vals[:n]
        
        # Get positions for the SPECIFIC conformer ID
        pos = mol.GetConformer(conf_id).GetPositions()
        pos_all[i, :n, :] = pos[:n, :]
        
        # Solvent (Using default conf 0, as we only generated 1 for solvent)
        s_mol = item['solvent_mol']
        sn = min(s_mol.GetNumAtoms(), MAX_ATOMS)
        sz_vals = [a.GetAtomicNum() for a in s_mol.GetAtoms()]
        z_solv_all[i, :sn] = sz_vals[:sn]
        pos_solv_all[i, :sn, :] = s_mol.GetConformer(0).GetPositions()[:sn, :]
        
        # Metadata
        phi_all[i] = item['phi_delta']
        lmax_all[i] = item['lambda_max']
        mol_ids[i] = item['mol_id']
        smiles_list.append(item['smiles'])

    # --- Save HDF5 ---
    os.makedirs(os.path.dirname(OUTPUT_HDF5), exist_ok=True)
    dt_str = h5py.special_dtype(vlen=str)
    
    with h5py.File(OUTPUT_HDF5, "w") as f:
        f.create_dataset("atomic_numbers", data=z_all, compression="gzip")
        f.create_dataset("geometries", data=pos_all, compression="gzip")
        f.create_dataset("solvent_atomic_numbers", data=z_solv_all, compression="gzip")
        f.create_dataset("solvent_geometries", data=pos_solv_all, compression="gzip")
        f.create_dataset("lambda_max", data=lmax_all)
        f.create_dataset("phi_delta", data=phi_all)
        f.create_dataset("mol_ids", data=mol_ids)
        ds_smiles = f.create_dataset("smiles", (N,), dtype=dt_str)
        ds_smiles[:] = smiles_list

    print(f"Success! HDF5 saved to {OUTPUT_HDF5}")

if __name__ == "__main__":
    build_phi_hdf5()