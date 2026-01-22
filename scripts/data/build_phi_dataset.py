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
NUM_CONFS = 1  

# --- Solvent Mapping (Name -> SMILES) ---
SOLVENT_TO_SMILES = {
    # --- Alcohols ---
    "MeOH": "CO",
    "methanol": "CO",
    "MeOD": "CO",       # Deuterated methanol -> Standard
    "CD3OD": "CO",      # Deuterated methanol -> Standard
    "EtOH": "CCO",
    "ethanol": "CCO",
    "1-PrOH": "CCCO",
    "n-propanol": "CCCO",
    "2-PrOH": "CC(O)C",
    "i-PrOH": "CC(O)C",
    "isopropanol": "CC(O)C",
    "1-BuOH": "CCCCO",
    "n-butanol": "CCCCO",
    "iso-BuOH": "CC(C)CO",
    "tert-BuOH": "CC(C)(O)C",
    "c-C6H11OH": "OC1CCCCC1", # Cyclohexanol
    "C6H5CH2OH": "OCc1ccccc1", # Benzyl alcohol
    "2-Methoxyethanol": "COCCO",
    "m-Cresol": "Cc1cccc(O)c1",
    "i-C5H11OH": "CCC(C)CO", # Isoamyl alcohol (approx)
    
    # --- Alkanes / Cycloalkanes ---
    "hexane": "CCCCCC",
    "Hexane": "CCCCCC",
    "n-C6H14": "CCCCCC",
    "Hexanes": "CCCCCC", # Treat mixture as n-hexane
    "cyclohexane": "C1CCCCC1",
    "c-C6H12": "C1CCCCC1",
    "heptane": "CCCCCCC",
    "n-C7H16": "CCCCCCC",
    "pentane": "CCCCC",
    "n-C5H12": "CCCCC",
    "i-octane": "CC(C)CC(C)(C)C", # Isooctane
    
    # --- Chlorinated / Fluorinated ---
    "CCl4": "ClC(Cl)(Cl)Cl",
    "CH2Cl2": "ClCCl",
    "DCM": "ClCCl",
    "CHCl3": "ClC(Cl)Cl",
    "CDCl3": "ClC(Cl)Cl", # Deuterated chloroform -> Standard
    "C6H5Cl": "Clc1ccccc1", # Chlorobenzene
    "C6H5F": "Fc1ccccc1",   # Fluorobenzene
    "C6F6": "Fc1c(F)c(F)c(F)c(F)c1F", # Hexafluorobenzene
    "ClCF2CCl2F": "FC(F)(Cl)C(F)(Cl)Cl", # Freon 113
    "n-C3H7I": "CCC1", # 1-Iodopropane (Solvent/Reagent)
    "C6H5Br": "Brc1ccccc1", # Bromobenzene
    "C6D5Br": "Brc1ccccc1", # Deuterated bromobenzene

    # --- Aromatics ---
    "toluene": "Cc1ccccc1",
    "C6H5CH3": "Cc1ccccc1",
    "benzene": "c1ccccc1",
    "C6H6": "c1ccccc1",
    "C6D6": "c1ccccc1", # Deuterated benzene

    # --- Ethers / Esters / Ketones ---
    "THF": "C1CCOC1",
    "c-C4H8O": "C1CCOC1",
    "diox": "C1COCCO1",
    "dioxane": "C1COCCO1",
    "(C2H5)2O": "CCOCC", # Diethyl ether
    "EtOAc": "CC(=O)OCC",
    "acetone": "CC(=O)C",
    "CH3COCH3": "CC(=O)C",
    "2-Propanedione": "CC(=O)C=O", # Methylglyoxal? Usually Acetone is intended but 2-propanedione is distinct.
    "pinacolone": "CC(=O)C(C)(C)C",
    "Propylene carbonate": "CC1COC(=O)O1",
    
    # --- Polar Aprotic / Others ---
    "CH3CN": "CC#N",
    "MeCN": "CC#N",
    "C6H5CN": "N#Cc1ccccc1", # Benzonitrile
    "DMSO": "CS(=O)C",
    "DMF": "CN(C)C=O",
    "HCONH2": "NC=O", # Formamide
    "C5H5N": "c1ccncc1", # Pyridine
    "CS2": "S=C=S", # Carbon disulfide
    
    # --- Water / Aqueous ---
    "water": "O",
    "H2O": "O",
    "D2O": "O",
    
    # Note: Micelles (mic) and Vesicles (ves) are essentially aqueous environments 
    # for the purpose of a solvent graph, though physically complex.
    # Mapping them to water is the safest baseline approximation.
    "H2O (mic)": "O",
    "D2O (mic)": "O",
    "H2O (ves)": "O",
    "D2O (ves)": "O",
    "air": "N", # Vacuum/Nitrogen? Using Nitrogen as placeholder or skip
    "air (mic)": "N",
}

def embed_mol(mol):
    """Robust 3D embedding helper"""
    mol = Chem.AddHs(mol)
    res = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
    if res != 0:
        res = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True, clearConfs=True)
    
    if res == 0:
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except:
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

    # Filter
    df = df.dropna(subset=['SMILES', 'PhiDelta', 'Solvent'])
    
    print(f"Found {len(df)} samples.")
    print("Mapping solvent names to structures...")

    valid_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # 1. Process Solute
            solute_smi = row['SMILES']
            solute_mol = Chem.MolFromSmiles(solute_smi)
            if solute_mol is None: continue
            
            solute_mol, res = embed_mol(solute_mol)
            if res != 0: continue 
            
            # 2. Process Solvent
            solv_name = str(row['Solvent']).strip()
            
            # Ignore headers/garbage
            if solv_name.lower() in ["solvent", "none", "nan", ""]:
                continue

            # Lookup
            solv_smi = SOLVENT_TO_SMILES.get(solv_name)
            
            # Direct SMILES fallback
            if solv_smi is None:
                if Chem.MolFromSmiles(solv_name) is not None:
                    solv_smi = solv_name
                else:
                    # Treat mixtures (e.g. "D2O/EtOH (90:10)") as the major component if possible
                    if "/" in solv_name:
                        major = solv_name.split("/")[0].strip()
                        solv_smi = SOLVENT_TO_SMILES.get(major)
                    
                    if solv_smi is None:
                        # print(f"Skipping unknown solvent: {solv_name}")
                        continue

            solv_mol = Chem.MolFromSmiles(solv_smi)
            if solv_mol is None: continue
            
            solv_mol, s_res = embed_mol(solv_mol)
            if s_res != 0: continue

            valid_data.append({
                "solute_mol": solute_mol,
                "solvent_mol": solv_mol,
                "phi_delta": float(row['PhiDelta']),
                "lambda_max": float(row['lambda_max']) if 'lambda_max' in row and pd.notna(row['lambda_max']) else np.nan,
                "mol_id": idx,
                "smiles": solute_smi,
                "solvent_name": solv_name
            })
            
        except Exception as e:
            pass

    N = len(valid_data)
    if N == 0: 
        print("Error: No valid molecules generated.")
        return

    print(f"\nSuccessfully processed {N} samples.")

    # Init Arrays
    z_all = np.zeros((N, MAX_ATOMS), dtype=np.int32)
    pos_all = np.zeros((N, MAX_ATOMS, 3), dtype=np.float32)
    z_solv_all = np.zeros((N, MAX_ATOMS), dtype=np.int32)
    pos_solv_all = np.zeros((N, MAX_ATOMS, 3), dtype=np.float32)
    
    phi_all = np.zeros((N,), dtype=np.float32)
    lmax_all = np.zeros((N,), dtype=np.float32)
    mol_ids = np.zeros((N,), dtype=np.int32)
    smiles_list = []

    for i, item in enumerate(valid_data):
        # Solute
        mol = item['solute_mol']
        n = min(mol.GetNumAtoms(), MAX_ATOMS)
        z_vals = [a.GetAtomicNum() for a in mol.GetAtoms()]
        z_all[i, :n] = z_vals[:n]
        pos_all[i, :n, :] = mol.GetConformer().GetPositions()[:n, :]
        
        # Solvent
        s_mol = item['solvent_mol']
        sn = min(s_mol.GetNumAtoms(), MAX_ATOMS)
        sz_vals = [a.GetAtomicNum() for a in s_mol.GetAtoms()]
        z_solv_all[i, :sn] = sz_vals[:sn]
        pos_solv_all[i, :sn, :] = s_mol.GetConformer().GetPositions()[:sn, :]
        
        phi_all[i] = item['phi_delta']
        lmax_all[i] = item['lambda_max']
        mol_ids[i] = item['mol_id']
        smiles_list.append(item['smiles'])

    os.makedirs(os.path.dirname(OUTPUT_HDF5), exist_ok=True)
    dt_str = h5py.special_dtype(vlen=str)
    
    with h5py.File(OUTPUT_HDF5, "w") as f:
        f.create_dataset("atomic_numbers", data=z_all)
        f.create_dataset("geometries", data=pos_all)
        f.create_dataset("solvent_atomic_numbers", data=z_solv_all)
        f.create_dataset("solvent_geometries", data=pos_solv_all)
        f.create_dataset("lambda_max", data=lmax_all)
        f.create_dataset("phi_delta", data=phi_all)
        f.create_dataset("mol_ids", data=mol_ids)
        ds_smiles = f.create_dataset("smiles", (N,), dtype=dt_str)
        ds_smiles[:] = smiles_list

    print(f"Success! HDF5 saved to {OUTPUT_HDF5}")

if __name__ == "__main__":
    build_phi_hdf5()