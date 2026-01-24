import os
import h5py
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# --- Configuration ---
INPUT_CSV = "data/lambda/lambdamax_dataset.csv"
OUTPUT_H5 = "data/pretrain/full_dataset.h5"
MAX_ATOMS = 100 

def embed_mol(mol):
    """Robust 3D embedding helper"""
    mol = Chem.AddHs(mol)
    res = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
    if res != 0:
        res = AllChem.EmbedMolecule(mol, randomSeed=42, clearConfs=True)
    if res == 0:
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except:
            pass
    return mol, res

def build_unified_dataset():
    print(f"Reading {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.strip()
    
    # 1. Normalize Columns
    col_map = {
        "Chromophore": "smiles", 
        "Absorption max (nm)": "lambda_max",
        "Quantum yield": "phi",
        "Solvent": "solvent"
    }
    df = df.rename(columns=col_map)

    # 2. Filter Valid Rows (only require smiles, lambda_max, solvent)
    df = df.dropna(subset=['smiles', 'lambda_max', 'solvent'])
    
    all_data = []

    print(f"Processing {len(df)} molecules...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Check lambda_max
            l_max = float(row['lambda_max'])
            
            # Check if QY exists and is valid, otherwise use NaN
            phi_val = np.nan
            if 'phi' in row:
                try:
                    p = float(row['phi'])
                    if not np.isnan(p) and 0.0 <= p <= 1.0:
                        phi_val = p
                except:
                    pass

            # Generate Structure (Solute)
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is None: continue
            mol, res = embed_mol(mol)
            if res != 0: continue

            # Generate Structure (Solvent)
            solv_mol = None
            if pd.notna(row['solvent']):
                s_smi = str(row['solvent'])
                if s_smi.lower() not in ['nan', 'none', 'gas', 'vacuum']:
                    try:
                        temp_s = Chem.MolFromSmiles(s_smi)
                        if temp_s:
                            temp_s, s_res = embed_mol(temp_s)
                            if s_res == 0:
                                solv_mol = temp_s
                    except Exception:
                        pass
            
            # Pack Data
            entry = {
                "mol": mol,
                "solvent_mol": solv_mol,
                "lambda_max": l_max,
                "phi_delta": phi_val,
                "mol_id": idx,
                "smiles": str(row['smiles'])
            }
            
            all_data.append(entry)

        except Exception as e:
            continue

    print(f"\nProcessing Complete.")
    print(f"Total valid entries: {len(all_data)}")
    print(f"  - With quantum yield: {sum(1 for d in all_data if not np.isnan(d['phi_delta']))}")
    print(f"  - Without quantum yield: {sum(1 for d in all_data if np.isnan(d['phi_delta']))}")

    # 3. Save to HDF5
    if not all_data:
        print("No valid data to save.")
        return
        
    N = len(all_data)
    z_all = np.zeros((N, MAX_ATOMS), dtype=np.int32)
    pos_all = np.zeros((N, MAX_ATOMS, 3), dtype=np.float32)
    z_s_all = np.zeros((N, MAX_ATOMS), dtype=np.int32)
    pos_s_all = np.zeros((N, MAX_ATOMS, 3), dtype=np.float32)
    
    l_all = np.zeros((N,), dtype=np.float32)
    p_all = np.full((N,), np.nan, dtype=np.float32)
    id_all = np.zeros((N,), dtype=np.int32)
    smiles_list = []

    for i, item in enumerate(all_data):
        # Solute
        m = item['mol']
        n = min(m.GetNumAtoms(), MAX_ATOMS)
        z_all[i, :n] = [a.GetAtomicNum() for a in m.GetAtoms()][:n]
        pos_all[i, :n, :] = m.GetConformer().GetPositions()[:n, :]
        
        # Solvent
        if item['solvent_mol']:
            s = item['solvent_mol']
            sn = min(s.GetNumAtoms(), MAX_ATOMS)
            z_s_all[i, :sn] = [a.GetAtomicNum() for a in s.GetAtoms()][:sn]
            pos_s_all[i, :sn, :] = s.GetConformer().GetPositions()[:sn, :]
        
        l_all[i] = item['lambda_max']
        p_all[i] = item['phi_delta']
        id_all[i] = item['mol_id']
        smiles_list.append(item['smiles'])

    os.makedirs(os.path.dirname(OUTPUT_H5), exist_ok=True)
    dt_str = h5py.special_dtype(vlen=str)
    
    with h5py.File(OUTPUT_H5, "w") as f:
        f.create_dataset("atomic_numbers", data=z_all)
        f.create_dataset("geometries", data=pos_all)
        f.create_dataset("solvent_atomic_numbers", data=z_s_all)
        f.create_dataset("solvent_geometries", data=pos_s_all)
        f.create_dataset("lambda_max", data=l_all)
        f.create_dataset("phi_delta", data=p_all)
        f.create_dataset("mol_ids", data=id_all)
        ds = f.create_dataset("smiles", (N,), dtype=dt_str)
        ds[:] = smiles_list
    
    print(f"Saved {OUTPUT_H5}")

if __name__ == "__main__":
    build_unified_dataset()