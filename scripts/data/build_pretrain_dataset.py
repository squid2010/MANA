import os
import h5py
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# --- Configuration ---
INPUT_CSV = "data/lambda/lambdamax_dataset.csv"

# Output 1: Subset with valid Quantum Yield (High Quality)
OUTPUT_FLUOR = "data/fluor/fluorescence_data.h5"
# Output 2: Subset with NaN Quantum Yield (Volume Data)
OUTPUT_LAMBDA = "data/lambda/lambda_only_data.h5"

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

def build_split_datasets():
    print(f"Reading {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    df.columns = df.columns.str.strip()
    
    # 1. Normalize Columns
    # Map common variations to standard names
    col_map = {
        "Chromophore": "smiles", 
        "Absorption max (nm)": "lambda_max",
        "Quantum yield": "phi",
        "Solvent": "solvent"
    }
    df = df.rename(columns=col_map)

    # 2. Filter Valid Rows
    df = df.dropna(subset=['smiles', 'lambda_max', 'solvent'])
    
    # Lists to hold processed data
    fluor_data = []  # Has QY
    lambda_data = [] # No QY

    print(f"Processing {len(df)} molecules...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Check targets
            l_max = float(row['lambda_max'])
            
            # Check if QY exists and is valid
            has_qy = False
            phi_val = np.nan
            if 'phi' in row:
                try:
                    p = float(row['phi'])
                    if not np.isnan(p) and 0.0 <= p <= 1.0:
                        phi_val = p
                        has_qy = True
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
                "phi_delta": phi_val, # Use uniform name for consistency
                "mol_id": idx,
                "smiles": str(row['smiles'])
            }

            # Route to correct list
            if has_qy:
                fluor_data.append(entry)
            else:
                lambda_data.append(entry)

        except Exception as e:
            continue

    print(f"\nProcessing Complete.")
    print(f" -> Fluorescence Subset (Has QY): {len(fluor_data)}")
    print(f" -> Lambda Subset (No QY):      {len(lambda_data)}")

    # 3. Save Function
    def save_hdf5(data_list, filename):
        if not data_list: return
        
        N = len(data_list)
        z_all = np.zeros((N, MAX_ATOMS), dtype=np.int32)
        pos_all = np.zeros((N, MAX_ATOMS, 3), dtype=np.float32)
        z_s_all = np.zeros((N, MAX_ATOMS), dtype=np.int32)
        pos_s_all = np.zeros((N, MAX_ATOMS, 3), dtype=np.float32)
        
        l_all = np.zeros((N,), dtype=np.float32)
        p_all = np.zeros((N,), dtype=np.float32)
        id_all = np.zeros((N,), dtype=np.int32)
        smiles_list = []

        for i, item in enumerate(data_list):
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
            p_all[i] = item['phi_delta'] if not np.isnan(item['phi_delta']) else -1.0
            id_all[i] = item['mol_id']
            smiles_list.append(item['smiles'])

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        dt_str = h5py.special_dtype(vlen=str)
        
        with h5py.File(filename, "w") as f:
            f.create_dataset("atomic_numbers", data=z_all)
            f.create_dataset("geometries", data=pos_all)
            f.create_dataset("solvent_atomic_numbers", data=z_s_all)
            f.create_dataset("solvent_geometries", data=pos_s_all)
            f.create_dataset("lambda_max", data=l_all)
            f.create_dataset("phi_delta", data=p_all)
            f.create_dataset("mol_ids", data=id_all)
            ds = f.create_dataset("smiles", (N,), dtype=dt_str)
            ds[:] = smiles_list
        
        print(f"Saved {filename}")

    # 4. Write Files
    save_hdf5(fluor_data, OUTPUT_FLUOR)
    save_hdf5(lambda_data, OUTPUT_LAMBDA)

if __name__ == "__main__":
    build_split_datasets()