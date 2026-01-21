import os
import h5py
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# --- Configuration ---
INPUT_CSV = "data/fluorescence_dataset.csv"
OUTPUT_HDF5 = "data/fluor_data.h5"
MAX_ATOMS = None
NUM_CONFS = 1 

def build_phi_hdf5():
    print(f"Reading {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # --- FIX 1: Strip whitespace from headers ---
    # This fixes the " lambda_max" vs "lambda_max" error
    df.columns = df.columns.str.strip()
    
    # Verify columns exist
    required_cols = ['Chromophore', 'Quantum Yield', 'dielectric']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing columns in CSV: {missing}")
        print(f"Found columns: {df.columns.tolist()}")
        return

    # Filter valid rows (ensure we have structure and target)
    # We allow lambda_max to be missing (NaN)
    df = df.dropna(subset=['Chromophore', 'Quantum Yield'])
    
    print(f"Found {len(df)} molecules to process.")
    print(f"Generating {NUM_CONFS} conformers per molecule...")

    valid_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # 1. Validate Target
            # Ensure it's a number and valid probability (0 to 1.0)
            phi_val = pd.to_numeric(row['Quantum Yield'], errors='coerce')
            if pd.isna(phi_val) or phi_val > 1.0 or phi_val < 0.0:
                # print(f"Skipping row {idx}: Invalid QY {phi_val}")
                continue
            
            # 2. Parse Molecule
            smi = row['Chromophore']
            mol = Chem.MolFromSmiles(smi) 
            if mol is None: 
                print(f"Skipping row {idx}: Invalid SMILES {smi}")
                continue
            mol = Chem.AddHs(mol) 
            
            # 3. Generate Conformers
            conf_ids = AllChem.EmbedMultipleConfs(
                mol, 
                numConfs=NUM_CONFS, 
                randomSeed=42, 
                pruneRmsThresh=0.5,
                useRandomCoords=True
            )
            
            # Fallback
            if len(conf_ids) == 0:
                conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=NUM_CONFS, useRandomCoords=True)

            if len(conf_ids) == 0:
                print(f"Skipping row {idx}: Embedding failed.")
                continue

            # 4. Handle Lambda Max (Optional column)
            lmax = np.nan
            if 'lambda_max' in row and not pd.isna(row['lambda_max']):
                try:
                    lmax = float(row['lambda_max'])
                except:
                    pass

            # 5. Store Data
            for conf_id in conf_ids:
                # Optimize (optional)
                try: AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
                except: pass

                valid_data.append({
                    "mol": mol,
                    "conf_id": conf_id,
                    "phi_delta": float(phi_val),
                    "lambda_max": lmax,
                    "dielectric": float(row.get('dielectric', 0.0)),
                    "smiles": str(smi),
                    "mol_id": idx
                })
        
        # --- FIX 2: Print errors instead of silent pass ---
        except Exception as e:
            print(f"CRASH on row {idx}: {e}")
            continue

    # Determine Array Sizes
    N = len(valid_data)
    if N == 0:
        print("\nError: No valid molecules generated.")
        return

    # Find max atoms for padding
    max_atoms_found = max(item['mol'].GetNumAtoms() for item in valid_data)
    n_atoms_dim = MAX_ATOMS if MAX_ATOMS else max_atoms_found
    
    print(f"\nDataset successfully generated: {N} samples.")
    print(f"Tensor Width: {n_atoms_dim} atoms")

    # Initialize Arrays
    z_all = np.zeros((N, n_atoms_dim), dtype=np.int32)
    pos_all = np.zeros((N, n_atoms_dim, 3), dtype=np.float32)
    phi_all = np.zeros((N,), dtype=np.float32)
    dielectric_all = np.zeros((N,), dtype=np.float32) 
    mol_ids_all = np.zeros((N,), dtype=np.int32)
    lmax_all = np.zeros((N,), dtype=np.float32)
    smiles_list = []

    # Fill Arrays
    print("Writing tensors...")
    for i, item in enumerate(valid_data):
        mol = item['mol']
        conf = mol.GetConformer(item['conf_id']) 
        num_atoms = mol.GetNumAtoms()
        
        limit = min(num_atoms, n_atoms_dim)
        
        z_vals = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        z_all[i, :limit] = z_vals[:limit]
        
        pos = conf.GetPositions()
        pos_all[i, :limit, :] = pos[:limit, :]
        
        lmax_all[i] = item['lambda_max']
        phi_all[i] = item['phi_delta']
        dielectric_all[i] = item['dielectric']
        mol_ids_all[i] = item['mol_id']
        smiles_list.append(item['smiles'])

    # Save to HDF5
    os.makedirs(os.path.dirname(OUTPUT_HDF5), exist_ok=True)
    dt_str = h5py.special_dtype(vlen=str)
    
    with h5py.File(OUTPUT_HDF5, "w") as f:
        f.create_dataset("atomic_numbers", data=z_all)
        f.create_dataset("geometries", data=pos_all)
        f.create_dataset("lambda_max", data=lmax_all)
        f.create_dataset("phi_delta", data=phi_all)
        f.create_dataset("mol_ids", data=mol_ids_all)
        f.create_dataset("dielectric", data=dielectric_all)
        ds_smiles = f.create_dataset("smiles", (N,), dtype=dt_str)
        ds_smiles[:] = smiles_list

    print(f"Success! Saved to {OUTPUT_HDF5}")

if __name__ == "__main__":
    build_phi_hdf5()