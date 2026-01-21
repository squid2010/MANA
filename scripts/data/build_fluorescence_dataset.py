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
NUM_CONFS = 1  # <--- AUGMENTATION FACTOR (1 Data)

def build_phi_hdf5():
    print(f"Reading {INPUT_CSV}...")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # Filter valid rows
    df = df[df.dropna(subset=['Chromophore', 'Solvent', 'Quantum Yield', 'dielectric'])['dielectric'] != 0]
    print(f"Found {len(df)} unique molecules.")
    print(f"Generating {NUM_CONFS} conformers per molecule...")

    valid_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Skip phi values that aren't valid
        if float(row['Quantum Yield']) > 1:
            continue
            
        smi = row['Chromophore']
        try:
            mol = Chem.MolFromSmiles(smi) #pyright: ignore[reportAttributeAccessIssue]
            if mol is None: 
                continue
            mol = Chem.AddHs(mol) #pyright: ignore[reportAttributeAccessIssue]
            
            # --- AUGMENTATION: Generate Multiple Conformers ---
            # pruneRmsThresh=0.5 removes conformers that look identical
            conf_ids = AllChem.EmbedMultipleConfs( #pyright: ignore[reportAttributeAccessIssue]
                mol, 
                numConfs=NUM_CONFS, 
                randomSeed=42, 
                pruneRmsThresh=0.5,
                useRandomCoords=True
            )
            
            # Fallback if embedding fails
            if len(conf_ids) == 0:
                conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=NUM_CONFS, useRandomCoords=True) #pyright:ignore[reportAttributeAccessIssue]

            # Process each conformer as a separate sample
            for conf_id in conf_ids:
                # Optional optimization (skip if it fails)
                try: 
                    AllChem.MMFFOptimizeMolecule(mol, confId=conf_id) #pyright: ignore[reportAttributeAccessIssue]
                except Exception as _:
                    pass

                valid_data.append({
                    "mol": mol,
                    "conf_id": conf_id, # Track which conformer this is
                    "phi_delta": float(row['Quantum Yield']),
                    "lambda_max": float(row['lambda_max']) if not pd.isna(row['lambda_max']) else np.nan, #pyright: ignore[reportGeneralTypeIssues]
                    "dielectric": float(row['dielectric']),
                    "smiles": str(smi),
                    "mol_id": idx # CRITICAL: This links augmentations back to original molecule
                })
            
        except Exception as _:
            pass

    # Determine Array Sizes
    N = len(valid_data)
    if N == 0:
        print("Error: No valid molecules generated.")
        return

    # Find max atoms for padding
    max_atoms_found = max(item['mol'].GetNumAtoms() for item in valid_data)
    n_atoms_dim = MAX_ATOMS if MAX_ATOMS else max_atoms_found
    
    print(f"\ndataset augmented to {N} samples (from {len(df)} originals).")
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
        # Retrieve specific conformer by ID
        conf = mol.GetConformer(item['conf_id']) 
        num_atoms = mol.GetNumAtoms()
        
        limit = min(num_atoms, n_atoms_dim)
        
        # Atomic Numbers
        z_vals = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        z_all[i, :limit] = z_vals[:limit]
        
        # Coordinates
        pos = conf.GetPositions()
        pos_all[i, :limit, :] = pos[:limit, :]
        
        # Metadata / Targets
        lmax_all[i] = item['lambda_max']
        phi_all[i] = item['phi_delta']
        dielectric_all[i] = item['dielectric']
        mol_ids_all[i] = item['mol_id']
        smiles_list.append(item['smiles'])

    # 5. Save to HDF5
    # Ensure directory exists
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

    print(f"Success! Augmented HDF5 saved to {OUTPUT_HDF5}")

if __name__ == "__main__":
    build_phi_hdf5()