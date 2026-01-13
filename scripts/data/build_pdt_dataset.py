import h5py
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import os

# --- Configuration ---
INPUT_CSV = "data/deepforchem.csv"
OUTPUT_HDF5 = "data/deep4chem_data.h5"
MAX_ATOMS = None  # Set to an integer (e.g., 100) to force a fixed size, or None to auto-calculate

def parse_deep4chem_to_hdf5(csv_path, hdf5_path):
    print(f"Reading {csv_path}...")
    
    # 1. Load and Clean Data
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return

    # Rename columns to match our internal logic
    # 'Chromophore' is usually the SMILES column in Deep4Chem
    df = df.rename(columns={
        "Chromophore": "smiles",
        "Absorption max (nm)": "lambda_max",
        "Quantum yield": "phi_delta" # WARNING: This is usually Fluorescence QY in Deep4Chem
    })

    # Drop rows without essential data (SMILES or Target)
    initial_count = len(df)
    df = df.dropna(subset=['smiles', 'lambda_max'])
    
    # Handle Quantum Yield: Fill NaNs with -1 or 0 so the code doesn't break, 
    # but you should mask these during training if they are missing.
    df['phi_delta'] = df['phi_delta'].fillna(np.nan) 

    print(f"Filtered {initial_count} -> {len(df)} rows with valid SMILES and Lambda Max.")

    # 2. Prepare Lists for Processing
    valid_data = []
    max_atom_count_found = 0

    print("Generating 3D conformers (this takes time)...")
    
    # We use tqdm for a progress bar
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row['smiles']
        
        try:
            # Create RDKit Molecule
            mol = Chem.MolFromSmiles(smiles) # pyright: ignore[reportAttributeAccessIssue]
            if mol is None:
                continue
            
            # Add Hydrogens (Critical for 3D geometry) 
            mol = Chem.AddHs(mol) # pyright:ignore[reportAttributeAccessIssue]
            
            # Generate 3D Embedding
            # useRandomCoords=True helps with complex rigid structures typical in dyes
            res = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True) # pyright: ignore[reportAttributeAccessIssue]
            if res != 0:
               continue  # Skip if embedding fails
            
            # Optimize Geometry (Optional, improves quality)
            try:
                AllChem.UFFOptimizeMolecule(mol) # pyright: ignore[reportAttributeAccessIssue]
            except Exception as _:
                pass # If optimization fails, keep the raw embedding

            # Track size for padding
            n_atoms = mol.GetNumAtoms()
            if n_atoms > max_atom_count_found:
                max_atom_count_found = n_atoms

            # Store the object and data
            valid_data.append({
                'mol': mol,
                'lambda_max': float(row['lambda_max']),
                'phi_delta': float(row['phi_delta']),
                'smiles': str(smiles),
                'mol_id': idx
            })

        except Exception as _:
            continue

    # Determine final array shapes
    N_samples = len(valid_data)
    N_atoms = MAX_ATOMS if MAX_ATOMS else max_atom_count_found
    
    print("\nProcessing complete.")
    print(f"Samples: {N_samples}")
    print(f"Max Atoms (Padding size): {N_atoms}")

    # 3. Allocate Numpy Arrays
    # Atomic numbers (Z)
    z_all = np.zeros((N_samples, N_atoms), dtype=np.int32)
    # Cartesian coordinates (R)
    pos_all = np.zeros((N_samples, N_atoms, 3), dtype=np.float32)
    # Targets
    lmax_all = np.zeros((N_samples,), dtype=np.float32)
    phi_all = np.zeros((N_samples,), dtype=np.float32)
    mol_ids_all = np.zeros((N_samples,), dtype=np.int32)
    
    # String storage for SMILES
    smiles_list = []

    # 4. Fill Arrays
    print("Compiling HDF5 arrays...")
    for i, item in enumerate(valid_data):
        mol = item['mol']
        conf = mol.GetConformer()
        n_atoms = mol.GetNumAtoms()
        
        # Safety clip
        limit = min(n_atoms, N_atoms)
        
        # Atomic Numbers
        z_vals = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        z_all[i, :limit] = z_vals[:limit]
        
        # Coordinates
        pos = conf.GetPositions()
        pos_all[i, :limit, :] = pos[:limit, :]
        
        # Metadata / Targets
        lmax_all[i] = item['lambda_max']
        phi_all[i] = item['phi_delta']
        mol_ids_all[i] = item['mol_id']
        smiles_list.append(item['smiles'])

    # 5. Save to HDF5
    # Ensure directory exists
    os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
    
    dt_str = h5py.special_dtype(vlen=str)
    
    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("atomic_numbers", data=z_all)
        f.create_dataset("geometries", data=pos_all)
        f.create_dataset("lambda_max", data=lmax_all)
        f.create_dataset("phi_delta", data=phi_all)
        f.create_dataset("mol_ids", data=mol_ids_all)
        
        ds_smiles = f.create_dataset("smiles", (N_samples,), dtype=dt_str)
        ds_smiles[:] = smiles_list

    print(f"Success! Database saved to: {hdf5_path}")

if __name__ == "__main__":
    parse_deep4chem_to_hdf5(INPUT_CSV, OUTPUT_HDF5)