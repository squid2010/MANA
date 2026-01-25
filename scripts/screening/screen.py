import argparse
import sys
import os
from pathlib import Path
import h5py

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

# -----------------------------------------------------------------------------
# 1. SETUP PATHS & IMPORTS
# -----------------------------------------------------------------------------
# Add project root to path so we can import model
current_dir = Path(__file__).resolve().parent.parent
project_root = current_dir.parent
sys.path.insert(0, str(current_dir))

from model.mana_model import MANA 
from data.dataset import DatasetConstructor  # Re-use your existing loader logic
from torch_geometric.data import Data

class PairData(Data):
    """Custom PyG Data object for Solute + Solvent graphs."""
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'batch_s':
            return 1 
        return super().__inc__(key, value, *args, **kwargs)

# -----------------------------------------------------------------------------
# 2. CONFIGURATION
# -----------------------------------------------------------------------------
CELLULAR_SOLVENT_SMILES = "O" 
SOLVENT_SHELL_SIZE = 20
SHELL_RADIUS = 6.0

# -----------------------------------------------------------------------------
# 3. GENERATOR DATASET CLASS (SMILES -> 3D)
# -----------------------------------------------------------------------------
class SMILESGeneratorDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, cutoff_radius=5.0):
        self.data_list = []
        self.cutoff_radius = cutoff_radius
        # Identity mapping for screening (1=H, 6=C, etc.)
        self.atom_to_index = {i: i for i in range(1, 119)} 

        print(f"Generating 3D structures & water shells for {len(smiles_list)} candidates...")
        
        for i, smiles in enumerate(tqdm(smiles_list)):
            smiles = smiles.strip()
            if not smiles: continue
            
            try:
                data = self._process_smiles(smiles, idx=i, solvent_smiles=CELLULAR_SOLVENT_SMILES)
                if data:
                    self.data_list.append(data)
            except Exception as e:
                pass

    def _smiles_to_3d(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None, None
        mol = Chem.AddHs(mol)
        res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if res != 0:
            res = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if res == 0:
            try: AllChem.MMFFOptimizeMolecule(mol)
            except: pass

        z = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        pos = mol.GetConformer().GetPositions().astype(np.float32)
        return z, pos

    def _generate_solvent_shell(self, solvent_smiles, center_pos, num_molecules=20, radius=6.0):
        mol = Chem.MolFromSmiles(solvent_smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        z_template = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        pos_template = mol.GetConformer().GetPositions().astype(np.float32)
        
        all_z = []
        all_pos = []
        for _ in range(num_molecules):
            vec = np.random.randn(3)
            vec /= np.linalg.norm(vec)
            dist = np.random.uniform(2.5, radius)
            new_center = center_pos + (vec * dist)
            shift = new_center - pos_template.mean(axis=0)
            all_pos.append(pos_template + shift)
            all_z.append(z_template)

        return np.concatenate(all_z), np.vstack(all_pos)

    def _tensor_from_raw(self, z_raw, pos_raw):
        z = torch.tensor([self.atom_to_index.get(a, 0) for a in z_raw], dtype=torch.long)
        pos = torch.tensor(pos_raw, dtype=torch.float32)
        mask = z > 0
        z = z[mask]
        pos = pos[mask]
        
        if pos.size(0) == 0:
            return z, pos, torch.empty((2, 0), dtype=torch.long), torch.empty((0, 4))

        dist = torch.cdist(pos, pos)
        mask = (dist < self.cutoff_radius) & (dist > 0)
        row, col = mask.nonzero(as_tuple=True)
        edge_index = torch.stack([row, col], dim=0)
        diff = pos[col] - pos[row]
        d = torch.norm(diff, dim=1, keepdim=True)
        u = diff / (d + 1e-8)
        edge_attr = torch.cat([d, u], dim=1)
        return z, pos, edge_index, edge_attr

    def _process_smiles(self, smiles, idx, solvent_smiles=None):
        z_raw, pos_raw = self._smiles_to_3d(smiles)
        if z_raw is None: return None
        z, pos, edge_index, edge_attr = self._tensor_from_raw(z_raw, pos_raw)

        if solvent_smiles:
            center_mass = np.mean(pos_raw, axis=0)
            z_s_raw, pos_s_raw = self._generate_solvent_shell(
                solvent_smiles, center_mass, SOLVENT_SHELL_SIZE, SHELL_RADIUS
            )
            z_s, pos_s, edge_index_s, edge_attr_s = self._tensor_from_raw(z_s_raw, pos_s_raw)
            batch_s = torch.zeros(z_s.size(0), dtype=torch.long)
        else:
            z_s = torch.tensor([], dtype=torch.long)
            pos_s = torch.tensor([], dtype=torch.float32)
            edge_index_s = torch.empty((2, 0), dtype=torch.long)
            edge_attr_s = torch.empty((0, 4))
            batch_s = torch.tensor([], dtype=torch.long)

        return PairData(
            x=z, pos=pos, edge_index=edge_index, edge_attr=edge_attr,
            x_s=z_s, pos_s=pos_s, edge_index_s=edge_index_s, edge_attr_s=edge_attr_s,
            batch_s=batch_s,
            lambda_max=torch.tensor([0.0]), 
            phi_delta=torch.tensor([0.0]),
            mol_id=torch.tensor([idx]), 
            smiles=smiles
        )

    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]

def pad_arrays(array_list, dim_size=1):
    """Pads a list of arrays to the maximum size along the specified dimension."""
    max_len = max([x.shape[0] for x in array_list])
    
    if dim_size == 1: # 1D arrays (Atomic Numbers)
        padded = np.zeros((len(array_list), max_len), dtype=array_list[0].dtype)
        for i, arr in enumerate(array_list):
            padded[i, :arr.shape[0]] = arr
            
    elif dim_size == 3: # 3D arrays (Geometries)
        padded = np.zeros((len(array_list), max_len, 3), dtype=array_list[0].dtype)
        for i, arr in enumerate(array_list):
            padded[i, :arr.shape[0], :] = arr
            
    return padded

# -----------------------------------------------------------------------------
# 4. MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Screen photosensitizers from SMILES (.txt) or H5 (.h5).")
    parser.add_argument("--input", required=True, help="Path to input file (.txt or .h5)")
    parser.add_argument("--model", default="models/phi/best_model.pth", help="Path to trained model")
    parser.add_argument("--output", default="results/screening/screening_results.h5", help="Path to save H5 results")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    args = parser.parse_args()

    # 1. Determine Input Type
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    # 2. Load Dataset
    if input_path.suffix == ".h5":
        print(f"Loading pre-computed structures from {input_path}...")
        # Use your standard DatasetConstructor for H5 files
        # split_by_mol_id=True ensures we treat it as a standard list of molecules
        dataset = DatasetConstructor(str(input_path), split_by_mol_id=True)
        # Note: DatasetConstructor returns 3 loaders (train, val, test)
        # We combine indices or just grab all data via a custom loader if needed.
        # But DatasetConstructor's internal split logic might hide some data in 'train'.
        # A safer bet for screening EXISTING H5 is to just load all indices.
        from data.dataset import GeometricSubset
        all_indices = range(len(dataset))
        # Override get_dataloaders logic essentially:
        from torch_geometric.loader import DataLoader
        dataset_subset = GeometricSubset(dataset, all_indices)
        loader = DataLoader(dataset_subset, batch_size=1, shuffle=False)

    elif input_path.suffix == ".csv":
        print(f"Reading SMILES from CSV {input_path}...")
        df = pd.read_csv(input_path)
        if 'smiles' not in df.columns:
            print("Error: CSV must contain a 'smiles' column.")
            return
        smiles_list = df['smiles'].dropna().astype(str).tolist()
        
        # Use our generator class
        dataset = SMILESGeneratorDataset(smiles_list)
        from torch_geometric.loader import DataLoader
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    else:
        # Assume Text/SMILES input
        print(f"Reading SMILES from {input_path}...")
        with open(input_path, "r") as f:
            content = f.read().strip()
            if "," in content: smiles_list = [s.strip() for s in content.split(",")]
            else: smiles_list = [s.strip() for s in content.splitlines()]
        
        # Use our generator class
        dataset = SMILESGeneratorDataset(smiles_list)
        from torch_geometric.loader import DataLoader
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

    if len(dataset) == 0:
        print("No valid structures found/generated.")
        return

    # 3. Setup Model
    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = project_root / args.model
    
    print("Loading model...")
    model = MANA(num_atom_types=118, hidden_dim=128, tasks=["lambda", "phi"]).to(device)
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model.eval()
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    # 4. Inference & Collection
    storage = {
        "mol_ids": [],
        "smiles": [],
        "atomic_numbers": [],
        "geometries": [],
        "solvent_atomic_numbers": [],
        "solvent_geometries": [],
        "lambda_pred": [],
        "phi_pred": []
    }

    print(f"\nRunning inference on {len(dataset)} molecules...")
    
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)
            preds = model(batch)
            
            # --- Store Predictions ---
            storage["lambda_pred"].append(preds["lambda"].item())
            storage["phi_pred"].append(preds["phi"].item())
            
            # --- Store Metadata ---
            storage["mol_ids"].append(batch.mol_id.item())
            # Handle smiles string list vs tensor depending on source
            s = batch.smiles[0] if isinstance(batch.smiles, list) else batch.smiles
            storage["smiles"].append(s)
            
            # --- Store Structure (Move back to CPU numpy) ---
            storage["atomic_numbers"].append(batch.x.cpu().numpy())
            storage["geometries"].append(batch.pos.cpu().numpy())
            
            # Solvent
            if hasattr(batch, 'x_s') and batch.x_s.numel() > 0:
                storage["solvent_atomic_numbers"].append(batch.x_s.cpu().numpy())
                storage["solvent_geometries"].append(batch.pos_s.cpu().numpy())
            else:
                storage["solvent_atomic_numbers"].append(np.array([], dtype=np.int64))
                storage["solvent_geometries"].append(np.array([], dtype=np.float32).reshape(0,3))

    # 5. Process & Save to H5
    out_path = Path(args.output)
    if out_path.suffix != '.h5': out_path = out_path.with_suffix('.h5')
    os.makedirs(out_path.parent, exist_ok=True)

    print(f"\nPadding arrays and saving to {out_path}...")
    
    # Pad structural arrays (Ragged -> Dense with 0 padding)
    atomic_numbers_padded = pad_arrays(storage["atomic_numbers"], dim_size=1)
    geometries_padded = pad_arrays(storage["geometries"], dim_size=3)
    solv_atomic_numbers_padded = pad_arrays(storage["solvent_atomic_numbers"], dim_size=1)
    solv_geometries_padded = pad_arrays(storage["solvent_geometries"], dim_size=3)

    with h5py.File(out_path, "w") as f:
        # 1. Structure Data
        f.create_dataset("atomic_numbers", data=atomic_numbers_padded)
        f.create_dataset("geometries", data=geometries_padded)
        f.create_dataset("solvent_atomic_numbers", data=solv_atomic_numbers_padded)
        f.create_dataset("solvent_geometries", data=solv_geometries_padded)
        
        # 2. Metadata
        f.create_dataset("mol_ids", data=np.array(storage["mol_ids"], dtype=np.int32))
        
        # SMILES (Variable length strings)
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset("smiles", data=np.array(storage["smiles"], dtype=object), dtype=dt)
        
        # 3. Predictions (Saved as targets so they can be loaded by DatasetConstructor as 'labels')
        f.create_dataset("lambda_max", data=np.array(storage["lambda_pred"], dtype=np.float32))
        f.create_dataset("phi_delta", data=np.array(storage["phi_pred"], dtype=np.float32))

    print(f"✓ Success! Saved {len(storage['mol_ids'])} molecules.")

if __name__ == "__main__":
    main()