import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import h5py
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

try:
    from model.mana_model import MANA
except ImportError:
    print("Error: Could not import MANA. Make sure you are running this from the scripts/ directory.")
    sys.exit(1)

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class PairData(Data):
    """Custom PyG Data object for Solute + Solvent graphs."""
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s': return self.x_s.size(0)
        if key == 'batch_s': return 1 
        return super().__inc__(key, value, *args, **kwargs)

# -----------------------------------------------------------------------------
# DATASET CLASSES
# -----------------------------------------------------------------------------
class HDF5Dataset(torch.utils.data.Dataset):
    """Loads pre-computed structures directly from HDF5."""
    def __init__(self, h5_path, cutoff_radius=5.0):
        super().__init__()
        self.cutoff_radius = cutoff_radius
        self.atom_to_index = {i: i for i in range(1, 119)}
        
        print(f"Loading HDF5 data from {h5_path}...")
        with h5py.File(h5_path, 'r') as f:
            self.atomic_numbers = f['atomic_numbers'][:]
            self.geometries = f['geometries'][:]
            self.solvent_atomic_numbers = f['solvent_atomic_numbers'][:]
            self.solvent_geometries = f['solvent_geometries'][:]
            self.lambda_max = f['lambda_max'][:]
            self.phi_delta = f['phi_delta'][:]
            
            # Robust string decoding
            self.smiles = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in f['smiles'][:]]
            
            if 'solvent_smiles' in f:
                self.solvent_smiles = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in f['solvent_smiles'][:]]
            else:
                self.solvent_smiles = ["Unknown"] * len(self.smiles)

    def _tensor_from_arrays(self, z_arr, pos_arr):
        mask = z_arr > 0
        z_arr = z_arr[mask]
        pos_arr = pos_arr[mask]
        
        z = torch.tensor([self.atom_to_index.get(a, 0) for a in z_arr], dtype=torch.long)
        pos = torch.tensor(pos_arr, dtype=torch.float32)

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

    def __len__(self): return len(self.atomic_numbers)

    def __getitem__(self, idx):
        z, pos, edge_index, edge_attr = self._tensor_from_arrays(self.atomic_numbers[idx], self.geometries[idx])
        z_s, pos_s, edge_index_s, edge_attr_s = self._tensor_from_arrays(self.solvent_atomic_numbers[idx], self.solvent_geometries[idx])
        batch_s = torch.zeros(z_s.size(0), dtype=torch.long)

        return PairData(
            x=z, pos=pos, edge_index=edge_index, edge_attr=edge_attr,
            x_s=z_s, pos_s=pos_s, edge_index_s=edge_index_s, edge_attr_s=edge_attr_s,
            batch_s=batch_s,
            lambda_max=torch.tensor([float(self.lambda_max[idx])], dtype=torch.float32), 
            phi_delta=torch.tensor([float(self.phi_delta[idx])], dtype=torch.float32),
            mol_id=torch.tensor([idx]), 
            smiles=self.smiles[idx],
            solvent_smiles=self.solvent_smiles[idx]
        )

class CSVDataset(torch.utils.data.Dataset):
    """Generates 3D structures on-the-fly from SMILES CSV."""
    def __init__(self, df, cutoff_radius=5.0, shell_radius=6.0, shell_size=20):
        self.data_list = []
        self.cutoff_radius = cutoff_radius
        self.shell_radius = shell_radius
        self.shell_size = shell_size
        self.atom_to_index = {i: i for i in range(1, 119)} 

        print(f"Generating 3D structures for {len(df)} candidates...")
        
        # Normalize columns for robust matching
        df.columns = df.columns.str.lower().str.strip()
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            smiles = str(row.get('smiles', '')).strip()
            if not smiles or smiles == 'nan': continue

            solvent_smiles = str(row.get('solvent_smiles', 'O')).strip()
            if solvent_smiles.lower() == 'nan' or not solvent_smiles: solvent_smiles = 'O'
            
            # --- Extract Truth (Robust Column Matching) ---
            l_true = 0.0
            for col in ['lambda_max', 'lambda', 'abs_max']:
                if col in row and pd.notna(row[col]):
                    try: l_true = float(row[col]); break
                    except: pass
            
            p_true = 0.0
            for col in ['phi_delta', 'phi', 'quantum_yield']:
                if col in row and pd.notna(row[col]):
                    try: p_true = float(row[col]); break
                    except: pass

            try:
                data = self._process_row(smiles, solvent_smiles, idx, l_true, p_true)
                if data: self.data_list.append(data)
            except Exception:
                pass

    def _smiles_to_3d(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None, None
        mol = Chem.AddHs(mol)
        res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if res != 0: res = AllChem.EmbedMolecule(mol, useRandomCoords=True)
        if res == 0:
            try: AllChem.MMFFOptimizeMolecule(mol)
            except: pass
        z = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        try: pos = mol.GetConformer().GetPositions().astype(np.float32)
        except: return None, None
        return z, pos

    def _generate_solvent_shell(self, solvent_smiles, center_pos):
        mol = Chem.MolFromSmiles(solvent_smiles)
        if mol is None: return None, None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        z_template = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        try: pos_template = mol.GetConformer().GetPositions().astype(np.float32)
        except: return None, None
        
        all_z = []
        all_pos = []
        for _ in range(self.shell_size):
            vec = np.random.randn(3)
            vec /= np.linalg.norm(vec)
            dist = np.random.uniform(2.5, self.shell_radius)
            new_center = center_pos + (vec * dist)
            shift = new_center - pos_template.mean(axis=0)
            all_pos.append(pos_template + shift)
            all_z.append(z_template)
        return np.concatenate(all_z), np.vstack(all_pos)

    def _tensor_from_raw(self, z_raw, pos_raw):
        z = torch.tensor([self.atom_to_index.get(a, 0) for a in z_raw], dtype=torch.long)
        pos = torch.tensor(pos_raw, dtype=torch.float32)
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

    def _process_row(self, smiles, solvent_smiles, idx, l_true, p_true):
        z_raw, pos_raw = self._smiles_to_3d(smiles)
        if z_raw is None: return None
        z, pos, edge_index, edge_attr = self._tensor_from_raw(z_raw, pos_raw)
        
        center_mass = np.mean(pos_raw, axis=0)
        z_s_raw, pos_s_raw = self._generate_solvent_shell(solvent_smiles, center_mass)
        
        if z_s_raw is not None:
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
            mol_id=torch.tensor([idx]), 
            lambda_max=torch.tensor([l_true], dtype=torch.float32), 
            phi_delta=torch.tensor([p_true], dtype=torch.float32),
            smiles=smiles,
            solvent_smiles=solvent_smiles
        )

    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]

def _to_numpy_flat(tensor):
    """Utility: make sure tensor -> 1d numpy array (float)"""
    if isinstance(tensor, np.ndarray):
        return tensor.ravel()
    if tensor is None:
        return np.array([], dtype=float)
    try:
        return tensor.detach().cpu().numpy().ravel()
    except Exception:
        return np.array([], dtype=float)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input file (.h5 or .csv)")
    parser.add_argument("--model", required=True, help="Path to .pth model file")
    parser.add_argument("--output", default="predictions.csv", help="Path to save output CSV")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    # 1. Select Dataset Loader
    input_path = Path(args.input)
    if input_path.suffix == '.h5':
        dataset = HDF5Dataset(str(input_path))
    else:
        print("CSV input detected.") 
        df = pd.read_csv(args.input)
        dataset = CSVDataset(df)

    if len(dataset) == 0:
        print("No valid molecules found.")
        return
        
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 2. Load Model
    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model}...")
    model = MANA(num_atom_types=118, hidden_dim=128, tasks=["lambda", "phi"]).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device), strict=False)
    model.eval()

    # 3. Predict
    results = []
    print("Running predictions...")
    
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)
            preds = model(batch)
            
            l_pred = _to_numpy_flat(preds.get("lambda"))
            p_pred = _to_numpy_flat(preds.get("phi"))

            results.append({
                "smiles": batch.smiles[0],
                "solvent_smiles": batch.solvent_smiles[0],
                "lambda_true": _to_numpy_flat(batch.lambda_max), # Saved from Dataset
                "lambda_pred": l_pred,
                "phi_true": _to_numpy_flat(batch.phi_delta),     # Saved from Dataset
                "phi_pred": p_pred,
            })

    # 4. Save
    res_df = pd.DataFrame(results)
    res_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")

if __name__ == "__main__":
    main()