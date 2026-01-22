import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

class PairData(Data):
    """
    Custom PyG Data object to handle two disjoint graphs (Solute + Solvent).
    This tells the DataLoader how to increment indices when stacking batches.
    """
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            # Increment solvent edge indices by the number of solvent nodes in the batch so far
            return self.x_s.size(0)
        if key == 'batch_s':
            # Increment the graph index for the solvent batch vector
            return 1 
        return super().__inc__(key, value, *args, **kwargs)

class GeometricSubset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

class DatasetConstructor(Dataset):
    def __init__(
        self,
        hdf5_file,
        cutoff_radius=5.0,
        batch_size=32,
        train_split=0.8,
        val_split=0.1,
        random_seed=42,
        num_atom_types=None,
        split_by_mol_id=False,
    ):
        super().__init__()
        
        self.cutoff_radius = cutoff_radius
        self.batch_size = batch_size

        print(f"Loading data from {hdf5_file}...")
        with h5py.File(hdf5_file, "r") as f:
            # Solute Data
            self.atomic_numbers = f["atomic_numbers"][()]
            self.positions = f["geometries"][()]
            
            # Solvent Data (Optional, but required for Phi models)
            if "solvent_atomic_numbers" in f:
                self.solvent_atomic_numbers = f["solvent_atomic_numbers"][()]
                self.solvent_positions = f["solvent_geometries"][()]
                self.has_solvent = True
            else:
                self.has_solvent = False

            # Targets
            self.lambda_max = f["lambda_max"][()]
            self.phi_delta = f["phi_delta"][()]
            self.mol_ids = f["mol_ids"][()]
            
            raw_smiles = f["smiles"][()]
            self.smiles = [s.decode("utf-8") if isinstance(s, bytes) else s for s in raw_smiles]

        # Build Vocabulary (Unified for both solute and solvent)
        unique_atoms = set()
        for z in self.atomic_numbers:
            unique_atoms.update(z[z > 0])
        
        if self.has_solvent:
            for z in self.solvent_atomic_numbers:
                unique_atoms.update(z[z > 0])

        self.unique_atoms = sorted(list(unique_atoms))
        self.atom_to_index = {a: i + 1 for i, a in enumerate(self.unique_atoms)}
        
        self.n_structures = self.atomic_numbers.shape[0]

        # PRE-COMPUTE GRAPHS
        print(f"Pre-processing {self.n_structures} graphs...")
        self.data_list = []
        for idx in tqdm(range(self.n_structures)):
            self.data_list.append(self._process_one(idx))

        # Create Splits
        np.random.seed(random_seed)
        if split_by_mol_id:
            unique_mol_ids = np.unique(self.mol_ids)
            np.random.shuffle(unique_mol_ids)
            n_mol_train = int(train_split * len(unique_mol_ids))
            n_mol_val = int(val_split * len(unique_mol_ids))
            
            train_ids = set(unique_mol_ids[:n_mol_train])
            val_ids = set(unique_mol_ids[n_mol_train : n_mol_train + n_mol_val])
            
            self.train_indices = [i for i in range(self.n_structures) if self.mol_ids[i] in train_ids]
            self.val_indices = [i for i in range(self.n_structures) if self.mol_ids[i] in val_ids]
            self.test_indices = [i for i in range(self.n_structures) if self.mol_ids[i] not in train_ids and self.mol_ids[i] not in val_ids]
        else:
            idx = np.random.permutation(self.n_structures)
            n_train = int(train_split * self.n_structures)
            n_val = int(val_split * self.n_structures)
            self.train_indices = idx[:n_train]
            self.val_indices = idx[n_train : n_train + n_val]
            self.test_indices = idx[n_train + n_val :]

        # Stats
        train_lambda = self.lambda_max[self.train_indices]
        self.lambda_mean = np.mean(train_lambda)
        self.lambda_std = np.std(train_lambda)

    def _tensor_from_raw(self, z_raw, pos_raw):
        """Helper to create graph tensors from raw arrays"""
        z = torch.tensor([self.atom_to_index.get(a, 0) for a in z_raw], dtype=torch.long)
        pos = torch.tensor(pos_raw, dtype=torch.float32)
        
        # Remove padding
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

    def _process_one(self, idx):
        # 1. Process Solute
        z, pos, edge_index, edge_attr = self._tensor_from_raw(
            self.atomic_numbers[idx], self.positions[idx]
        )

        # 2. Process Solvent (if available)
        if self.has_solvent:
            z_s, pos_s, edge_index_s, edge_attr_s = self._tensor_from_raw(
                self.solvent_atomic_numbers[idx], self.solvent_positions[idx]
            )
            # Create a batch vector for the solvent (all zeros for a single graph)
            # DataLoader will stack these. PairData.__inc__ handles the graph index increment.
            batch_s = torch.zeros(z_s.size(0), dtype=torch.long)
        else:
            # Dummy solvent data to prevent crashes if loading non-solvent datasets
            z_s = torch.tensor([], dtype=torch.long)
            pos_s = torch.tensor([], dtype=torch.float32)
            edge_index_s = torch.empty((2, 0), dtype=torch.long)
            edge_attr_s = torch.empty((0, 4))
            batch_s = torch.tensor([], dtype=torch.long)

        # 3. Create PairData Object
        return PairData(
            # Solute
            x=z,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            
            # Solvent (Suffix _s)
            x_s=z_s,
            pos_s=pos_s,
            edge_index_s=edge_index_s,
            edge_attr_s=edge_attr_s,
            batch_s=batch_s,

            # Targets
            lambda_max=torch.tensor([self.lambda_max[idx]], dtype=torch.float32),
            phi_delta=torch.tensor([self.phi_delta[idx]], dtype=torch.float32),
            mol_id=torch.tensor([self.mol_ids[idx]], dtype=torch.int32),
            smiles=self.smiles[idx],
        )

    def len(self):
        return self.n_structures

    def get(self, idx):
        return self.data_list[idx]
    
    def __getitem__(self, idx):
        return self.data_list[idx]

    def get_dataloaders(self, num_workers=0):
        return (
            DataLoader(GeometricSubset(self, self.train_indices), batch_size=self.batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(GeometricSubset(self, self.val_indices), batch_size=self.batch_size, shuffle=False, num_workers=num_workers),
            DataLoader(GeometricSubset(self, self.test_indices), batch_size=self.batch_size, shuffle=False, num_workers=num_workers),
        )