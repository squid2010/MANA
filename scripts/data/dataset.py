import h5py
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


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
    ):
        super().__init__()

        with h5py.File(hdf5_file, "r") as f:
            self.atomic_numbers = f["atomic_numbers"][()] # pyright: ignore[reportIndexIssue]
            self.positions = f["geometries"][()] # pyright: ignore[reportIndexIssue]
            self.lambda_max = f["lambda_max"][()] # pyright: ignore[reportIndexIssue]
            self.phi_delta = f["phi_delta"][()] # pyright: ignore[reportIndexIssue]
            self.mol_ids = f["mol_ids"][()] # pyright: ignore[reportIndexIssue]
            self.smiles = f["smiles"][()] # pyright: ignore[reportIndexIssue]

        unique_atoms = set()
        for z in self.atomic_numbers: # pyright: ignore[reportGeneralTypeIssues]
            unique_atoms.update(z[z > 0])

        self.unique_atoms = sorted(unique_atoms)
        self.atom_to_index = {a: i for i, a in enumerate(self.unique_atoms)}
        self.num_atom_types = len(self.unique_atoms)

        self.cutoff_radius = cutoff_radius
        self.batch_size = batch_size

        self.n_structures = self.atomic_numbers.shape[0] # pyright: ignore[reportAttributeAccessIssue]

        np.random.seed(random_seed)
        idx = np.random.permutation(self.n_structures)

        n_train = int(train_split * self.n_structures)
        n_val = int(val_split * self.n_structures)

        self.train_indices = idx[:n_train]
        self.val_indices = idx[n_train : n_train + n_val]
        self.test_indices = idx[n_train + n_val :]


    def get(self, idx):
        z_raw = self.atomic_numbers[idx] # pyright: ignore[reportIndexIssue]
        pos = torch.tensor(self.positions[idx], dtype=torch.float32) # pyright: ignore[reportIndexIssue]

        atom_mask = torch.tensor(z_raw > 0) # pyright: ignore[reportOperatorIssue]
        z = torch.tensor(
            [self.atom_to_index[a] if a > 0 else 0 for a in z_raw], # pyright: ignore[reportGeneralTypeIssues]
            dtype=torch.long,
        )

        edge_index, edge_attr = self._create_edges(pos, atom_mask)

        return Data(
            x=z,
            pos=pos,
            edge_index=edge_index,
            edge_attr=edge_attr,
            atom_mask=atom_mask,
            lambda_max=torch.tensor(self.lambda_max[idx], dtype=torch.float32), # pyright: ignore[reportIndexIssue]
            phi_delta=torch.tensor(self.phi_delta[idx], dtype=torch.float32), # pyright: ignore[reportIndexIssue]
            idx=idx,
        )

    def len(self):
        return self.n_structures

    def get_dataloaders(self, num_workers=0):
        return (
            DataLoader(
                GeometricSubset(self, self.train_indices), # pyright: ignore[reportArgumentType]
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=num_workers,
            ),
            DataLoader(
                GeometricSubset(self, self.val_indices), # pyright: ignore[reportArgumentType]
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
            DataLoader(
                GeometricSubset(self, self.test_indices), # pyright: ignore[reportArgumentType]
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=num_workers,
            ),
        )

    def _create_edges(self, pos, atom_mask):
        real_idx = torch.where(atom_mask)[0]
        real_pos = pos[atom_mask]

        if real_pos.size(0) == 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 4))

        dist = torch.cdist(real_pos, real_pos)
        mask = (dist < self.cutoff_radius) & (dist > 0)

        row, col = mask.nonzero(as_tuple=True)
        edge_index = torch.stack(
            [real_idx[row], real_idx[col]], dim=0
        )

        diff = pos[edge_index[1]] - pos[edge_index[0]]
        d = torch.norm(diff, dim=1, keepdim=True)
        u = diff / (d + 1e-8)

        edge_attr = torch.cat([d, u], dim=1)
        return edge_index, edge_attr
